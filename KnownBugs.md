# Known Bugs

Running list of observed-but-not-yet-fixed issues. Add an entry when you spot one
and can't fix it immediately, so it isn't lost.

---

## 1. VkQueue used from two threads during USD scene switch (Vulkan validation error)

**Observed:** 2026-06-08, `skinny-gui --execution-mode wavefront` on Apple M5 Pro
(MoltenVK), while a USD stage finishes baking in the background and the renderer
switches to it.

**Symptom (validation layer):**
```
Validation Error: [ UNASSIGNED-Threading-MultipleThreads-Write ] | MessageID = 0xa05b236e
vkQueueWaitIdle(): THREADING ERROR : object of type VkQueue is simultaneously used
in current thread <A> and thread <B>
Objects: 1
    [0] VkQueue 0x...
```

**Cause (suspected):** the background USD-bake thread and the main render thread
touch the **same `VkQueue`** concurrently — most likely a `vkQueueWaitIdle` (or a
submit) on the bake/streaming thread racing the main-thread render submit. Vulkan
queues are externally synchronized; concurrent use from two threads without a lock
is undefined behaviour (here it only trips the validation layer, but it can corrupt
on some drivers).

**Scope / impact:** validation-layer error only so far; no observed crash or
visible corruption on MoltenVK. Pre-existing — **not** introduced by the
`neural-trainer-backends` change (surfaced incidentally while testing
`--neural-trainer`).

**Likely fix:** serialize all queue access behind a single mutex, or hand the
background bake/streaming work its own queue (or a transfer queue), or marshal the
final `vkQueueWaitIdle`/submit back onto the main thread. Audit every
`vkQueueSubmit` / `vkQueueWaitIdle` / `vkQueuePresentKHR` call site for the
scene-switch / mesh-stream path (`renderer.py` USD streaming + `vk_context` queue
ops).

**Repro:** `skinny-gui --execution-mode wavefront <usd-stage>` and let the
background bake complete → "switching to USD scene" → error fires.

---

## 2. Metal bindless textures can't honour per-texture wrap modes (one shared sampler)

**Observed:** 2026-06-12, `--backend metal` megakernel on Apple M5 Pro, while
chasing the WOOD material's ~11 % brightness in `three_materials_demo.usda`.

**Symptom:** a material/graph texture authored with a non-repeat USD `wrapS` /
`wrapT` (clamp, mirror, border) renders as if it were **repeat** on Metal. The
inverse bug (clamp applied where repeat was wanted) caused the wood overshoot:
`tiledimage` at `uvtiling=4` sampled past v=1 clamped to the edge row, leaving
wood ~11 % bright (per-material M/V≈1.110, rel-MSE≈0.031) — **now fixed** by
making the shared sampler repeat/repeat (commit `7a4b351`), but the underlying
limitation remains for the *other* wrap modes.

**Cause:** design D8 — slang-rhi's Metal backend can't bind a combined
`Sampler2D[]`, so the 120-slot bindless pool (binding 14) is a plain
`Texture2D[]` sampled through **one shared `commonSampler`** (binding 38). Vulkan
gives every `TexturePool` slot its **own** `VkSampler` carrying that texture's
`wrapS`/`wrapT` (`SampledImage` in `vk_compute.py`); Metal has a single sampler
for all 120 slots, so per-texture wrap (and filter/anisotropy) modes are lost. It
is now hard-coded repeat/repeat (`renderer._metal_common_sampler`), the correct
default for tiling material textures and matching the `TexturePool` default, but
it cannot vary per slot.

**Scope / impact:** Metal only; Vulkan is correct. No effect on the common case
(material/UV textures almost always tile → repeat). Only bites a texture that
explicitly wants clamp/mirror/border *and* is sampled outside [0,1] — none in the
current demo assets after the repeat/repeat fix.

**Likely fix:** give the Metal pool a small set of `SamplerState`s (one per
distinct wrap/filter combination in use, ≤ the 16-sampler argument budget) and
have `SamplerTexture2D`/`fetchFlatHitData` index the right one per texture; or
bucket pool slots by sampler and bind a per-bucket sampler. Until then, document
that Metal material textures are repeat/repeat.

**Repro:** author a USD material whose albedo/`tiledimage` texture sets
`inputs:wrapT = "clamp"` (or mirror) and is sampled past v=1 (e.g. `uvtiling > 1`
or UVs outside [0,1]); render `--backend metal` vs `--backend vulkan` → Metal
tiles where Vulkan clamps.

---

## 3. pbrt material mapping: frozen divergences between the two export flavours

**Observed:** 2026-07-25, while extracting the shared resolver (change
`pbrt-material-shared-resolver`). Four were recorded; three are now fixed (this
entry's #1, plus the two in the closing note) and one remains. All **predate**
that change and were preserved deliberately — the change's gate is byte-identical importer output, so fixing any
of them there would have silently changed committed `.usda` fixtures. Each
remaining one is flavour-gated in `resolve_material` with a comment pointing here.

**1. FIXED** (2026-07-29, change `coatedconductor-roughness-spelling`) — the
`coatedconductor` base-metal roughness read a different param per flavour. Both
flavours now read `conductor.roughness` (with `conductor.uroughness` /
`conductor.vroughness`, which **neither** read before), through the one shared
calibration chain.

The entry's premise was wrong and is corrected here for the record: it said the
top-level `roughness` "drives the *coat*". It does not.
`CoatedConductorMaterial::Create` reads `interface.roughness` for the coat,
`conductor.roughness` for the metal, and **no top-level `roughness` at all** —
pbrt does not even ignore one, it **refuses the scene** (`"roughness": unused
parameter`, measured against the pinned pbrt v4). So the UsdPreviewSurface path
was not reading the coat's roughness into the metal; it was reading a parameter
no valid pbrt scene carries. `coateddiffuse` is the asymmetric case where the
top-level spelling IS correct, and it is unchanged.

Gated by the new `tests/assets/suite/mat_coated_metal/` scene: path|wavefront
pbrt-truth relMSE **0.4287 → 0.03797**, and the plain-vs-MaterialX equivalence
pair **0.08661 → 0.009122**.

**2. Two pbrt params are read only on the `-mtlx` path** —
`diffusetransmission` `transmittance` and `interface.eta` on
`coateddiffuse`/`coatedconductor`. UsdPreviewSurface has no input for either, so
the value would be dropped anyway; what is *lost* is the report signal — a
texture-bound or unrecognised one produces a note (and possibly an EXACT→APPROX
escalation) on the `-mtlx` import and nothing at all on the plain one. **Fix:**
decide per param whether the note is worth the divergence, then read
unconditionally and re-baseline the reports.

**Fixed since:** `subsurface` `reflectance` and `radius` used to be on this list. Change
`subsurface-promoting-accessors` removed both: `reflectance` is now resolved once
outside the gate (the usd path already read it for the coefficient chain, so the
gate suppressed its note rather than its read), and the `radius` read is gone —
pbrt's `SubsurfaceMaterial::Create` defines no such param, so reading one was
skinny inventing behaviour. Only the LOBES stay mtlx-only.

---

## 4. A COATED metal renders ~1.6x too bright

**Observed:** 2026-07-29, on the new `tests/assets/suite/mat_coated_metal/` scene
(a gold `coatedconductor`, `--backend metal`, path|wavefront, 128², 256 spp),
while confirming change `coatedconductor-roughness-spelling`.

**Symptom:** the sphere is visibly lighter and less saturated than pbrt's, and its
bright region is broader — pbrt's highlight is more concentrated. Over the sphere
disc: mean luminance **0.2917 vs pbrt 0.1818 (1.60×)**, p99 0.9713 vs 0.8124,
mean RGB (0.350, 0.292, 0.121) vs (0.238, 0.176, 0.078).

**Cause — the COAT, localized by measurement.** Same renderer, same lighting,
sphere-region luminance ratios skinny/pbrt:

| scene | coat | ratio |
|-------|------|-------|
| `mat_conductor`, gold, roughness 0 | none | 0.923 |
| `mat_conductor`, gold, roughness 0.3 | none | 0.978 |
| `mat_coated_metal`, gold, roughness 0.45 | yes | **1.604** |

The BARE conductor matches pbrt (and is slightly *dark*, which is the expected
single-scatter GGX energy loss). Only the coated one blows up, so the metal
response is not the defect. The environment background matches to **0.999**, so
it is not lighting either.

**Probably under-attenuation rather than addition** — both renderers darken a
metal when it is coated, skinny just less. Coated sphere ÷ bare sphere within
each renderer: pbrt **0.298×**, skinny **0.489×**, a ratio of 1.64 that is
suspiciously close to the 1.604 disc ratio.

**That last comparison is UNCONTROLLED and is not load-bearing.** The two scenes
differ in base roughness (0.3 vs 0.45) and geometry, so the nuisance variables
cancel only if both renderers respond to roughness identically — assumed, not
shown. A matching background rules out *miss* rays, not sphere illumination or
area-light visibility, and the floor's 0.868× shows those paths already disagree.
Establishing the mechanism needs an exact counterfactual: one scene, one sphere,
one base roughness, the coat the only difference, rendered in both systems.

What is worth carrying forward: skinny removes energy from the base only through
the `pCoat` selection probability — one Fresnel reflection's worth —
and `flat_material.slang:80,171` shows there is no second channel for any further
loss, because `coatAttenuation = m.coatColor` and the importer sets
`coat_color = [1,1,1]` for every pbrt coated material. pbrt's
`CoatedConductorBxDF` is a stochastic layered BxDF with a medium
(`thickness`/`albedo`) that absorbs at every crossing. Note that interface
transmission and TIR are **redistribution, not net loss**, and that pbrt's
`maxdepth`/`nsamples` truncation is an estimator property, not physics to
reproduce.

**Do not trust the "zero the clearcoat" probe.** Zeroing skinny's coat makes it
*brighter* (0.930×) and still 1.73× pbrt — which looks like an acquittal of the
coat but is not: the probe removes skinny's coat while pbrt's reference keeps its
layered one, so it measures the wrong difference. Two earlier hypotheses died on
the real measurements: the metal's RGB `metallic` response is fine (the named-metal
F0 table reproduces the spectrally-integrated normal-incidence reflectance to
within 0.3% luminance for Au/Ag/Al/Cu), and `--spectral` does not close the gap
(0.04112 vs RGB 0.03792).

**Scope / impact:** both coated types, both backends, independent of execution
mode and of the roughness spelling. `coateddiffuse` is affected by the same
under-attenuation, less visibly, because a Lambert base reflects far less back
into the coat. Every combo still passes the scene's pbrt-truth gate (0.06 relMSE /
0.08 FLIP), so this is a fidelity gap, not a regression.

**Likely fix:** attenuate the base lobe by the coat's transmission, through ONE
shared transfer function that `sample()`, `evaluate()` and
`flatBsdfResponseSpectral` all consume. Note `(1-F)` entering and exiting is only
~0.92 near normal incidence at IOR 1.5, so it cannot account for 1.6× alone.
Tracked as change `coat-layer-energy`.

**Repro:**
```bash
PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
  tests/pbrt/test_suite.py -k coated -m gpu -q -s
```
