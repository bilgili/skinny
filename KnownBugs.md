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
`pbrt-material-shared-resolver`). All four **predate** that change and were
preserved deliberately — the change's gate is byte-identical importer output, so
fixing any of them there would have silently changed committed `.usda` fixtures.
Each is flavour-gated in `resolve_material` with a comment pointing here.

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

`subsurface` `reflectance` and `radius` used to be on this list. Change
`subsurface-promoting-accessors` removed both: `reflectance` is now resolved once
outside the gate (the usd path already read it for the coefficient chain, so the
gate suppressed its note rather than its read), and the `radius` read is gone —
pbrt's `SubsurfaceMaterial::Create` defines no such param, so reading one was
skinny inventing behaviour. Only the LOBES stay mtlx-only.

---

## 4. An imported metal renders brighter and less saturated than pbrt

**Observed:** 2026-07-29, on the new `tests/assets/suite/mat_coated_metal/` scene
(a gold `coatedconductor`, `--backend metal`, path|wavefront, 128², 256 spp),
while confirming change `coatedconductor-roughness-spelling`.

**Symptom:** the sphere is visibly lighter and less saturated than pbrt's, and its
bright region is broader — pbrt's highlight is more concentrated. Over the sphere
disc: mean luminance **0.2917 vs pbrt 0.1818 (1.60×)**, p99 0.9713 vs 0.8124,
mean RGB (0.350, 0.292, 0.121) vs (0.238, 0.176, 0.078).

**Cause (measured, not assumed):** it is **not** the coat. A probe render with the
clearcoat weight zeroed comes out *brighter*, not darker (coat = 0.930×), and is
still **1.73×** pbrt's sphere luminance with no coat at all. The excess is in the
base metal: skinny imports a pbrt `conductor` as a UsdPreviewSurface `metallic`
surface tinted by an RGB reflectance derived from the named metal, where pbrt
evaluates the conductor Fresnel from spectral `eta`/`k` per wavelength. The
MaterialX sibling, whose `standard_surface` models the layered metal more
directly, sits closer: pbrt-truth relMSE **0.02436 vs 0.03797**.

**Scope / impact:** appearance-level, both backends, independent of execution
mode and of the roughness spelling. Every combo still passes the scene's
pbrt-truth gate (0.06 relMSE / 0.08 FLIP) and mega ≡ wave is exact, so this is a
fidelity gap, not a regression. Same class as the recorded `mat_conductor`
divergence.

**Likely fix:** carry the conductor's spectral `eta`/`k` into the flat BSDF's
Fresnel instead of collapsing them to an RGB reflectance at import — a resolver +
shader change, not an import-spelling one. `--spectral` does not close it today
(spectral relMSE 0.04112 vs RGB 0.03797 on this scene), so the gap is in the
material model rather than the transport.

**Repro:**
```bash
PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
  tests/pbrt/test_suite.py -k coated -m gpu -q -s
```
