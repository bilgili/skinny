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
`pbrt-material-shared-resolver`). Four were recorded; two are now fixed (see the
closing note) and two remain. All **predate** that change and were preserved
deliberately — the change's gate is byte-identical importer output, so fixing any
of them there would have silently changed committed `.usda` fixtures. Each
remaining one is flavour-gated in `resolve_material` with a comment pointing here.

**1. `coatedconductor` base-metal roughness reads a different param per flavour.**
pbrt-v4 spells the metal's roughness `conductor.roughness` and the coat's
`interface.roughness`; `CoatedConductorMaterial::Create` reads **no top-level
`roughness` at all**. The `-mtlx` path reads `conductor.roughness` (correct); the
UsdPreviewSurface path reads top-level `roughness`, so on a `coatedconductor`
that sets both, the two exports render a differently-rough metal — and the USD
path is not reading the *coat's* roughness into the metal, it is reading a
parameter pbrt ignores for this material type.

> An earlier revision of this entry claimed the top-level `roughness` "drives the
> coat". It does not, for `coatedconductor`. The two coated types are asymmetric
> in pbrt, which is what makes this easy to get wrong: `CoatedDiffuseMaterial::
> Create` *does* read top-level `roughness` for its coat, so skinny's
> `coateddiffuse` handling is correct and must not change.

There are in fact **three** defects on this branch, not one: the metal roughness
spelling above, plus `conductor.uroughness`/`conductor.vroughness` dropped
silently on **both** flavours (an anisotropic coated metal imports isotropic with
no note), and the phantom top-level read itself. Nothing gates any of it —
`coatedconductor` appears only in `tests/pbrt/fixtures/all_mtypes.pbrt`, which no
test consumes, and no corpus or confirming-suite scene uses the material.

**Fix:** proposed as change `coatedconductor-roughness-spelling` (both flavours
read `conductor.roughness`, the top-level read is deleted, the anisotropic
spellings flow through the same unreduced `ResolvedRoughness` path). Not yet
implemented.

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
