# Design — film `maxcomponentvalue` per-sample radiance clamp

## Evidence (why this is the root cause)

Measured on Metal, path/megakernel ≡ wavefront, `assets/bathroom.usda` 256²:

| comparison | relMSE | FLIP | firefly mass |
|---|---|---|---|
| path vs pbrt EXR | 232.8 | 0.387 | top-9 px = 99.7 % |
| bdpt vs path anchor | 6612 | **0.085** | firefly px = 100 % |

skinny max = 7985 (path) / 9132 (bdpt); pbrt ref max = 2120, p99.9 = 50.98 (the
`maxcomponentvalue 50` clip). Offline clamp of skinny@50 collapses path relMSE
232.8 → 10.9 and re-centres `align_exposure` from s=0.25 → 1.07 (the fireflies were
also corrupting the least-squares exposure fit). The clamp is the whole story.

## pbrt semantics being matched

`RGBFilm::AddSample` (pbrt-v4): after a sample's spectral radiance is converted to
sensor RGB (which already includes the `imagingRatio = exposureTime·ISO/100` gain),
if `maxComponentValue` is finite and `m = max(rgb) > maxComponentValue`, the sample
is scaled by `maxComponentValue / m` before splatting. Proportional ⇒ hue-preserving.
It is a per-**sample** clamp, not a per-pixel clamp: a single firefly sample then
contributes only `≤ C/spp` to the pixel mean, which is why a coarse per-pixel
post-clamp leaves residual but the true per-sample clamp does not.

## Domain alignment

skinny already bakes the pbrt film exposure into emitters at import
(`_film_exposure_scale`, `api.py`). So a skinny path sample's radiance is in the
same exposure-scaled domain pbrt clamps in — the threshold `C` transfers directly,
no extra scaling.

## Plumbing (mirrors the existing `iso` and `sppm` tails)

1. Importer: nothing to add — `metadata.scene_metadata` already serialises the whole
   pbrt film into `customLayerData.pbrt.film.params`, so `maxcomponentvalue` is
   present today (verified in `assets/bathroom.usda`).
2. `usd_loader.py`: read `film.params.maxcomponentvalue` into a renderer field
   (default `0.0` = disabled).
3. `renderer.py`: append `filmMaxComponent` (1 × f32) to the `FrameConstants` UBO
   tail in `_pack_uniforms`, add it to `_current_state_hash`, and re-check the
   `_VK_UNIFORM_BUFFER_BYTES` budget (Vulkan UBO is a fixed upload size — a tail
   field past the cap is silently dropped on Vulkan; Metal fc stride absorbs it).
4. `common.slang`: add `float filmMaxComponent;` as the last scalar tail field and
   a `float3 clampSampleRadiance(float3 L, float maxc)` helper
   (`maxc <= 0 ? L : L * min(1, maxc / max(maxc*1e-30, maxComp(L)))`).
5. Apply the helper at each film-accumulation site:
   - megakernel `main_pass.slang`: clamp the integrator's returned radiance before
     it is blended into the accumulation image (covers path + bdpt camera term).
   - wavefront path/bdpt/sppm: clamp at the per-sample accumulation / write-display
     site; clamp the BDPT light splat at its `lightSplatBuffer` add.

## Risk / no-op guarantee

- `filmMaxComponent == 0` ⇒ `clampSampleRadiance` returns `L` unchanged ⇒ every
  existing scene (corpus + skin + interactive) renders byte-identically. The
  recompiled SPIR-V differs only by the added cbuffer field + guarded branch.
- The threshold is per-scene data, so there is no new CLI/GUI surface required;
  the importer is the single source.

## Validation

- Unit: clamp helper math (hue-preserved, no-op below threshold); importer writes
  the field; loader reads it; pack ordering / state-hash reset.
- GPU: the bathroom matrix gate (path/bdpt/sppm × mega/wave + neural + restir).
  Measured post-fix (Metal, 256² path-ref): pbrt-truth ~0.34–0.37 relMSE all
  combos (was 232.8); BDPT-vs-path self-consistency 6612 → 0.36 (MSE 0.017 —
  structurally identical); path mega-vs-wave `0.0000`. The firefly catastrophe is
  gone; the recorded baselines drop ~500×. The scene **stays `known_divergent`**
  for the residual RGB-vs-spectral pbrt-truth (~0.34) and the sppm-vs-path /
  dark-region self-consistency relMSE (real + amplification) — both follow-ups,
  neither loosened here.
- No regression on the other corpus scenes (their films have no
  `maxcomponentvalue` ⇒ the clamp is a no-op ⇒ unchanged).
