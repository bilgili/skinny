## Why

An imported pbrt scene lit by area lights (the `contemporary-bathroom` corpus
scene) rendered ~1.36× dimmer than the pbrt path reference, and the dimness was
hypothesised to be a `blackbody` emitter-normalisation error in the importer.

That hypothesis is **disproven**. The blackbody emitter is exact:

- pbrt's `BlackbodySpectrum` peak-normalisation is cancelled at light
  construction by `scale /= SpectrumToPhotometric(L)` (lights.cpp), so pbrt emits
  **unit-luminance × scale × imagingRatio** — identical to what skinny bakes.
- Direct-view test (camera straight at the emitter, maxdepth 1): skinny luminance
  **11.9995** vs pbrt **11.9999** (ratio **1.0000**) for both `blackbody` and
  `rgb` lights. The baked bathroom emission matches pbrt's `s·imagingRatio` to
  <0.1 % (window 60, bulbs 42000).

The real cause is a **next-event-estimation MIS under-count for emissive
triangles**. `allLightsNEE` weights the emissive-triangle NEE sample with the
power heuristic (`wNEE < 1`), which is only correct when the complementary
BSDF-sampling strategy re-adds the light it hits with weight `wBSDF`. The path
tracer instead **gated that BSDF-hit emission off entirely** for emissive
triangles (it was only added at bounce 0, after a delta lobe, or with no emissive
triangles present — the `spawnedBySpecular` gate). Sphere lights already add their
BSDF-hit term (the `intersectSphereLights` MIS branch); emissive triangles did
not. The result is a dropped `(1 − wNEE)` fraction — a dim bias that **grows with
the light's solid angle**:

- neutral light-size sweep, skinny vs pbrt luminance ratio: tiny light **1.14**,
  large light **1.43** (the size-dependent excess is the missing complement; the
  size-independent ~1.14 residual is the inherent RGB-vs-spectral gap, out of
  scope here).
- bathroom (large window + small bulbs): mean ratio **1.36 → 0.97**, FLIP
  **0.222 → 0.155** after the fix; the exposure-aligned relMSE is unchanged at
  0.342 (it aligns brightness away, so it never measured this bias — it tracks the
  spectral residual).

## What Changes

- Add the **BSDF-sampling MIS complement** for emissive-triangle hits in the path
  tracer. When a non-delta bounce lands on an emissive triangle that NEE also
  sampled, add `throughput · emission · powerHeuristic(bsdfPdf, pdfLightSA)`
  instead of dropping it. The NEE solid-angle pdf at the hit is reconstructed
  **without the emissive-buffer index** — the per-triangle area cancels in the
  power-weighted (`area·lum`) selection, leaving
  `pdfLightSA = Rec709-lum(emission) · d² / (emissiveTotalPower · cosLight)`.
- Add a `FrameConstants.emissiveTotalPower` scalar = `Σ(area · Rec709-lum)` over
  all emissive triangles, computed in `_upload_emissive_triangles`. It **reuses
  the retired `irisZ` slot**, so the UBO byte layout is unchanged.
- Apply identically in the megakernel path (`path.slang`) and the wavefront path
  (`wf_shade_common.slang`) so `megakernel ≡ wavefront` (verified relMSE 0.0000).

Scope: the **Path tracer on the Metal backend**, verified. BDPT, SPPM and the
Vulkan backend keep their existing emission handling and are out of scope here —
they are separate follow-up changes. Because the parity self-consistency anchor is
`(Path, wavefront)` and it now resolves more energy, the BDPT/SPPM-vs-anchor
numbers on the (already `known_divergent`) bathroom shift; their baselines are
re-measured in their own changes.

## Impact

- Affected spec: `emissive-light-sampling` (adds the BSDF-hit MIS-complement
  requirement; the estimator is now MIS-complete, not NEE-only-with-a-gate).
- Affected code: `src/skinny/shaders/integrators/path.slang`,
  `src/skinny/shaders/wavefront/wf_shade_common.slang`,
  `src/skinny/shaders/common.slang` (FrameConstants field),
  `src/skinny/renderer.py` (pack `emissiveTotalPower`).
- Parity corpus: `tests/pbrt/corpus/manifest.json` — lower the `path|megakernel`
  and `path|wavefront` bathroom FLIP baselines 0.222 → 0.155 (relMSE stays 0.342).
- No change to the no-emissive-triangle path (the gate's bounce-0 / delta / no-tri
  branches are byte-unchanged), so non-area-light scenes are unaffected.
