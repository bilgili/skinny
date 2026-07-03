# Constant Sampled Spectra Reduce to Achromatic RGB

## Why

`sampled_spectrum_to_rgb` in `src/skinny/pbrt/spectra.py` projects every pbrt
sampled spectrum through CIE XYZ → linear sRGB. For a **constant** SPD (e.g.
`"spectrum sigma_s" [200 10 900 10]`) the equal-energy whitepoint lands off the
sRGB (D65) white axis, so a value pbrt treats as achromatic comes out tinted:
`[12.0, 9.5, 9.08]` instead of `[10, 10, 10]`. Every constant-spectrum medium
coefficient and reflectance in the corpus (disney-cloud `sigma_s`/`sigma_a` and
ground reflectance, bunny-cloud and clouds `sigma_s`/`sigma_a`) is imported with
a red-shifted tint versus pbrt ground truth, inflating the recorded pbrt-truth
baselines of the volume scenes.

## What Changes

- `sampled_spectrum_to_rgb` detects a constant interpolated SPD and returns the
  achromatic `[v, v, v]` directly, bypassing the XYZ→sRGB projection. Genuinely
  colored spectra keep the current projection path unchanged.
- Baked corpus assets that embedded the tinted values are regenerated
  (`assets/disney_cloud.usda`, `assets/bunny_cloud.usda`; main-checkout viewer
  assets like `assets/clouds.usda` re-imported the same way).
- The recorded pbrt-truth `measured` numbers for `disney_cloud` and
  `bunny_cloud` in `tests/pbrt/corpus/manifest.json` are re-measured. Expected
  direction: improvement — a baseline is never raised to absorb a divergence.
- mega≡wave self-consistency is re-verified on the affected volume scenes.

## Capabilities

### New Capabilities

- `pbrt-spectrum-conversion`: reduction of pbrt spectral parameters (sampled
  spectra, blackbody, named conductor spectra) to linear RGB for skinny's RGB
  renderer, including the achromatic shortcut for constant SPDs.

### Modified Capabilities

<!-- none: pbrt-volume-import's requirement ("sigma_a/sigma_s as RGB via the
existing spectrum path") is unchanged; only that path's output for constant
SPDs changes, which the new capability's spec covers. -->

## Impact

- `src/skinny/pbrt/spectra.py` — the only code change (pure Python, no shader,
  no GPU code).
- All `param_to_rgb` call-sites inherit the fix: media coefficients
  (`media.py`), material reflectance/eta/k (`materials.py`), light `L`/`I`
  (`lights.py`), area lights (`api.py`).
- Rendered output changes for scenes with constant sampled spectra:
  `disney_cloud`, `bunny_cloud` corpus scenes (recorded baselines re-measured)
  and the main-checkout `clouds` viewer asset.
- `tests/pbrt/test_spectra.py` — the weak neutrality assertion
  (`max/min < 1.5`) tightens to exact achromaticity for constant SPDs.
- No descriptor bindings, CLI flags, or public Python API touched.
