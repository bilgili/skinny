# Tasks

## 1. Diagnosis (disprove the blackbody hypothesis)
- [x] 1.1 Read pbrt-v4 `BlackbodySpectrum` / `DiffuseAreaLight::Create` / `PixelSensor`; confirm peak-norm cancels in `scale /= SpectrumToPhotometric` ⇒ pbrt emits unit-luminance × scale × imagingRatio
- [x] 1.2 Numeric: `blackbody_rgb` Y = 1.00000; baked bathroom emission = `s·imagingRatio` (60, 42000) to <0.1 %
- [x] 1.3 Ground-truth direct-view render: skinny emission ≡ pbrt (ratio 1.0000) for `blackbody` and `rgb`
- [x] 1.4 Localise the residual: light-size sweep shows a size-dependent excess (1.43 big vs 1.14 tiny) ⇒ MIS under-count, not emission

## 2. Implementation (Path tracer, Metal)
- [x] 2.1 Add `FrameConstants.emissiveTotalPower` (reuse retired `irisZ` slot) in `common.slang`
- [x] 2.2 Pack `Σ(area·Rec709-lum)` into it from `_upload_emissive_triangles` (and 0 on the degenerate early-out) in `renderer.py`
- [x] 2.3 Megakernel `path.slang`: track `prevBsdfPdf`; replace the emissive-triangle emission gate's drop with the BSDF-hit MIS complement
- [x] 2.4 Wavefront `wf_shade_common.slang`: same complement using the carried `s.bsdfPdf` / `s.rayDir`
- [x] 2.5 Keep `nee.slang` on the power heuristic (the complement lives in the integrator), so the estimator is MIS-complete

## 3. Verification (Metal)
- [x] 3.1 Light-size sweep: big-light ratio 1.43 → 1.15 (matches the tiny-light constant — size bias gone)
- [x] 3.2 Bathroom megakernel path: mean ratio 1.36 → 0.97, FLIP 0.222 → 0.155, ≤1 firefly pixel
- [x] 3.3 `megakernel ≡ wavefront` self-consistency on the bathroom: relMSE 0.0000, mean ratio 1.0000
- [x] 3.4 Confirm Path combos pass the lowered baselines via `pbrt_truth_result` (path|megakernel + path|wavefront PASS=True, relMSE 0.3421/FLIP 0.1547); non-GPU `test_matrix`+`test_metrics` green (23)

## 4. Corpus + docs
- [x] 4.1 Lower `path|megakernel` + `path|wavefront` bathroom FLIP baselines 0.222 → 0.155 in `manifest.json` (relMSE stays 0.342)
- [x] 4.2 Note in the manifest that BDPT/SPPM/neural/restir bathroom baselines re-measure in their own changes (the Path anchor moved)
- [x] 4.3 Update `CHANGELOG.md` and `docs/Megakernel.md` §3.1 + `docs/Wavefront.md`

## 5. Out of scope (follow-up changes)
- [ ] 5.1 BDPT emissive-triangle BSDF-hit MIS
- [ ] 5.2 SPPM emissive-triangle handling under the moved anchor
- [ ] 5.3 Vulkan backend parity re-measure
- [ ] 5.4 The residual ~1.14× RGB-vs-spectral constant (inherent to an RGB renderer)
