## 1. s=1 splat MIS

- [x] 1.1 Add `splatMisWeight` (misWeight specialised to s=1 — light-side ratios
      only) in `bdpt.slang`.
- [x] 1.2 `splatLightVertex` takes `(lightPath, i)`, computes the camera-reverse
      area pdf (`We·cosCam·cosY/dist²`) + `pdf_ytm2_rev` (y BSDF → lightPath[i-1]
      from toCam), and multiplies the splat by `splatMisWeight`. `splatLightWalk`
      updated; wavefront inherits via the shared helper.
- [x] 1.3 Verified: diffuse corpus display ×1.117→×1.030; bathroom ×1.19→×1.14;
      accum gates unchanged (3/3).

## 2. connectT1 (t=1 NEE) onto misWeight

- [x] 2.1 For the emissive-triangle NEE in `connectT1`: build the sampled light
      point as `lit[0]` (pdfFwd = light area pdf incl. light-selection prob),
      keep the emitter contribution form (`β·f·cosZ·Le·cosY/(d²·pdfArea)`).
- [x] 2.2 Compute endpoint reverse pdfs: `pdf_ytm1_rev` (eye BSDF at z toward the
      light, area-converted to the light point), `pdf_zsm1_rev` (diffuse-emitter
      dir pdf cosLight/π, area-converted to z), `pdf_zsm2_rev` (z BSDF toward
      eye[s-2] viewed from the light dir, area-converted).
- [x] 2.3 Weight with `misWeight(eye, s, {lit0}, 1, pdf_zsm1_rev, pdf_zsm2_rev,
      pdf_ytm1_rev, 0)` instead of `powerHeuristic`. Removed the 2-strategy path
      for emissive triangles.
- [x] 2.4 Sphere/directional NEE left on their current 2-strategy MIS (out of
      scope); only the emissive-triangle branch changed.

## 3. Verification

- [x] 3.1 Diffuse corpus: BDPT **accum** mean/path 1.02 → 0.992 (eye-side now
      tracks path), **display** mean/path 1.117 → 1.017 (residual = BDPT
      legitimately including the s=1 strategy the path tracer lacks).
- [x] 3.2 `bathroom.usda` +4EV display: bdpt/path 1.194 → 1.137 (most of the
      residual is genuine caustic/glossy energy the path tracer is biased dark on,
      not double-count; no bathroom ground-truth to split it exactly).
- [x] 3.3 No regression: `test_bdpt_energy` accum gate 3/3; `test_convergence`
      3/3 + corpus `test_parity` 3/3 (path tracer + exposure-aligned relMSE/FLIP).
- [x] 3.4 New **display** convergence gate
      `test_diffuse_arealight_bdpt_display_tracks_path` (BDPT display vs path,
      tol 0.07) added and green.

## 4. Follow-ups (separate changes)

- [ ] 4.1 Abstract camera importance `We` over `ICamera` (pinhole + thick_lens) so
      the splat is correct for the realistic-lens camera.
- [ ] 4.2 Unify sphere / directional / env NEE onto `misWeight` too.
