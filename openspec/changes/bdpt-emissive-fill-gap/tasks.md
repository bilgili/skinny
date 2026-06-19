## 1. Diagnose (done — runtime instrumentation)

- [x] 1.1 Reproduce the gap headless on `bathroom.usda` (Metal, default mode):
      bdpt/path mean ≈ 0.79, median ≈ 0.005.
- [x] 1.2 Decompose BDPT env contribution (direct z1 / deeper-vertex / escape) vs
      the path tracer's per-bounce env NEE — env total ratio **0.994** ⇒ env is
      not the cause.
- [x] 1.3 Rule out the s=1 light-tracer splat (read `lightSplatBuffer`, reconstruct
      `accum + splat/1024/frames`): splat is 0.2 % of BDPT's total.
- [x] 1.4 Depth-cap sweep (`MAX_BOUNCES`/`BDPT_MAX_DEPTH` = 1) + high-spp run:
      gap is a bias (persists at 512 spp), present from the first bounce.
- [x] 1.5 Identify root cause: `connectT1` selects emissive triangles uniformly
      while `samplePoint` reports the power-weighted `pdfArea` (selection/pdf
      mismatch ⇒ biased under-delivery of the emissive fill).

## 2. Implement

- [x] 2.1 In `bdpt.slang` `connectT1`, replace uniform-by-index emissive selection
      with `sampleEmissiveTriangle(rng.next(), fc.numEmissiveTriangles)`.
- [x] 2.2 Confirm the wavefront BDPT path is fixed too (it imports the same
      `connectT1`).

## 3. Verify

- [x] 3.1 `bathroom.usda` default mode: bdpt/path mean 0.79 → ~0.97; visually
      indistinguishable from the path tracer (labelled before/after grid at shared
      exposure).
- [x] 3.2 Gates green: `tests/pbrt/test_bdpt_energy.py`,
      `tests/pbrt/test_emissive_nee.py`, `tests/pbrt/test_convergence.py`,
      `tests/test_metal_wavefront_bdpt_ab.py` (11 passed, 1 skipped).

## 4. Documentation

- [x] 4.1 Note the power-weighted emissive selection in the BDPT NEE description
      (`docs/Architecture.md` BDPTIntegrator entry) so it does not drift from `nee.slang`.

## 5. Follow-up (not in this change)

- [ ] 5.1 Power-weight + MIS-align the BDPT **light subpath** (`sampleLightOrigin`
      selection + light-origin area pdf, and the `connectGeneric` / `splatMisWeight`
      reverse-pdf terms) so the t≥2 connections deliver their misWeight share and
      the median residual (bdpt/path ≈ 0.66) closes — corpus-risking, separate change.
