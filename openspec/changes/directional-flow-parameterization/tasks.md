## 1. Lambert chart in the shader

- [ ] 1.1 Add Slang `nf_concentric_square_to_disk(z) -> float2` and its inverse
      `nf_disk_to_square(c) -> float2`, transliterated from `spline_flow/train.py`
      (`_square_to_disk` / `_disk_to_square`); clamp the inverse `(u,v)` to
      `[1e-5, 1-1e-5]`
- [ ] 1.2 Replace `nf_square_to_hemi(z)` with the Lambert forward: concentric disk →
      `cosθ = 1 − r²`, `sinθ = √(1−cos²θ)`, `(cosφ,sinφ) = c/r` (pole at `r→0`)
- [ ] 1.3 Replace `nf_hemi_to_square(wi)` with the Lambert inverse:
      `r = √(1−ω_y)`, `(cosφ,sinφ) = (ω_x,ω_z)/sinθ`, `c = r(cosφ,sinφ)` →
      `nf_disk_to_square`
- [ ] 1.4 Confirm `NF_LOG2PI`, `NF_MLP_IN`, `sampleNeural`, `pdfNeural`, and the
      `wi.y ≤ 0 → 0` guard are byte-unchanged (equal-area ⟹ same constant Jacobian)

## 2. In-shader correctness harness

- [ ] 2.1 Add a Slang harness under `tests/harnesses/` that round-trips the new chart
      (`square→dir→square` and `dir→square→dir` identity within tol) over an interior
      grid
- [ ] 2.2 Add a solid-angle normalization check: integrate `q_ω sinθ` over a dense
      `(θ,φ)` grid and assert `≈ 1` (mirrors `spline_flow/test_charts.py` 3.1)

## 3. Parity with spline_flow

- [ ] 3.1 Train a `spline_flow` net with `chart="V1"` and export via
      `export_weights.py` to a `.nrec`
- [ ] 3.2 Parity check: load the V1 `.nrec`, evaluate `pdfNeural` at sampled directions,
      assert it matches the `spline_flow` V1 reference pdf within tol
- [ ] 3.3 Negative check: a V0-trained record fails the parity gate (guards against
      silently rendering a biased chart-mismatched net)

## 4. Validate + close

- [ ] 4.1 `slangc` compiles `neural_flow.slang` (and `main_pass.slang`) clean
- [ ] 4.2 Render-smoke: a scene with the neural sampler enabled produces finite,
      unbiased frames (no fireflies from chart edge/pole)
- [ ] 4.3 `openspec validate directional-flow-parameterization`
- [ ] 4.4 Note the result back into `spline_flow/ParametrizationResults.md` (port done)
