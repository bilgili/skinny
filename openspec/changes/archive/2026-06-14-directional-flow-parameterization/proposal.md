## Why

The renderer's neural flow (`neural_flow.slang`) maps the square-space sample to a
direction through a single hard-coded chart — the **cylindrical equal-area** map
`nf_square_to_hemi` / `nf_hemi_to_square` (`φ=2πu, cosθ=v`, `|J|=2π`, the `NF_LOG2PI`
constant). That chart was never a studied choice: it distorts shape badly near the
pole (the top edge `v=1` collapses to the normal) and carries an azimuth seam at the
`u=0≡u=1` wrap that the clamped RQ spline cannot join — both worst exactly where
guiding lobes concentrate.

The `spline_flow` study `directional-flow-param-study` (archived
`spline_flow/openspec/changes/archive/2026-06-11-directional-flow-param-study/`;
full math + numbers in `spline_flow/ParametrizationResults.md`) measured a
**chart × conditioning** efficiency A/B against analytic references for both the BRDF
and path-guiding regimes. Verdict: the **Lambert azimuthal equal-area chart (V1)** is
the one variant that beats the cylindrical baseline `V0×E0` in **both** regimes
(BRDF **1.23×**, path **1.09–1.49×** equal-time efficiency) at **zero Jacobian cost** —
it is equal-area, so `|J|=2π` is unchanged and the entire pdf/MIS path stays
byte-identical. The Lambert chart removes the azimuth seam and puts the pole at the
isotropic disk centre (no edge collapse), making the same lobe a strictly easier
target for the same net size.

The other variants were rejected by the same study: the lobe-rotation chart **V2**
loses BRDF (0.58×) and only ties V1 on path while adding a per-sample in-shader frame
rotation (its lobe-alignment **substitutes** for positional encoding rather than
stacking); the non-equal-area spherical chart **V5** is marginal and would force a
per-sample `log(π² sinθ)` term and a re-exposed pole into both `sampleNeural` and
`pdfNeural`. So the recommended port is **V1 only**.

## What Changes

- Replace the chart pair in `neural_flow.slang`:
  - `nf_square_to_hemi(z)` → the **Shirley concentric square→disk + Lambert azimuthal
    lift** (`cosθ = 1 − r²`, `r = |concentric(z)|`; pole at the disk centre).
  - `nf_hemi_to_square(wi)` → its inverse (`r = √(1−ω_y)`, `φ = atan2(ω_z, ω_x)`,
    inverse-concentric → `(u,v)`).
- **Keep `NF_LOG2PI` and the entire `sampleNeural` / `pdfNeural` plumbing unchanged** —
  V1 is equal-area, `|J|=2π`, so `pdfOmega = exp(log_q_square − NF_LOG2PI)` and the
  `wi.y ≤ 0 → 0` hemisphere guard are byte-identical.
- Clamp the inverse output `(u,v)` off the exact `[0,1]` boundary (`1e-5` margin),
  matching the `_Z_EPS` guard the study added to `spline_flow/train.py`: the RQ spline
  is numerically fragile at a knot edge and the Lambert inverse concentrates points
  there more than the cylindrical inverse did.
- **No retrain required by this change**: the chart is a measure map applied after the
  flow; nets exported for V0 are not weight-compatible with V1 (they were trained
  against the V0 target), so a V1-trained net is uploaded — but the export/upload path
  (`export_weights.py` → `.nrec`) is unchanged. The trained-against-V1 weights come
  from the existing `spline_flow` training pipeline run with `chart="V1"`.

This is a **targeted shader swap** gated behind the completed study; the RQ coupling
core, the MLP, the weight-record format, and the MIS integration are all reused
unchanged.

## Capabilities

### New Capabilities
- `neural-flow-directional-chart`: the renderer's neural flow selects the **Lambert
  azimuthal equal-area** square↔direction chart (seam-free, pole-at-centre) in place of
  the cylindrical baseline, keeping the equal-area constant `|J|=2π` pdf path
  byte-identical, gated on the `spline_flow` directional-flow study's port criteria.

## Impact

- **Code:** `src/skinny/shaders/sampling/neural_flow.slang` — the
  `nf_square_to_hemi` / `nf_hemi_to_square` pair only. `NF_LOG2PI`, `NF_MLP_IN`,
  `sampleNeural`, `pdfNeural` untouched.
- **Correctness:** equal-area ⟹ `q_ω = q_□/2π` unchanged; the study's
  `∫ q_ω dω = 1` proof passes for V1 by construction. A parity harness asserts the new
  chart round-trips (`square→dir→square` identity) and integrates to 1 in-shader.
- **Tests:** a Slang harness under `tests/harnesses/` exercising the new chart pair
  (round-trip + normalization), plus a parity check that V1 `pdfNeural` matches the
  `spline_flow` reference at sampled directions.
- **Performance:** the chart is a handful of transcendentals per sample — negligible
  vs the O(NF_HIDDEN²) GEMM; the efficiency win is the **easier target** (lower
  variance per equal time), already measured in the study.
- **Dependencies:** consumes a V1-trained `.nrec` from `spline_flow` (train with
  `chart="V1"`); no new renderer dependencies. The optional path-only `E3` encoding
  win (in-shader `nf_encode` + `NF_MLP_IN` split) is **out of scope** here — a separate
  change if pursued.
- **Out of scope:** V2 (frame rotation), V5 (non-equal-area `π² sinθ` plumbing), and
  the conditioner positional-encoding port.
