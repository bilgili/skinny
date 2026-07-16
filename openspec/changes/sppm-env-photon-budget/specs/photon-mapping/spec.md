# photon-mapping — delta

## ADDED Requirements

### Requirement: Per-pass photon budget is environment-aware

The SPPM per-pass photon count SHALL be
`round(pixels / max(1 − pmfEnv, 1/CAP))` with `CAP = 8`, where `pixels` is
`width·height` and `pmfEnv` is the environment entry of the power-proportional
photon-group pmf, so the expected non-environment photon count stays exactly
`pixels` (uncapped regime) and the environment group's photons ride on top of —
never dilute — the local-light budget. When `pmfEnv` is zero (no live
environment) the count SHALL equal `pixels`, keeping env-free SPPM renders
bit-identical to the flat budget. An explicit photons-per-pass override SHALL
take absolute precedence over the formula. The update-stage flux normalisation
SHALL use the actually-emitted per-pass count so the estimator remains unbiased
for every budget value.

#### Scenario: Env-lit weak-local-light scene loses the env speckle component

The guarantee is a variance reduction of the ENV noise component by ≈ √(budget
multiplier) — not "reaches the pre-env floor" in general: a scene whose pmfEnv
saturates the cap keeps a residual (unbiased) env noise term.

- **WHEN** `assets/glass_caustics_test.usda` (sphere light r = 0.2 + live
  environment, pmfEnv ≈ 0.84 → ×6.25 budget) is rendered by SPPM at 384²/48 spp
  before and after the change
- **THEN** the env noise component
  `sqrt(noise_sigma² − noise_sigma_floor²)` (from `metrics.compute_metrics` vs
  the path anchor; floor = the pmf-forced-sphere-only render, measured 0.0154)
  SHALL shrink by ≈ √6.25 (flat-budget component measured 0.0224; whole-image
  `noise_sigma` 0.0272 → expected ≈ 0.0165 on this scene), **AND** the image
  mean SHALL stay within the recorded unbiasedness tolerance of the path anchor
  (flat-budget measured mean ratio ≈ 1.003)

#### Scenario: Env-free scene is bit-identical

- **WHEN** a scene with no live environment (`pmfEnv = 0`) is rendered by SPPM
- **THEN** the per-pass photon count equals `width·height` and the render is
  bit-identical to the pre-change renderer

#### Scenario: Override wins

- **WHEN** a photons-per-pass override is set
- **THEN** the emitted count equals the override regardless of `pmfEnv`

### Requirement: Photon dispatches never exceed the portable workgroup-count limit

The shared SPPM photon-dispatch loop SHALL bound every single dispatch to at
most `65535 × 64` photons (Vulkan's minimum-guaranteed
`maxComputeWorkGroupCount[0]` times the threadgroup width), tiling larger
budgets into additional 64-aligned sub-dispatches on every backend, so a driver
can never silently clamp `groupCountX` and drop photons while the update stage
divides by the full emitted count (dim bias).

#### Scenario: Oversized budget is tiled, not clamped

- **WHEN** the per-pass photon budget exceeds 4,194,240 on a backend that does
  not request watchdog batching (`photon_batch <= 0`)
- **THEN** the photon stage records multiple sub-dispatches whose 64-aligned
  batch sizes sum to the full budget, bit-identically to a single dispatch
