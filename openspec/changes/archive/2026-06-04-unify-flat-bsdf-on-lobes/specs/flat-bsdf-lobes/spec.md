## ADDED Requirements

### Requirement: Unified lobe BSDF with a single pdf

The flat / `std_surface` material SHALL expose `sample()` and `evaluate()` built
from **one** lobe set (`{coat, spec, diffuse}`) over a **single** parameter
source, such that for any `(wo, wi)` the solid-angle pdf reported by `sample()`
equals the pdf reported by `evaluate()`. The two methods SHALL NOT use different
BSDF models, different parameter structs, or different roughness/opacity handling.

#### Scenario: sample and evaluate pdf agree on a layered material

- **WHEN** a coat+metal material (e.g. brass) is hit and a non-delta direction
  `wi` is produced
- **THEN** `sample().pdf` and `evaluate().pdf` for that `(wo, wi)` are equal to
  floating-point tolerance

### Requirement: Bounded per-lobe weight without clamping

`evaluate().response / evaluate().pdf` SHALL reduce to the lobe's bounded native
importance weight (`F·G₁` for the GGX coat/spec lobes, the diffuse albedo term
for the Lambert lobe). The unified BSDF SHALL stay firefly-free **by
construction** and SHALL NOT rely on a weight clamp, firefly cap, or other
biasing safeguard to bound throughput.

#### Scenario: no spec-lobe fireflies under the proposal mixture

- **WHEN** the `{bsdf, env}` or `{env}` proposal renders a glossy or coated
  surface
- **THEN** per-bounce throughput stays bounded (no firefly spikes) without any
  weight clamp, matching the bounded-weight behaviour of the BSDF-only path

### Requirement: Canonical BSDF for Path Tracer and BDPT in both execution modes

The unified lobe BSDF SHALL be the single BSDF used by **both** the path tracer
and the bidirectional path tracer, in **both** the megakernel and wavefront
execution modes. No integrator SHALL carry a separate or divergent flat-BSDF
evaluation; `evalStdSurfaceBSDF` SHALL NOT appear in the path-traced or BDPT
estimator path (it remains only for the raster preview pass).

#### Scenario: PT and BDPT converge to one per-column value

- **WHEN** `three_materials` renders IBL-only at high sample count under
  PT-BSDF, PT-(BSDF+Env), PT-Env, and BDPT
- **THEN** all four converge to the same per-column radiance (brass ≈ 0.219, plus
  marble and wood), in both megakernel and wavefront modes

### Requirement: Proposal mixture is unbiased on layered materials

The unified model SHALL keep the proposal mixture unbiased on layered / coated
materials: with a non-BSDF proposal active (BSDF+Env or Env), the converged image
SHALL match the BSDF-only and BDPT references with no bias, because the drawing
density and the weighting density come from the same model.

#### Scenario: brass no longer darkens under BSDF+Env

- **WHEN** brass renders IBL-only under the BSDF+Env proposal
- **THEN** its converged radiance matches the BSDF-only / BDPT reference within
  tolerance, and the prior ~3.7% darkening is gone

### Requirement: Per-lobe runtime-pluggable sampler seam with native defaults

Each lobe SHALL carry a sampler identifier resolved at runtime to an `ISampler`,
defaulting to the lobe's native strategy (GGX VNDF for coat and spec, cosine /
Lambert for diffuse). This change SHALL register **only** the native strategies;
the dispatch SHALL be structured so additional strategies can be registered later
without modifying `sample()` or `evaluate()`. With no alternative sampler
selected, behaviour SHALL be identical to a hard-coded native BSDF.

#### Scenario: the native sampler seam is a no-op indirection

- **WHEN** the renderer runs the default selection (no alternative per-lobe
  sampler) on a fixed scene, seed, and frame count
- **THEN** each lobe resolves to its native strategy and the directions drawn by
  `sample()` — and the indirect bounce throughput — are bit-identical to a
  hard-coded native sampler, in both megakernel and wavefront modes (the full
  image still re-baselines, because `evaluate()` / NEE now uses the unified lobe
  BSDF rather than the `std_surface` closure)
