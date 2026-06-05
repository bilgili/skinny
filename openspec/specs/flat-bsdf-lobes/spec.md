# flat-bsdf-lobes Specification

## Purpose
TBD - created by archiving change unify-flat-bsdf-on-lobes. Update Purpose after archive.
## Requirements
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

Each lobe SHALL carry a sampler identifier resolved at runtime to a concrete
draw/density strategy, defaulting to the lobe's native strategy (GGX VNDF for
coat and spec, cosine / Lambert for diffuse). The seam SHALL be **populated**:
beyond the native strategies it SHALL register at least one alternate per GGX
lobe (the Heitz-2018 basis-form VNDF, a different warp of the same visible-normal
distribution the native 2023 spherical-cap sampler draws from) and one alternate
for the diffuse lobe (uniform-hemisphere), dispatched by the lobe's sampler
identifier without
modifying the `sample()` or `evaluate()` control flow. Adding a strategy SHALL
require only a new dispatch case plus a host registry entry. With no alternative
sampler selected, behaviour SHALL be identical to a hard-coded native BSDF, and
an unrecognized `(lobe, samplerId)` pair SHALL fall back to the native strategy.

#### Scenario: the native sampler seam is a no-op indirection

- **WHEN** the renderer runs the default selection (no alternative per-lobe
  sampler) on a fixed scene, seed, and frame count
- **THEN** each lobe resolves to its native strategy and the directions drawn by
  `sample()` — and the indirect bounce throughput — are bit-identical to a
  hard-coded native sampler, in both megakernel and wavefront modes

#### Scenario: an alternate strategy is selectable per lobe

- **WHEN** the coat and spec lobes are switched to the Heitz-2018 basis-form
  VNDF, or the diffuse lobe is switched to uniform-hemisphere, via the runtime
  selection
- **THEN** only that lobe's draw/density/weight changes to the selected strategy
  while the other lobes keep their current strategy, and `sample()`/`evaluate()`
  control flow is unchanged

#### Scenario: unknown sampler id degrades to native

- **WHEN** a lobe's sampler identifier does not match any registered strategy for
  that lobe (e.g. a stale persisted selection)
- **THEN** the lobe falls back to its native strategy rather than producing an
  undefined direction or density

### Requirement: Selected per-lobe strategy is shared by sample() and evaluate()

The renderer-selected sampler identifier for each lobe SHALL be read by **both**
`sample()` (the draw) and `evaluate()` (the density/weight), so that for any
registered strategy and any `(wo, wi)` the solid-angle pdf reported by `sample()`
equals the pdf reported by `evaluate()`. `flatBsdfResponse` (= `f·cos`) SHALL
remain sampler-invariant; only the per-lobe draw, density, and bounded weight
SHALL depend on the sampler identifier. Each registered strategy SHALL supply a
bounded per-lobe importance weight (no clamp, no MIS net): the Heitz-2018
basis-form VNDF SHALL share the GGX visible-normal density so its weight remains
`F·G₁`, and uniform-hemisphere SHALL use its own bounded weight (`2·diffHue·cos`).

#### Scenario: basis-form VNDF stays unbiased and matches native convergence

- **WHEN** `three_materials` renders IBL-only at high sample count with the coat
  and spec lobes set to the Heitz-2018 basis-form VNDF
- **THEN** every column converges to the same per-column radiance as the native
  render (brass ≈ 0.219, plus marble and wood), in both megakernel and wavefront
  modes and under both PT and BDPT, because the draw and weight densities are the
  shared GGX VNDF pdf

#### Scenario: uniform diffuse is unbiased with a bounded weight

- **WHEN** a diffuse-dominated surface renders IBL-only with the diffuse lobe set
  to uniform-hemisphere
- **THEN** the converged radiance matches the native (cosine) reference, the
  per-bounce diffuse weight stays bounded (no fireflies), and the low-sample
  variance is higher than the cosine strategy — confirming the seam swapped the
  strategy without introducing bias

