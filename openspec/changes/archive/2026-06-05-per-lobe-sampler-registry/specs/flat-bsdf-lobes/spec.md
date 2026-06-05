## MODIFIED Requirements

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

## ADDED Requirements

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
