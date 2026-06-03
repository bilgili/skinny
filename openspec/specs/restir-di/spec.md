# restir-di Specification

## Purpose
TBD - created by archiving change restir-di. Update Purpose after archive.
## Requirements
### Requirement: ReSTIR DI reuse mode (wavefront-only)

The renderer SHALL provide a `RESTIR_DI` reuse mode, realized as a `ReusePlugin`
that owns wavefront compute passes plus persistent per-pixel buffers and their
descriptor bindings. It SHALL run only on the wavefront execution backend;
selecting it on the megakernel or Metal backend SHALL fall back to identity
reuse (stock NEE), mirroring the wavefront-bdpt capability gate. Switching the
reuse mode SHALL rebuild the wavefront pass set (the pass-structural contract of
the scene-sampling reuse hook).

#### Scenario: ReSTIR builds its passes on wavefront, falls back elsewhere

- **WHEN** ReSTIR DI is selected with the wavefront backend
- **THEN** the reservoir/G-buffer passes and buffers are built and the scene
  renders via ReSTIR

#### Scenario: Megakernel falls back to identity

- **WHEN** ReSTIR DI is selected with the megakernel (or Metal) backend
- **THEN** no reservoir passes are built and direct lighting is the stock NEE
  result (identity reuse)

### Requirement: Primary-hit screen-space reservoirs

ReSTIR DI SHALL maintain one reservoir per pixel at the primary visible point,
holding the surviving light sample, its RIS weight `W`, the sample count `M`,
and the cached unshadowed target value `p̂`. The reservoir buffer SHALL be
double-buffered (previous/current) to support temporal reuse, and SHALL be
backed by a per-pixel G-buffer (world position, shading normal, materialId, wo)
used for spatial-neighbor rejection and re-evaluating a neighbor's `p̂` at the
current shading point. Secondary path vertices SHALL continue to use stock NEE.

#### Scenario: Reservoirs and G-buffer are per pixel at the primary hit

- **WHEN** a frame renders with ReSTIR DI
- **THEN** each pixel has a reservoir and G-buffer record at its primary hit, and
  bounces beyond the primary vertex use stock NEE

### Requirement: Unified light-domain RIS with deferred visibility

The initial candidate generation SHALL resample, via weighted reservoir
sampling, over the unified light set — directional, sphere, emissive-triangle,
and environment — mixing both light-sampled and BSDF-sampled candidates. A
reservoir sample SHALL identify the light and the point on it
(`lightType, lightId, point-on-light`). Candidate target evaluation SHALL use
the **unshadowed** target function `p̂ = luminance(f · Le · G)`; a shadow ray
SHALL be cast only for the single surviving sample at resolve time (deferred
visibility).

#### Scenario: Candidates are unshadowed; only the survivor is shadow-tested

- **WHEN** the initial RIS generates candidates for a pixel
- **THEN** each candidate is weighted by its unshadowed target `p̂`, and exactly
  one shadow ray (for the surviving sample) is cast when the pixel is resolved

### Requirement: Unbiased spatiotemporal combination with a biased toggle

By default ReSTIR DI SHALL combine reservoirs across temporal and spatial reuse
using the unbiased estimator: the RIS contribution weight
`W = wSum / (M · p̂(y))`, per-neighbor MIS weights with the domain-aware `1/Z`
normalization (counting only neighbors whose domain could have produced the
sample), and the reconnection Jacobian for the spatial shift, plus a
horizon/visibility domain check. A `biased` toggle SHALL be provided that sums
reservoirs and normalizes by `ΣM` (faster, with discontinuity darkening). The
unbiased mode SHALL converge to the same direct-lighting result as stock NEE.

#### Scenario: Unbiased converges to the stock-NEE reference

- **WHEN** an emissive-lit scene is rendered to high sample count with ReSTIR DI
  unbiased and, separately, with stock NEE
- **THEN** the two converged images agree in integrated radiance within
  tolerance

#### Scenario: Biased toggle trades bias for speed

- **WHEN** the biased toggle is enabled
- **THEN** reservoir combination skips the per-neighbor MIS weights and Jacobian,
  and the result may differ from the reference (bounded darkening at
  discontinuities) without diverging

### Requirement: Canonical integration owns primary direct

When ReSTIR DI is active, the path tracer SHALL treat ReSTIR as the owner of
primary-hit direct lighting: at the primary vertex (`depth == 0`) it SHALL skip
`allLightsNEE` and the BSDF-sampled sphere-light / environment-miss direct term,
taking the primary direct estimate from ReSTIR's resolve pass instead. The
primary bounce SHALL still sample its outgoing direction via the proposal
mixture and spawn the indirect ray; vertices at `depth ≥ 1` SHALL be unchanged
(stock NEE + proposal mixture). Identity reuse SHALL preserve current behavior
exactly.

#### Scenario: Primary direct comes from ReSTIR, indirect is unchanged

- **WHEN** a path is traced with ReSTIR DI active
- **THEN** the primary vertex's direct lighting is ReSTIR's resolved estimate
  (the bounce's own depth-0 light/env direct term is not added), and the indirect
  path beyond the primary vertex behaves as without ReSTIR

### Requirement: Selectable regimes and front-end selection

ReSTIR DI SHALL expose its reuse regime as configuration: spatial reuse on/off
and temporal reuse off/progressive/reprojected. This change SHALL implement the
spatial and progressive-temporal regimes; the reprojected regime SHALL be
reserved in the configuration enum but MAY be unimplemented (falling back) until
a follow-on change adds the motion-vector subsystem. The reuse mode SHALL be
selectable on every front-end (the `reuse_modes` list gains `"ReSTIR DI"`,
surfaced by the data-driven UI) and persisted in settings; changing the reuse
mode or any ReSTIR config value SHALL reset progressive accumulation.

#### Scenario: Reprojected mode is reserved but not yet active

- **WHEN** the reprojected temporal regime is selected before its follow-on
  change lands
- **THEN** ReSTIR falls back to a supported temporal regime rather than failing

#### Scenario: Changing ReSTIR config resets accumulation

- **WHEN** the reuse mode or a ReSTIR config value (candidate counts, neighbor
  count/radius, M-cap, biased toggle, regime) changes
- **THEN** progressive accumulation resets so the new configuration converges
  cleanly

### Requirement: Variance reduction

At equal low sample count, ReSTIR DI (spatial reuse, the default) SHALL produce a
lower-error direct-lighting image than stock NEE on a scene with many emissive
triangles or a large area light.

> **Amended during implementation:** the original requirement also stated that
> progressive-temporal reuse reduces error *further* than spatial-only. On
> skinny's PROGRESSIVE accumulator this does not hold — temporal reuse
> double-counts correlated history (it fights the accumulator's own frame
> averaging; the bias grows with `M_cap` and shows on glossy surfaces). That
> "temporal beats spatial" property belongs to the real-time **reprojected**
> temporal regime, the deferred P3 follow-on (reserved in the selector). Spatial
> reuse is the unbiased, variance-reducing default.

#### Scenario: ReSTIR beats stock NEE at low sample count

- **WHEN** a many-light scene is rendered at a low sample count with ReSTIR DI
  (spatial) and with stock NEE, each compared to a converged reference
- **THEN** the ReSTIR image has lower error
  (`tests/test_restir_variance.py`, `assets/restir_variance_demo.usda`)

