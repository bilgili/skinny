# restir-di Specification

## Purpose
TBD - created by archiving change restir-di. Update Purpose after archive.
## Requirements
### Requirement: ReSTIR DI reuse mode (wavefront-only)

The renderer SHALL provide a `RESTIR_DI` reuse mode, realized as a `ReusePlugin`
that owns wavefront compute passes plus persistent per-pixel buffers and their
descriptor bindings. It SHALL run on the wavefront execution backend on **both**
Vulkan and Metal; its compute passes and persistent reservoir/G-buffer buffers
SHALL be built from backend-neutral resources so the same plugin runs on either
device. Selecting it on the megakernel backend (on either device) SHALL fall back
to identity reuse (stock NEE), mirroring the wavefront-bdpt capability gate.
Switching the reuse mode SHALL rebuild the wavefront pass set (the pass-structural
contract of the scene-sampling reuse hook).

#### Scenario: ReSTIR builds its passes on wavefront, falls back on megakernel

- **WHEN** ReSTIR DI is selected with the wavefront backend on either Vulkan or
  Metal
- **THEN** the reservoir/G-buffer passes and persistent buffers are built and the
  scene renders via ReSTIR on that device

#### Scenario: Megakernel falls back to identity

- **WHEN** ReSTIR DI is selected with the megakernel execution mode (on either the
  Vulkan or Metal device)
- **THEN** no reservoir passes are built and direct lighting is the stock NEE
  result (identity reuse)

#### Scenario: Metal ReSTIR matches Vulkan ReSTIR

- **WHEN** the same scene is rendered with ReSTIR DI on the Metal wavefront path
  and on the Vulkan wavefront path at an equal sample budget
- **THEN** the converged direct-lighting result agrees within the established
  perceptual tolerance across backends

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

The `Reuse` selector together with the ReSTIR config controls (regime, the
biased-combine toggle, the `M light` / `M bsdf` candidate counts, the spatial
neighbour count and radius, and the `M cap`) SHALL be presented in a dedicated
**ReSTIR** control group, separate from the general Render group, on every
interactive front-end — the windowed app, the Qt GUI, the web panel, and the
debug viewport. The group SHALL be defined once in the shared control tree so the
front-ends stay layout-identical, and SHALL remain present regardless of the
active reuse mode so the user can switch into ReSTIR from it.

#### Scenario: Reprojected mode is reserved but not yet active

- **WHEN** the reprojected temporal regime is selected before its follow-on
  change lands
- **THEN** ReSTIR falls back to a supported temporal regime rather than failing

#### Scenario: Changing ReSTIR config resets accumulation

- **WHEN** the reuse mode or a ReSTIR config value (candidate counts, neighbor
  count/radius, M-cap, biased toggle, regime) changes
- **THEN** progressive accumulation resets so the new configuration converges
  cleanly

#### Scenario: ReSTIR controls live in a dedicated group on every front-end

- **WHEN** any interactive front-end (the windowed app, the Qt GUI, the web
  panel, or the debug viewport) builds its control panel
- **THEN** the `Reuse` selector and the ReSTIR tuning controls appear together
  under a single dedicated **ReSTIR** group, separate from the Render group, and
  that group is present even when the active reuse mode is identity

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

### Requirement: Equations are backed by embedded shader code and a symbol map

`docs/ReSTIR.md` SHALL present, beneath each governing equation, a generated
Slang excerpt of the function that implements it (its signature plus the line(s)
performing the equation's arithmetic, embedded via the
`docs-equation-code-embedding` generator) and a per-equation table mapping each
math symbol to the Slang identifier that carries it. This is in addition to the
existing equation → implementation mapping. The embedded excerpts SHALL be generated from the shipped shaders
(`restir/reservoir.slang`, `restir/light_ris.slang`, `restir/restir_primary.slang`,
`sampling/reuse.slang`), not hand-transcribed.

#### Scenario: Each equation shows the code that computes it

- **WHEN** a reader views an equation in `docs/ReSTIR.md` (e.g. the contribution
  weight `W = Σwᵢ/(M·p̂)` or the GRIS `m_s`)
- **THEN** a generated fenced `slang` excerpt of the implementing function
  (`reservoirFinalize`, `restirSpatial`, …) appears beneath it, tagged with its
  source file

#### Scenario: Math symbols are mapped to Slang identifiers

- **WHEN** a reader needs the variable correspondence for an equation
- **THEN** a per-equation table maps each symbol to its Slang name
  (e.g. `W → r.W`, `p̂(y) → r.pHat`, `M → r.M`, `Σwᵢ → r.wSum`)

#### Scenario: Embedded code cannot silently drift

- **WHEN** a documented ReSTIR shader function is renamed or moved
- **THEN** regenerating the doc (or running the embed check) surfaces the change
  rather than leaving a stale snippet, because the excerpts are generated from
  marked source regions

