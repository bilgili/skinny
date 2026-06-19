# photon-mapping Specification

## Purpose
TBD - created by archiving change photon-mapping-sppm. Update Purpose after archive.
## Requirements
### Requirement: SPPM integrator selectable, wavefront-only

The renderer SHALL provide a Stochastic Progressive Photon Mapping (SPPM)
integrator selectable as a third integrator (`INTEGRATOR_SPPM`) alongside the
path tracer and BDPT, wired end to end through the same selection seam: the
shared CLI `--integrator sppm`, the GUI integrator list, and the `integratorType`
GPU dispatch. SPPM SHALL run only under the wavefront execution mode; a
megakernel selection of SPPM SHALL be refused or fall back with a clear message,
the same way the neural directional proposal is wavefront-gated. Switching to or
from SPPM SHALL reset progressive accumulation (the integrator index is part of
the accumulation state hash).

#### Scenario: SPPM is selectable from every front-end

- **WHEN** any front-end is launched with `--integrator sppm --execution-mode wavefront`
- **THEN** the renderer starts on the SPPM integrator, and on the interactive
  front-ends the integrator remains runtime-cycleable among path / bdpt / sppm

#### Scenario: SPPM under the megakernel is refused

- **WHEN** a front-end is launched with `--integrator sppm --execution-mode megakernel`
- **THEN** it exits (or refuses to start SPPM) with a clear message that SPPM
  requires `--execution-mode wavefront`, naming the fix, without initializing a
  broken pipeline

#### Scenario: Switching to SPPM resets accumulation

- **WHEN** the interactive renderer is cycled from path or bdpt to sppm
- **THEN** the progressive accumulation buffer resets to frame 0 and begins a
  fresh SPPM accumulation

### Requirement: Per-pass SPPM pipeline produces a consistent caustic estimate

Each SPPM accumulation pass SHALL execute four stages over the wavefront driver:
(1) an **eye stage** that traces one stochastic camera path per pixel through
specular/perfectly-glossy bounces and stores a single visible point at the first
non-specular surface hit (position, shading normal, path throughput to that
point, material parameters, search radius, photon count, accumulated flux);
(2) a **grid-build stage** that indexes visible points in a uniform spatial hash
by a deterministic counting sort (per-cell count → prefix sum → scatter);
(3) a **photon stage** that emits photons from the scene lights using the
existing power-weighted emissive / light importance distribution, traces them
with Russian roulette, and deposits flux into every visible point found within
its radius via the grid; and (4) an **update stage** that applies the SPPM
radius-reduction and flux-accumulation rules per visible point and resolves each
pixel's radiance estimate. Direct lighting SHALL continue to use the existing
NEE path; SPPM SHALL contribute the indirect / caustic term. The estimator SHALL
be **consistent**: as the per-pixel search radius shrinks across passes, the
SPPM contribution SHALL converge toward the reference solution.

#### Scenario: Radius shrinks monotonically across passes

- **WHEN** SPPM runs for multiple accumulation passes on a fixed scene and camera
- **THEN** each visible point's search radius is non-increasing across passes per
  the SPPM reduction rule, and the accumulated photon count is non-decreasing

#### Scenario: Indirect caustic estimate converges as radius shrinks

- **WHEN** a glass-over-diffuse caustic scene is rendered with increasing pass
  budgets (shrinking radius)
- **THEN** the SPPM caustic estimate's error against the reference trends
  downward (a radius-sweep convergence trend), rather than plateauing at a biased
  value

#### Scenario: Direct lighting is not double-counted

- **WHEN** a scene with both direct illumination and caustics is rendered under SPPM
- **THEN** direct lighting is supplied by NEE and the photon contribution adds the
  indirect/caustic term only, with overall energy matching the reference within
  the parity tolerance (no double-counted direct term)

### Requirement: SPPM scope is surface flat materials; skin and volumetric are deferred

PM-1 SPPM SHALL apply to flat materials only (`UsdPreviewSurface`,
`standard_surface`, `OpenPBR`, and Python flat materials), using the same
material gate as neural guiding and the BDPT flat path. The layered skin /
BSSRDF estimator chain SHALL be left untouched by this change. Photon transport
through participating media / volumes SHALL NOT be attempted in PM-1. These two
extensions — skin/BSSRDF photon deposition and volumetric/media photon transport
— are explicitly deferred to follow-up changes that extend this same capability.

#### Scenario: SPPM on a flat-material scene

- **WHEN** SPPM renders a scene whose surfaces are all flat materials
- **THEN** the integrator runs and produces a caustic estimate

#### Scenario: Skin and volume paths are unchanged

- **WHEN** the SPPM change is applied
- **THEN** the layered skin/BSSRDF estimator chain and the volume-transport path
  produce byte-identical output to before the change for path/bdpt rendering, and
  SPPM does not route photons through them

### Requirement: SPPM runs on both backends, Metal first, at parity

SPPM SHALL be implemented and verified on the native Metal backend first and
SHALL reach parity on the Vulkan backend within this change, so both backends
produce equivalent results within the parity tolerance. The new GPU buffers
(visible points, spatial-hash grid, photon records) SHALL be folded so the Metal
argument-table 31-slot limit is not exceeded even with SPPM active alongside
multiple graph materials. Per-pass queue compaction MAY use the Metal CPU
read-back fallback already used by the wavefront driver; this cost SHALL be
documented in the compatibility matrix.

#### Scenario: SPPM compiles and renders on Metal with multiple graph materials

- **WHEN** SPPM is active on the Metal backend in a scene with two or more
  MaterialX graph materials
- **THEN** the Metal pipeline compiles and dispatches without exceeding the
  argument-table slot limit

#### Scenario: Metal and Vulkan SPPM agree on the caustic scene

- **WHEN** the caustic parity scene is rendered under SPPM on Metal and on Vulkan
  at the same pass budget
- **THEN** the two outputs agree within the parity tolerance

### Requirement: pbrt importer maps the sppm integrator

The pbrt→USD importer SHALL recognize `Integrator "sppm"`: it SHALL translate
the integrator parameters (`numiterations`, `maxdepth`, `photonsperiteration`,
`radius`, `seed`) into USD metadata **and** select the skinny SPPM integrator on
the imported stage. The translation report SHALL classify `sppm` as *mapped*
(for the surface case) rather than *skipped*, and the prior "sppm / photon
integrators out of scope" limitation SHALL be lifted for surfaces. When the pbrt
`radius` parameter is present it SHALL seed the initial SPPM search radius;
otherwise a scene-bounds-derived default SHALL be used.

#### Scenario: Importing an sppm scene selects the SPPM integrator

- **WHEN** a pbrt scene declaring `Integrator "sppm"` is imported
- **THEN** the resulting USD stage carries the sppm parameters in metadata and
  records the skinny SPPM integrator as the selected integrator, and the report
  lists the integrator as mapped

#### Scenario: sppm radius parameter seeds the initial search radius

- **WHEN** the imported sppm integrator specifies a `radius` parameter
- **THEN** the SPPM initial search radius is seeded from that value rather than
  the scene-bounds default

### Requirement: Caustic parity gate against pbrt v4 sppm

The change SHALL include a caustic parity scene (a glass object over a diffuse
plane) with a pbrt v4 `sppm` reference image, compared to skinny's SPPM output
through the existing parity harness using relMSE and FLIP thresholds at a fixed
pass budget, on both Metal and Vulkan. The comparison SHALL also report an
energy ratio to guard against gross bias, and SHALL surface a labelled
side-by-side image (reference · skinny) artifact.

#### Scenario: SPPM matches the pbrt sppm reference within tolerance

- **WHEN** the caustic parity scene is rendered under skinny SPPM and compared to
  the pbrt v4 sppm reference at the gate's pass budget
- **THEN** relMSE and FLIP are within the configured thresholds and the energy
  ratio is within its tolerance band, on both Metal and Vulkan

