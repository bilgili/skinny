# heterogeneous-media Specification

## Purpose
TBD - created by archiving change nanovdb-volume-rendering. Update Purpose after archive.
## Requirements
### Requirement: NanoVDB density grids load as a renderer medium

The renderer SHALL load a NanoVDB (`.nvdb`) float-density grid referenced by a `UsdVol.Volume`
prim (via its `OpenVDBAsset` field `filePath`/`fieldName`) into a GPU-sampled 3D density field,
together with its index→world transform and value range. Parsing SHALL be a pure-Python reader
supporting FloatGrid/FogVolume class grids with NONE and ZIP codecs; unsupported grid classes,
codecs, or file versions SHALL fail with an explicit error naming what was found (never a silent
empty medium).

#### Scenario: density grid decodes to the expected field

- **WHEN** a `.nvdb` FloatGrid with known synthetic contents (authored fixture) is loaded
- **THEN** the decoded dense field matches the authored voxel values, dimensions, and
  index→world transform within float tolerance

#### Scenario: unsupported grid fails loudly

- **WHEN** a `.nvdb` with an unsupported codec or grid class is loaded
- **THEN** loading raises an error identifying the unsupported feature and the file, and the scene
  reports the volume as failed rather than rendering an empty medium

### Requirement: heterogeneous transport goes through the reserved medium seam

The `MEDIUM_NANOVDB` medium kind SHALL be implemented exclusively as new `case` bodies for
`densityAt(medium, p)` (trilinear sample of the density grid in index space) and
`mediumMajorant(medium, segment)` (global majorant σ_t · scale · maxDensity), dispatched on
`MediumParams.kind`. The transport walk, phase function, NEE, RR, per-channel throughput, and
integrator wiring SHALL be unchanged; homogeneous media SHALL render byte-identically to before
the change.

#### Scenario: homogeneous rendering is unchanged

- **WHEN** the existing subsurface/homogeneous parity scenes render after the change
- **THEN** results are identical to the pre-change renderer (same seeds → same image), and the
  subsurface gates stay green

#### Scenario: constant-density grid matches homogeneous

- **WHEN** a `MEDIUM_NANOVDB` grid with constant density 1.0 renders side-by-side with a
  homogeneous medium of the same σ_a/σ_s/g and bounds
- **THEN** the two images agree within Monte Carlo noise (relMSE below the self-consistency
  threshold), demonstrating the grid path is the same walk with a density multiplier

### Requirement: free-standing medium boundaries are index-matched pass-through

A shape bound to an interface (null) material carrying `volume_*` medium overrides SHALL route the
path into the medium walk with an index-matched boundary: no Fresnel reflection, no refraction
bend, no BSDF shading at the boundary — entering and exiting rays pass straight through, and rays
that traverse the bounding shape without a real collision continue as if the shape were absent.

#### Scenario: empty interface shape is invisible

- **WHEN** a bounding sphere with an interface material and zero-density medium is placed in a lit
  scene
- **THEN** the rendered image matches the same scene without the sphere within Monte Carlo noise

#### Scenario: medium scatters by grid density

- **WHEN** the disney-cloud grid renders inside its interface-bounded sphere under
  distant + infinite lighting
- **THEN** cloud detail follows the density field (structured cloud, not a homogeneous blob), and
  regions of zero density contribute no scattering

### Requirement: heterogeneous media render at parity across modes and backends

Heterogeneous-volume scenes SHALL pass the standing dual parity gates for the Path integrator:
megakernel ≡ wavefront self-consistency on Metal, and pbrt-truth against checked-in pbrt v4
references (with any known divergence recorded as a per-combo `baseline`, never a loosened
self-consistency tolerance). `disney_cloud` and `bunny_cloud` SHALL be corpus scenes; integrator
combos that do not support media (BDPT, SPPM) SHALL be recorded exclusions in `combo_is_valid`,
not silent skips.

#### Scenario: dual gate on the cloud corpus scenes

- **WHEN** the parity matrix runs `disney_cloud` and `bunny_cloud` with the Path integrator
- **THEN** megakernel and wavefront agree within the self-consistency threshold and the pbrt-truth
  metric passes at its recorded baseline

#### Scenario: unsupported combos are recorded

- **WHEN** the matrix enumerates BDPT or SPPM against a volume scene
- **THEN** the combo is excluded by a `combo_is_valid` rule (visible in the coverage meta-test),
  not attempted and not silently dropped

