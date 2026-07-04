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
self-consistency tolerance). `disney_cloud`, `bunny_cloud` (grid) **and** `clouds` (procedural
`MEDIUM_CLOUD`) SHALL be corpus scenes; integrator combos that do not support media (BDPT, SPPM)
SHALL be recorded exclusions in `combo_is_valid`, not silent skips.

#### Scenario: dual gate on the cloud corpus scenes

- **WHEN** the parity matrix runs `disney_cloud`, `bunny_cloud`, and `clouds` with the Path
  integrator
- **THEN** megakernel and wavefront agree within the self-consistency threshold and the pbrt-truth
  metric passes at its recorded baseline

#### Scenario: unsupported combos are recorded

- **WHEN** the matrix enumerates BDPT or SPPM against a volume scene
- **THEN** the combo is excluded by a `combo_is_valid` rule (visible in the coverage meta-test),
  not attempted and not silently dropped

### Requirement: procedural cloud density goes through the same medium seam

The `MEDIUM_CLOUD` medium kind SHALL be implemented exclusively as a new `densityAt(medium, p)`
`case` (pbrt `CloudMedium::Density`: a 5-octave classic-Perlin fBm with a 2-iteration wispiness
domain warp and the altitude falloff), with `mediumMajorant` returning the packed σ_t (the density
clamps to [0,1], so σ_t is the exact global majorant — identical to the grid case). The transport
walk, phase function, NEE, RR, per-channel throughput, boundary handling, and integrator wiring
SHALL be unchanged, and it SHALL require no new GPU binding (the density is analytic). The ported
Perlin `Noise`/`DNoise` (the 256→512 `NoisePerm` table, `Grad`, quintic `NoiseWeight`) SHALL match
pbrt's implementation so the density is pbrt's, not a look-alike.

#### Scenario: ported noise matches the pbrt algorithm

- **WHEN** the Slang-ported `cloudDensity` constants are evaluated at a grid of medium-local points
  and compared to a CPU reference of the identical pbrt algorithm
- **THEN** the two agree to float tolerance at every sampled point

#### Scenario: cloud renders structured, not homogeneous

- **WHEN** `clouds.pbrt` renders inside its interface-bounded sphere under the sky environment
- **THEN** the medium shows fBm cloud structure with the top-lit altitude falloff (denser low,
  wispy top), not a uniform blob, and a zero-**σ** cloud renders as an invisible boundary
  (matches the same scene without the sphere within Monte Carlo noise). Note: pbrt's `density 0`
  is NOT empty — `CloudMedium::Density` keeps the altitude floor term `2·max(0, 0.5−p.y)`
  regardless of `density`, and the port matches pbrt exactly, so the empty-boundary check uses
  σ=0 instead

#### Scenario: other medium kinds unchanged

- **WHEN** the homogeneous and `MEDIUM_NANOVDB` parity scenes render after `MEDIUM_CLOUD` lands
- **THEN** results are unchanged (same seeds → same image) and their gates stay green

### Requirement: the density-field binding is declared on every Vulkan pipeline layout

The `volumeDensity` density-field texture (set 0, binding 26) SHALL be declared
in the Vulkan descriptor-set layout produced by
`ComputePipeline._create_descriptor_set_layout` — the layout shared by the
megakernel driver pipeline and every wavefront stage pipeline (via
`scene_bindings_only`) — as a combined image sampler, matching the
`[[vk::binding(26)]] Sampler3D<float>` declaration in `bindings.slang`. More
generally, every `[[vk::binding(N)]]` declaration active in the Vulkan branch
of `bindings.slang` SHALL have a corresponding entry in that layout: a shader
that references a binding absent from the pipeline layout is undefined
behaviour on Vulkan and a hard `SPIR-V to MSL conversion error: nullptr`
pipeline-build failure on MoltenVK.

#### Scenario: Vulkan pipelines build on MoltenVK

- **WHEN** a `VulkanContext`-backed `Renderer` compiles the megakernel driver
  pipeline (and subsequently the wavefront SPPM/path/BDPT stage pipelines) on
  macOS/MoltenVK
- **THEN** pipeline creation succeeds with no
  `VUID-VkComputePipelineCreateInfo-layout-07988` validation error and no
  MoltenVK SPIR-V→MSL conversion failure, and the SPPM energy-vs-path GPU gate
  (`tests/test_sppm_gpu.py`) runs to a verdict

#### Scenario: shared scene-set bindings are audited hostlessly

- **WHEN** the hostless binding audit compares the Vulkan-branch
  `[[vk::binding(N)]]` declarations of `bindings.slang` against the
  `_create_descriptor_set_layout` binding list
- **THEN** every shader-declared binding is present in the layout (the
  conditional MaterialX graph-param binding 25 counts as declared), so a new
  shared scene binding cannot ship without its Vulkan layout entry

