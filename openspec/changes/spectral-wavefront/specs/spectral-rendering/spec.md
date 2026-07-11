# Spectral rendering — wavefront delta

## MODIFIED Requirements

### Requirement: Session-fixed spectral mode selection

The renderer SHALL provide an opt-in spectral render mode selected at startup via a
`--spectral` CLI flag on all front-ends (env var `SKINNY_SPECTRAL`), fixed for the session
like the execution mode, and not persisted. The spectral-vs-RGB decision SHALL be made once at
startup — kernels compile as the spectral variant or the RGB variant, never switchable
mid-session. Spectral mode SHALL support the `path`, `bdpt`, and `sppm` integrators under
either the `megakernel` or the `wavefront` execution mode (SPPM under wavefront only, as in
RGB). Startup SHALL refuse the flag with a clear error when combined with an unsupported
configuration: ReSTIR DI reuse, the neural directional proposal, a non-BSDF directional
proposal, or a scene that requires subsurface/skin or heterogeneous-volume transport. When the
flag is absent, behavior SHALL be identical to today's RGB pipeline.

#### Scenario: spectral path session starts

- **WHEN** the app starts with `--spectral --integrator path` (either backend, either
  execution mode) on a flat-material scene
- **THEN** the session runs with spectrally compiled kernels and reports the mode in startup
  logs

#### Scenario: spectral wavefront session starts

- **WHEN** the app starts with `--spectral --execution-mode wavefront --integrator path`
  (or `bdpt`, or `sppm`) on a flat-material scene
- **THEN** the session runs the wavefront staged dispatches with spectrally compiled kernels,
  before any refusal

#### Scenario: spectral SPPM starts under wavefront

- **WHEN** the app starts with `--spectral --integrator sppm` (which resolves to the wavefront
  execution mode)
- **THEN** the session runs the spectral SPPM photon + gather passes, not a refusal

#### Scenario: unsupported transport refused

- **WHEN** `--spectral` is combined with ReSTIR DI reuse, the neural proposal, a non-BSDF
  directional proposal, or a scene containing skin/subsurface or heterogeneous-volume materials
- **THEN** startup fails with an error naming the unsupported combination

#### Scenario: default sessions unaffected

- **WHEN** the app starts without `--spectral`
- **THEN** the RGB pipeline runs unchanged and no spectral resources are created

## ADDED Requirements

### Requirement: Hero-wavelength spectral transport in the wavefront integrators

In spectral mode the wavefront execution mode SHALL transport the same 4 hero wavelengths per
camera path as the megakernel — drawn once at ray generation from the visible-wavelength
importance distribution and carried, together with radiance/throughput as a spectrum, through
the staged compute dispatches (generate → intersect → shade/logic → scatter → connect →
resolve) via the GPU-resident path-state stream. The `path`, `bdpt`, and `sppm` integrators
SHALL each transport spectrally over flat materials, analytic lights, emissive triangles, and
the environment, reusing the shared spectral flat helpers and CIE film resolve. The wavelengths
drawn at path start SHALL be used consistently at every stage and at the film resolve of that
path.

#### Scenario: wavefront path matches the megakernel path spectrally

- **WHEN** the same spectral scene is rendered with `(path, megakernel)` and
  `(path, wavefront)` on the same backend
- **THEN** the accumulated images agree within the recorded self-consistency tolerance

#### Scenario: wavefront BDPT splat resolves per wavelength

- **WHEN** the spectral wavefront BDPT light-tracer splats a camera-subpath connection
- **THEN** the contribution is resolved from its hero wavelengths to linear sRGB before the
  atomic add into the splat buffer, consistent with the megakernel spectral splat

#### Scenario: spectral SPPM carries per-wavelength photon flux

- **WHEN** the spectral SPPM photon pass deposits photons at visible points
- **THEN** photon flux is accumulated per hero wavelength and the eye pass resolves the
  per-wavelength gathered flux to linear sRGB at the visible-point measurement

### Requirement: Wavefront RGB build stays byte-identical under the spectral define

The spectral wavefront kernels SHALL be implemented as `SKINNY_SPECTRAL` compile-time
branches over the shared staged kernels and path-state structs, such that every checked-in
non-spectral wavefront SPIR-V binary is byte-identical to its pre-change form and the widened
path-state record layout appears only when the spectral define is set.

#### Scenario: default wavefront SPIR-V unchanged

- **WHEN** the wavefront kernels are compiled without the spectral define after the change
  lands
- **THEN** each produced non-spectral `.spv` is byte-identical to its checked-in pre-change
  binary (verified by a hostless guard test)
