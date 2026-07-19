# Spectral rendering — MLT delta

## MODIFIED Requirements

### Requirement: Session-fixed spectral mode selection

The renderer SHALL provide an opt-in spectral render mode selected at startup via a
`--spectral` CLI flag on all front-ends (env var `SKINNY_SPECTRAL`), fixed for the session
like the execution mode, and not persisted. The spectral-vs-RGB decision SHALL be made once at
startup — kernels compile as the spectral variant or the RGB variant, never switchable
mid-session. Spectral mode SHALL support the `path`, `bdpt`, `sppm`, and `mlt` integrators
under either the `megakernel` or the `wavefront` execution mode (SPPM and MLT under wavefront
only, as in RGB). Startup SHALL refuse the flag with a clear error when combined with an
unsupported configuration: ReSTIR DI reuse, the neural directional proposal, a non-BSDF
directional proposal, or a scene that requires subsurface/skin or heterogeneous-volume
transport. When the flag is absent, behavior SHALL be identical to today's RGB pipeline.

#### Scenario: spectral path session starts

- **WHEN** the app starts with `--spectral --integrator path` (either backend, either
  execution mode) on a flat-material scene
- **THEN** the session runs with spectrally compiled kernels and reports the mode in startup
  logs

#### Scenario: spectral wavefront session starts

- **WHEN** the app starts with `--spectral --execution-mode wavefront --integrator path`
  (or `bdpt`, or `sppm`, or `mlt`) on a flat-material scene
- **THEN** the session runs the wavefront staged dispatches with spectrally compiled kernels,
  before any refusal

#### Scenario: spectral SPPM starts under wavefront

- **WHEN** the app starts with `--spectral --integrator sppm` (which resolves to the wavefront
  execution mode)
- **THEN** the session runs the spectral SPPM photon + gather passes, not a refusal

#### Scenario: spectral MLT starts under wavefront

- **WHEN** the app starts with `--spectral --integrator mlt` (which resolves to the wavefront
  execution mode) on a flat-material scene
- **THEN** the session runs the spectral MLT bootstrap/mutate/resolve sequence, not a refusal

#### Scenario: unsupported transport refused

- **WHEN** `--spectral` is combined with ReSTIR DI reuse, the neural proposal, a non-BSDF
  directional proposal, or a scene containing skin/subsurface or heterogeneous-volume materials
- **THEN** startup fails with an error naming the unsupported combination

#### Scenario: default sessions unaffected

- **WHEN** the app starts without `--spectral`
- **THEN** the RGB pipeline runs unchanged and no spectral resources are created
