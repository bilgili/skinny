# Spectral rendering — environment proposal delta

## MODIFIED Requirements

### Requirement: Session-fixed spectral mode selection

The renderer SHALL provide an opt-in spectral render mode selected at startup
via a `--spectral` CLI flag on all front-ends (env var `SKINNY_SPECTRAL`), fixed
for the session like the execution mode, and not persisted. Spectral mode SHALL
support the `path`, `bdpt`, and `sppm` integrators in their existing execution
envelopes on flat-material scenes. The spectral `path` integrator SHALL admit
the analytic directional-proposal sets `{bsdf}`, `{bsdf,env}`, and `{env}` in
both megakernel and wavefront execution modes. Startup SHALL refuse the flag
with a clear error when combined with ReSTIR DI reuse, the neural directional
proposal, or a scene requiring subsurface/skin or heterogeneous-volume
transport. When the flag is absent, behavior SHALL be identical to the RGB
pipeline.

#### Scenario: spectral analytic environment proposal starts

- **WHEN** the app starts with `--spectral --integrator path --proposals
  bsdf,env` (or `env`) in either execution mode
- **THEN** the session runs the spectral path integrator with the requested
  analytic proposal set and reports that set as resolved, without a spectral
  pin

#### Scenario: unsupported spectral layers remain refused

- **WHEN** `--spectral` is combined with a neural proposal or ReSTIR DI reuse
- **THEN** startup fails before GPU initialization with an error naming the
  unsupported combination

#### Scenario: interactive neural preset resolves safely

- **WHEN** an interactive spectral session selects a preset containing the
  neural proposal
- **THEN** the renderer removes the neural proposal, retains any supported
  analytic proposal in the preset, falls back to BSDF if none remains, and
  reports the pin in resolved configuration

#### Scenario: spectral environment proposal is path-only

- **WHEN** an explicit spectral BDPT/SPPM launch requests an environment
  proposal, or an interactive spectral session switches from path with an
  environment proposal selected to BDPT/SPPM
- **THEN** explicit startup is refused with an error naming `path` as the
  required integrator, while an interactive switch resolves to native BSDF
  sampling and reports the pin

## ADDED Requirements

### Requirement: Spectral path transport supports the environment-importance proposal

The spectral path integrator SHALL sample non-delta bounce directions through
the shared directional-proposal seam. For `{bsdf,env}` and `{env}`, proposal
selection and pdf evaluation SHALL remain scalar and wavelength-independent,
while the chosen direction's BSDF response and environment radiance SHALL be
evaluated at the path's hero wavelengths. Spectral throughput SHALL divide by
the full proposal-mixture solid-angle pdf, and NEE, environment-miss,
emissive-hit, and sphere-hit MIS SHALL use that same generating density. The
environment proposal SHALL reuse the existing environment CDF resources and add
no GPU state.

#### Scenario: spectral environment mixture stays unbiased

- **WHEN** an IBL-lit flat-material scene is rendered with `--spectral
  --integrator path --proposals bsdf,env`
- **THEN** each non-delta bounce divides its per-wavelength response by
  `α_bsdf·p_bsdf + α_env·p_env`, and the converged result matches spectral
  BSDF-only transport within the recorded stochastic tolerance

#### Scenario: both execution modes use the same proposal contract

- **WHEN** the same spectral path scene and analytic proposal set render under
  megakernel and wavefront execution
- **THEN** both execution modes sample through `sampleBounceDirection`, recolor
  the response per hero wavelength, and use the returned mixture pdf for
  downstream MIS

#### Scenario: BSDF-only spectral behavior is preserved

- **WHEN** spectral path tracing runs with the default `{bsdf}` proposal
- **THEN** the shared proposal sampler takes its BSDF-only fast path and the
  spectral response/pdf estimator is unchanged
