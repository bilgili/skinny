## Why

Spectral path tracing currently pins every directional-proposal selection to
`bsdf`. A request for `--spectral --proposals bsdf,env` therefore reports
`spectral pin; requested bsdf,env`, even though the environment proposal is an
analytic proposal that needs no new spectral resources: its sampled direction
and solid-angle pdf are wavelength-independent, and the existing spectral path
already evaluates the BSDF response and environment radiance per hero
wavelength.

The pin also conflicts with the scene-sampling contract that analytic
`{bsdf,env}` mixtures work in both megakernel and wavefront execution modes.

## What Changes

- Widen the spectral path-tracing envelope from BSDF-only proposals to the
  analytic proposal sets `{bsdf}`, `{bsdf,env}`, and `{env}`.
- Route megakernel `SpectralPathTracer` bounce sampling through the shared
  `sampleBounceDirection` seam and divide the per-wavelength response by the
  returned mixture pdf.
- Preserve the selected analytic proposal set in renderer uniforms and resolved
  configuration reporting instead of pinning it to BSDF.
- Keep the neural directional proposal and ReSTIR reuse outside the spectral
  envelope. Interactive neural selections continue to pin to their analytic
  subset, falling back to BSDF when no analytic proposal remains.
- Leave BDPT and SPPM proposal behavior unchanged; this change applies to the
  path integrator, which owns the directional-proposal seam.

## Capabilities

### Modified Capabilities

- `spectral-rendering`: spectral path transport admits the analytic environment
  proposal and its BSDF/environment mixture in both execution modes.

## Impact

- **Shaders:** `integrators/path_spectral.slang`; comments in the shared
  wavefront spectral shade path.
- **Host:** spectral CLI envelope and renderer proposal resolution/reporting.
- **Tests:** CLI acceptance/refusal, runtime/config resolution, parity-matrix
  coverage, spectral BSDF-vs-environment convergence under both execution
  modes, shader source contracts, and spectral shader compilation.
- **Docs:** spectral scope, directional-proposal compatibility, wavefront
  spectral behavior, and changelog.
- **No new GPU state:** the environment proposal reuses the existing environment
  CDF and pdf bindings; no descriptor, buffer, or pass changes are required.
