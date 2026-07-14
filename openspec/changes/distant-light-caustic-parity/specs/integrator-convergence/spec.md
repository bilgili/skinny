## ADDED Requirements

### Requirement: BDPT walks light subpaths from distant lights

The BDPT light-subpath sampler SHALL emit real light rays from distant
(directional) lights — origin sampled on a disk covering the scene bounds,
direction along the light, with the disk-area factor in the subpath throughput
and the matching area-measure origin pdf — and SHALL launch the light-subpath
random walk from that ray, so distant-light radiance participates in the s ≥ 2
connection strategies and the s = 1 camera splat. Distant-light **specular
caustics** (delta light through a specular chain onto a diffuse receiver), which
unidirectional path tracing cannot sample, SHALL be rendered by BDPT and SHALL
agree with the SPPM photon estimate of the same transport at an equal-time or
explicitly recorded budget using firefly-robust same-budget region statistics.
The extended strategy set SHALL remain a single MIS partition: the eye-side
distant NEE (`connectT1`'s distant branch, today an unconditional full-weight
term) SHALL join the `misWeight` partition against the walked strategies and the
camera splat, and its weight SHALL degrade to exactly full weight when no walked
partner exists, so distant-light direct illumination is neither double-counted
nor dimmed on scenes without specular transport. The first walked vertex's
forward pdf SHALL use the parallel-projection area density of the emission disk
(no distance-squared falloff — the disk placement distance is arbitrary). The
behavior SHALL hold in both the RGB and spectral builds (the spectral build
recoloring the distant emission per-λ via the authored SPD, matching the SPPM
photon emitter), on both backends, and in both execution modes (the megakernel
and wavefront BDPT variants SHALL enable the walk consistently).

#### Scenario: Distant-light glass caustic agrees between BDPT and SPPM

- **WHEN** a scene authoring a DistantLight above a non-dispersive
  delta-transmissive glass object over a diffuse ground is rendered with `bdpt`
  and `sppm` at equal time (or an explicitly recorded high budget), compared via
  firefly-robust same-budget region medians
- **THEN** the caustic region's energy agrees between the two integrators within
  the recorded tolerance, while `path` is recorded as the excluded integrator
  for that component (delta-delta SDS is unsampleable by unidirectional path
  tracing)

#### Scenario: Distant-light direct lighting is not double-counted

- **WHEN** any scene lit by distant lights (with or without specular geometry)
  is rendered with `bdpt` before and after the distant-walk strategy is enabled
- **THEN** directly-lit (non-caustic) regions keep their energy within noise —
  the walked strategies add only transport that eye-side NEE could not sample,
  and full-image energy stays MIS-consistent with the `path` reference wherever
  path can sample the transport

#### Scenario: Spectral BDPT carries the distant walk

- **WHEN** the same distant-caustic scene is rendered with `--spectral bdpt`
- **THEN** the distant-light subpath emission is recolored per hero wavelength
  (authored SPD when present, upsampled RGB otherwise) and the RGB build's
  compiled kernels remain byte-identical under the spectral split guard
