# Render parity matrix — spectral wavefront delta

## MODIFIED Requirements

### Requirement: Spectral combos gate against pbrt truth

Spectral combos SHALL be gated by pbrt-truth versus the checked-in (spectrally rendered)
pbrt v4 reference EXRs. On scenes whose pbrt divergence is attributable to RGB reduction, the
spectral combo's recorded pbrt-truth measurement SHALL be at or below the RGB combo's; no
tolerance or baseline SHALL be loosened to admit a spectral combo. With spectral transport
available in both execution modes, the mode-equivalence (mega≡wave) assertion SHALL apply to
spectral `path`/`bdpt` combos exactly as it does to their RGB counterparts — wavefront spectral
anchoring to the **megakernel spectral path** image within the recorded self-consistency
tolerance — and SHALL no longer be a blanket "spectral is megakernel-only" skip. The one
retained skip SHALL be spectral `bdpt` on an out-of-gamut dispersion (light-tracer splat)
scene, whose per-splat gamut clamp is nonlinear and differs by splat granularity between the
fused and staged pipelines; that skip SHALL record its reason. The spectral-versus-RGB anchor
delta SHALL still be reported, not asserted tight.

#### Scenario: spectral tightens a spectrum-authored scene

- **WHEN** a corpus scene authored with spectra (blackbody or sampled illuminant) is rendered
  with `(Path, megakernel, spectral)` and `(Path, megakernel, RGB)`
- **THEN** the spectral combo's exposure-aligned relMSE/FLIP against the pbrt reference is at
  or below the RGB combo's recorded measurement

#### Scenario: spectral mode-equivalence is asserted

- **WHEN** the self-consistency gate enumerates a spectral `path` or `bdpt` combo in the
  wavefront execution mode on an in-gamut (non-dispersion-splat) scene
- **THEN** its accumulated image is asserted equivalent to the megakernel spectral path image
  within the recorded tolerance, not skipped

#### Scenario: spectral BDPT dispersion splat stays a recorded skip

- **WHEN** the self-consistency gate enumerates spectral `bdpt` on an out-of-gamut dispersion
  (light-tracer splat) scene
- **THEN** the mega≡wave assertion records a skip naming the per-splat clamp granularity, rather
  than failing or asserting tight

## ADDED Requirements

### Requirement: Spectral wavefront combos are valid rendered combos

The validity table SHALL admit `(path, wavefront, spectral)`, `(bdpt, wavefront, spectral)`,
and `(sppm, wavefront, spectral)` into the rendered set (flat materials, BSDF proposal, no
reuse). Spectral SPPM SHALL be gated for self-consistency against the **spectral** path anchor
(the megakernel spectral path image), NOT the RGB golden — on a spectrum-authored scene
spectral and RGB differ by construction, so an RGB anchor would be invalid; if
spectral-path-anchoring is infeasible, spectral SPPM SHALL instead be gated by pbrt-truth only,
with no RGB-golden self-consistency claim. The self-consistency gate SHALL select the anchor
image per the spectral axis, so a spectral combo is never compared against an RGB anchor. The
remaining spectral rejections — the neural directional proposal, ReSTIR reuse, and
skin/subsurface/heterogeneous-volume scenes — SHALL stay recorded exclusions.

#### Scenario: spectral wavefront combos render and gate

- **WHEN** the parity matrix enumerates the spectral axis under the wavefront execution mode
- **THEN** the `path`, `bdpt`, and `sppm` integrators are admitted as rendered combos and gated
  against pbrt truth and the appropriate self-consistency anchor

#### Scenario: spectral SPPM validity entry exists

- **WHEN** the coverage meta-test enumerates `sppm` on the spectral axis
- **THEN** a validity entry exists (rendered under wavefront), so the build does not fail for a
  missing spectral SPPM combo
