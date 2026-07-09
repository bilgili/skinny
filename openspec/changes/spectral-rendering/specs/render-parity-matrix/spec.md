# render-parity-matrix Specification (delta)

## MODIFIED Requirements

### Requirement: Parity matrix is derived from a validity table
The parity harness SHALL derive the set of rendered combinations from a single
validity table over the axes `integrator ∈ {Path, BDPT, SPPM}`, `execution_mode ∈
{megakernel, wavefront}`, `proposals ⊇ {neural}`, `reuse ⊇ {ReSTIR DI}`, and
`spectral ∈ {off, on}`. The table SHALL mirror the documented compatibility matrix. Every
(scene × combo) SHALL be either exercised or skipped with an explicit, machine-readable
reason; no valid combo SHALL be silently dropped. The spectral **envelope** SHALL admit only
`(Path, megakernel)` without proposal or reuse layers, on flat-material scenes without
subsurface/skin or heterogeneous-volume transport; the wavefront execution mode is a recorded
spectral skip until the wavefront follow-up lands. An envelope-eligible spectral combo SHALL
enter the rendered set only once the megakernel spectral transport is wired (a capability
gate); while unwired it SHALL be a recorded "not yet wired" skip and SHALL be absent from the
rendered set, so the matrix never renders a spectral combo as an ordinary RGB frame and gates
it as if it were spectral.

#### Scenario: SPPM is wavefront-only
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(SPPM, megakernel)` is skipped with reason "SPPM is wavefront-only"
- **AND** `(SPPM, wavefront)` is present in the rendered set

#### Scenario: neural proposal requires wavefront and a flat material scene
- **WHEN** the matrix is enumerated for a subsurface/skin scene (e.g. the SSS dragon)
- **THEN** every combo carrying the neural proposal is skipped with reason
  "neural proposal is flat-material + wavefront only"
- **AND** for a flat-material scene, `(Path, wavefront, +neural)` is present

#### Scenario: BDPT ignores the neural proposal
- **WHEN** the matrix is enumerated
- **THEN** no `BDPT` combo carries the neural proposal (skipped by design)

#### Scenario: spectral envelope is Path-megakernel-only without layers
- **WHEN** the spectral envelope is evaluated for a flat-material scene
- **THEN** every `(BDPT, spectral)` / `(SPPM, spectral)` / `(wavefront, spectral)` /
  spectral+proposal / spectral+reuse combo is rejected with a recorded reason
- **AND** `(Path, megakernel, spectral)` is the only envelope-eligible spectral combo

#### Scenario: spectral combos are gated until the transport is wired
- **WHEN** the matrix is enumerated while the megakernel spectral transport is not yet wired
  (the capability gate is off)
- **THEN** the envelope-eligible `(Path, megakernel, spectral)` combo is skipped with a
  "not yet wired" reason and is absent from the rendered set
- **AND** once the transport is wired (the capability gate is on), `(Path, megakernel,
  spectral)` is present in the rendered set for a flat-material scene

#### Scenario: spectral skips volume and skin scenes
- **WHEN** the matrix is enumerated for a scene with heterogeneous media or skin/subsurface
  materials
- **THEN** every spectral combo is skipped with a recorded reason

## ADDED Requirements

### Requirement: Spectral combos gate against pbrt truth

Spectral combos SHALL be gated by pbrt-truth versus the checked-in (spectrally rendered)
pbrt v4 reference EXRs. On scenes whose pbrt divergence is attributable to RGB reduction, the
spectral combo's recorded pbrt-truth measurement SHALL be at or below the RGB combo's; no
tolerance or baseline SHALL be loosened to admit a spectral combo. The mode-equivalence
(mega≡wave) assertion SHALL be a recorded skip for spectral combos with reason "spectral is
megakernel-only" until the wavefront follow-up; the spectral-versus-RGB anchor delta SHALL be
reported, not asserted tight.

#### Scenario: spectral tightens a spectrum-authored scene

- **WHEN** a corpus scene authored with spectra (blackbody or sampled illuminant) is rendered
  with `(Path, megakernel, spectral)` and `(Path, megakernel, RGB)`
- **THEN** the spectral combo's exposure-aligned relMSE/FLIP against the pbrt reference is at
  or below the RGB combo's recorded measurement

#### Scenario: mode-equivalence is a recorded spectral skip

- **WHEN** the self-consistency gate enumerates a spectral combo
- **THEN** the mega≡wave assertion reports a skip with reason "spectral is megakernel-only"
  instead of failing or passing silently

### Requirement: Coverage meta-test spans the spectral axis

The coverage meta-test SHALL fail when any exposed integrator × spectral combination lacks a
validity entry, and the confirming suite SHALL include at least one spectral-discriminating
scene (dispersive dielectric and/or blackbody-lit) with dispositions for the pbrt-truth gate
in both spectral and RGB modes.

#### Scenario: missing spectral validity entry fails the build

- **WHEN** an integrator exposed by `renderer.integrator_modes` has no validity entry for
  `spectral = on`
- **THEN** the hostless coverage meta-test fails, naming the uncovered combo

#### Scenario: spectral discriminating scene is registered

- **WHEN** the suite coverage meta-test runs
- **THEN** at least one suite scene declares a spectral-discriminating disposition with pbrt
  references gated in both modes
