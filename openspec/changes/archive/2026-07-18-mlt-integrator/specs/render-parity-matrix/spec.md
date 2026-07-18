# Render Parity Matrix — MLT integrator axis value

## MODIFIED Requirements

### Requirement: Parity matrix is derived from a validity table
The parity harness SHALL derive the set of rendered combinations from a single
validity table over the axes `integrator ∈ {Path, BDPT, SPPM, MLT}`, `execution_mode ∈
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

#### Scenario: MLT is wavefront-only, RGB-only, and layer-free
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(MLT, megakernel)` is skipped with reason "MLT is wavefront-only"
- **AND** every `(MLT, spectral)` / MLT+neural / MLT+ReSTIR combo is skipped
  with a recorded reason
- **AND** `(MLT, wavefront)` is present in the rendered set for flat-material
  scenes, while skin/subsurface/volume-dominated scenes record an
  out-of-envelope skip

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
- **THEN** every `(BDPT, spectral)` / `(SPPM, spectral)` / `(MLT, spectral)` /
  `(wavefront, spectral)` / spectral+proposal / spectral+reuse combo is rejected
  with a recorded reason
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

### Requirement: Integrators converge to a common golden image
The harness SHALL designate `(Path, wavefront, no-proposal, no-reuse)` as the
self-consistency anchor per scene and assert that every other valid integrator
combo agrees with the anchor within a per-integrator tolerance (looser for SPPM
caustics and BDPT connection noise than for a pure mode change). MLT SHALL gate
against the same anchor with its own recorded MLT-equivalence tolerance,
measured harness-first: Markov-chain samples are correlated, so at equal spp the
per-pixel noise structure differs from independent sampling even though the
means agree; the tolerance SHALL only ever be tightened, never loosened to hide
a real divergence.

#### Scenario: BDPT matches the Path anchor
- **WHEN** a scene is rendered with BDPT and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the
  integrator-equivalence tolerance

#### Scenario: SPPM matches the Path anchor within its caustic tolerance
- **WHEN** a scene is rendered with `(SPPM, wavefront)` and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the (looser)
  SPPM-equivalence tolerance

#### Scenario: MLT matches the Path anchor within its recorded tolerance
- **WHEN** a scene is rendered with `(MLT, wavefront)` and with the Path anchor
  at the manifest spp
- **THEN** the exposure-aligned relMSE/FLIP between them is within the recorded
  MLT-equivalence tolerance for that scene
