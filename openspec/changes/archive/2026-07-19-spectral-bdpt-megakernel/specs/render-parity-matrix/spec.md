# render-parity-matrix — delta (spectral-bdpt-megakernel)

## MODIFIED Requirements

### Requirement: Parity matrix is derived from a validity table
The parity harness SHALL derive the set of rendered combinations from a single
validity table over the axes `integrator ∈ {Path, BDPT, SPPM}`, `execution_mode ∈
{megakernel, wavefront}`, `proposals ⊇ {neural}`, `reuse ⊇ {ReSTIR DI}`, and
`spectral ∈ {off, on}`. The table SHALL mirror the documented compatibility matrix. Every
(scene × combo) SHALL be either exercised or skipped with an explicit, machine-readable
reason; no valid combo SHALL be silently dropped. The spectral **envelope** SHALL admit
`(Path, megakernel)` and `(BDPT, megakernel)` without proposal or reuse layers, on
flat-material scenes without subsurface/skin or heterogeneous-volume transport; `(SPPM,
spectral)` is a recorded skip naming the spectral wavefront follow-up, and the wavefront
execution mode is a recorded spectral skip until the wavefront follow-up lands. An
envelope-eligible spectral combo SHALL enter the rendered set only once the megakernel
spectral transport is wired (a capability gate); while unwired it SHALL be a recorded "not
yet wired" skip and SHALL be absent from the rendered set, so the matrix never renders a
spectral combo as an ordinary RGB frame and gates it as if it were spectral.

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

#### Scenario: spectral envelope admits Path and BDPT under the megakernel without layers
- **WHEN** the spectral envelope is evaluated for a flat-material scene
- **THEN** `(Path, megakernel, spectral)` and `(BDPT, megakernel, spectral)` are the only
  envelope-eligible spectral combos
- **AND** every `(SPPM, spectral)` / `(wavefront, spectral)` / spectral+proposal /
  spectral+reuse combo is rejected with a recorded reason, the SPPM reason naming the
  spectral wavefront follow-up

#### Scenario: spectral combos are gated until the transport is wired
- **WHEN** the matrix is enumerated while the megakernel spectral transport is not yet wired
  (the capability gate is off)
- **THEN** every envelope-eligible spectral combo is skipped with a "not yet wired" reason
  and is absent from the rendered set
- **AND** once the transport is wired (the capability gate is on), `(Path, megakernel,
  spectral)` and `(BDPT, megakernel, spectral)` are present in the rendered set for a
  flat-material scene

#### Scenario: spectral skips volume and skin scenes
- **WHEN** the matrix is enumerated for a scene with heterogeneous media or skin/subsurface
  materials
- **THEN** every spectral combo is skipped with a recorded reason
