# render-parity-matrix (delta)

## MODIFIED Requirements

### Requirement: Parity matrix is derived from a validity table
The parity harness SHALL derive the set of rendered combinations from a single
validity table over the axes `integrator ∈ {Path, BDPT, SPPM, MLT}`, `execution_mode ∈
{megakernel, wavefront}`, `proposals ⊇ {neural}`, `reuse ⊇ {ReSTIR DI}`, and
`spectral ∈ {off, on}`. The table SHALL be derived from the shared render-envelope
predicate (`skinny.render_envelope`) — the same predicate the CLI refusal guards and
the renderer scene-level gate consume — and SHALL NOT restate envelope rules as
independent logic; the documented compatibility matrix (CLAUDE.md / README) is
documentation of that predicate, not an independent source the table mirrors. Every
(scene × combo) SHALL be either exercised or skipped with an explicit, machine-readable
reason taken from the predicate's verdict; no valid combo SHALL be silently dropped.
The spectral **envelope** SHALL admit
`path`/`bdpt` under either execution mode, `sppm` under the wavefront mode, and `mlt` under
the wavefront mode — all without proposal or reuse layers, on flat-material scenes without
subsurface/skin or heterogeneous-volume transport. An envelope-eligible spectral combo SHALL
enter the rendered set only once its transport is wired (the live capability gates —
`SPECTRAL_IMPLEMENTED`, and `MLT_IMPLEMENTED` for the `mlt` combo); while unwired it SHALL be
a recorded "not yet wired" skip and SHALL be absent from the rendered set, so the matrix never
renders a spectral combo as an ordinary RGB frame and gates it as if it were spectral.

#### Scenario: SPPM is wavefront-only
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(SPPM, megakernel)` is skipped with reason "SPPM is wavefront-only"
- **AND** `(SPPM, wavefront)` is present in the rendered set

#### Scenario: MLT is wavefront-only and layer-free
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(MLT, megakernel)` is skipped with reason "MLT is wavefront-only"
- **AND** every MLT+neural / MLT+ReSTIR combo is skipped with a recorded reason
- **AND** `(MLT, wavefront)` and `(MLT, wavefront, spectral)` are present in
  the rendered set for flat-material scenes, while skin/subsurface/volume-
  dominated scenes record an out-of-envelope skip

#### Scenario: neural proposal requires wavefront and a flat material scene
- **WHEN** the matrix is enumerated for a subsurface/skin scene (e.g. the SSS dragon)
- **THEN** every combo carrying the neural proposal is skipped with reason
  "neural proposal is flat-material + wavefront only"
- **AND** for a flat-material scene, `(Path, wavefront, +neural)` is present

#### Scenario: BDPT ignores the neural proposal
- **WHEN** the matrix is enumerated
- **THEN** no `BDPT` combo carries the neural proposal (skipped by design)

#### Scenario: spectral envelope admits path/bdpt/sppm/mlt without layers
- **WHEN** the spectral envelope is evaluated for a flat-material scene
- **THEN** spectral+proposal and spectral+reuse combos are rejected with a
  recorded reason, `(SPPM, megakernel, spectral)` and `(MLT, megakernel,
  spectral)` are rejected as wavefront-only, and the envelope-eligible spectral
  combos are `path`/`bdpt` under either execution mode plus `sppm` and `mlt`
  under the wavefront mode

#### Scenario: spectral combos are gated until the transport is wired
- **WHEN** the matrix is enumerated while a spectral transport capability gate is off
- **THEN** the envelope-eligible spectral combos it guards are skipped with a
  "not yet wired" reason and are absent from the rendered set
- **AND** once the gate is on, those combos are present in the rendered set for a
  flat-material scene

#### Scenario: spectral skips volume and skin scenes
- **WHEN** the matrix is enumerated for a scene with heterogeneous media or skin/subsurface
  materials
- **THEN** every spectral combo is skipped with a recorded reason

#### Scenario: matrix validity and CLI refusals cannot diverge
- **WHEN** the validity of a combination is compared between the parity matrix
  and the CLI refusal guards
- **THEN** both derive from the same render-envelope predicate verdict: a combo
  the CLI refuses is never in the matrix's rendered set, and a combo in the
  rendered set is never CLI-refused — while one-sided CLI acceptances
  (violation codes owned by no guard, e.g. the runtime-stripped neural
  proposal under the megakernel) are recorded in the predicate's
  code-ownership data
