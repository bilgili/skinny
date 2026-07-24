# render-cli (delta)

## MODIFIED Requirements

### Requirement: Reject impossible render-flag combinations at startup

The CLI front-ends SHALL validate the parsed render flags at startup and SHALL
exit with a clear error when a combination cannot work, rather than crashing
later or silently doing nothing. Each refusal decision SHALL be derived from
the shared render-envelope predicate (`skinny.render_envelope`): a guard builds
the combination from the parsed args and the resolved execution mode
(tolerating a parsed `Namespace` that omits an axis a front-end suppresses,
via defaults — as today), consults the predicate's verdict — an ordered list
of **all** violated rule codes — and scans it for the codes that guard
**owns**, recorded as code-ownership data in the predicate module, in the
guard's own precedence. No envelope **rule** SHALL be restated as independent
logic inside a guard; a guard owns only its code selection and the code→prose
mapping. A guard's precedence among its owned codes SHALL match today's
behavior (e.g. the MLT guard reports the not-yet-wired refusal before the
megakernel refusal, though the parity matrix's precedence over the same two
codes is the opposite — both derived from the same verdict). Violation codes
owned by no guard SHALL NOT cause a CLI refusal: a neural proposal under the
megakernel with the path integrator is accepted at the CLI (the renderer
strips the neural bit at runtime), as is an environment proposal on
integrators where only the parity matrix skips it — these one-sided
acceptances are recorded in the ownership data. The user-facing refusal text
SHALL remain equivalent to the pre-unification messages (same incompatibility
named, same suggested fix). Specifically:

- When the integrator is explicitly `bdpt` AND either `--online-training` is set
  or the directional-proposal mixture includes the neural proposal, the
  front-ends SHALL raise a usage error naming the incompatible option and the fix
  (`--integrator path`), because BDPT does not consume the neural directional
  proposal on any backend or execution mode.
- When the **effective startup integrator** is `sppm` or `mlt` AND the
  **resolved** execution mode is `megakernel`, the front-ends SHALL raise a
  usage error stating that the integrator has no megakernel path and naming the
  fix (drop the explicit `--execution-mode megakernel`, or use `--execution-mode
  wavefront`). Because the execution mode defaults to `auto` and `sppm`/`mlt`
  auto-derive to `wavefront`, this error SHALL trip only when the user has
  **explicitly** forced `--execution-mode megakernel` (via flag or
  `SKINNY_EXECUTION_MODE`). The **effective startup integrator** is the one
  active at launch — an explicit `--integrator`, else the persisted integrator
  on the interactive front-ends — so a persisted `sppm`/`mlt` under an
  explicitly-forced `megakernel` is refused too.
- When the integrator is explicitly `mlt` AND any of `--online-training`, a
  neural proposal in the mixture, or ReSTIR DI reuse is requested, the
  front-ends SHALL raise a usage error naming the incompatible option — MLT is
  layer-free and wavefront-only. `--spectral --integrator mlt` SHALL NOT be
  refused (change `spectral-mlt`): it starts the spectral MLT wavefront
  session.

The validation SHALL run after the execution mode is resolved and SHALL be shared
across all front-ends (`skinny`, `skinny-gui`, `skinny-web`, `skinny-render`).
The `bdpt` × neural/online-training and `mlt` × axis guards SHALL trip only on
an **explicit** `--integrator bdpt`/`--integrator mlt` (a default/persisted
value SHALL NOT trip them, since the integrator stays runtime-cycleable); the
integrator × `megakernel` guard SHALL consider the effective startup integrator
(explicit or persisted `sppm`/`mlt`) as above.

#### Scenario: bdpt + online-training exits with an error

- **WHEN** `skinny` (or any front-end) is launched with `--integrator bdpt
  --online-training`
- **THEN** it prints a clear error stating the combination is incompatible and
  to use `--integrator path`, and exits without initializing the GPU

#### Scenario: bdpt + neural proposal exits with an error

- **WHEN** a front-end that exposes `--proposals` is launched with `--integrator
  bdpt --proposals bsdf,neural`
- **THEN** it exits with the same clear error

#### Scenario: sppm alone does not error

- **WHEN** a front-end is launched with `--integrator sppm` and no explicit
  `--execution-mode`
- **THEN** the execution mode resolves to `wavefront` and startup proceeds
  normally with no error

#### Scenario: mlt alone does not error

- **WHEN** a front-end is launched with `--integrator mlt` and no explicit
  `--execution-mode`
- **THEN** the execution mode resolves to `wavefront` and startup proceeds
  normally with no error

#### Scenario: sppm + explicit megakernel exits with an error

- **WHEN** a front-end is launched with `--integrator sppm --execution-mode
  megakernel`
- **THEN** it prints a clear error stating that SPPM has no megakernel path and
  to drop the flag or use `--execution-mode wavefront`, and exits without
  initializing the GPU

#### Scenario: mlt + explicit megakernel exits with an error

- **WHEN** a front-end is launched with `--integrator mlt --execution-mode
  megakernel` (or with a persisted `mlt` integrator and an explicitly-forced
  `megakernel` mode)
- **THEN** it prints a clear error stating that MLT has no megakernel path and
  to drop the flag or use `--execution-mode wavefront`, and exits without
  initializing the GPU

#### Scenario: mlt + neural / ReSTIR / online-training exits with an error

- **WHEN** a front-end is launched with `--integrator mlt` plus any of
  `--proposals bsdf,neural`, ReSTIR DI reuse, or `--online-training` (with or
  without `--spectral`)
- **THEN** it prints a clear error naming the incompatible option and exits
  without initializing the GPU

#### Scenario: mlt + spectral does not error

- **WHEN** a front-end is launched with `--integrator mlt --spectral` on a
  flat-material scene
- **THEN** startup proceeds into the spectral MLT wavefront session with no
  unsupported-combination error

#### Scenario: persisted sppm + explicit megakernel exits with an error

- **WHEN** an interactive front-end is launched with no `--integrator`, a
  persisted `sppm` integrator, and an explicit `--execution-mode megakernel`
- **THEN** it prints the same SPPM-has-no-megakernel-path error and exits without
  initializing the GPU, because the effective startup integrator is `sppm`

#### Scenario: compatible combinations are unaffected

- **WHEN** a front-end is launched with `--integrator path` (with or without a
  neural proposal / `--online-training`), or with `--integrator bdpt` and no
  neural proposal and no `--online-training`, or with `--integrator sppm` or
  `--integrator mlt` (with or without an explicit `--execution-mode wavefront`),
  or with no explicit `--integrator`
- **THEN** startup proceeds normally with no error

#### Scenario: refusal decisions come from the shared predicate

- **WHEN** any of the above refusals trips
- **THEN** the decision was produced by scanning the shared render-envelope
  predicate's verdict for the guard's owned codes (not by envelope rules local
  to the guard), and the printed message is equivalent to the pre-unification
  wording for that combination

#### Scenario: unowned violation codes do not refuse at the CLI

- **WHEN** a front-end is launched with `--integrator path --proposals
  bsdf,neural` and an explicit `--execution-mode megakernel`
- **THEN** startup proceeds with no error even though the parity matrix skips
  that combination (the renderer strips the neural bit at runtime) — the
  neural-wavefront-only violation code is owned by no CLI guard, per the
  recorded code-ownership data
