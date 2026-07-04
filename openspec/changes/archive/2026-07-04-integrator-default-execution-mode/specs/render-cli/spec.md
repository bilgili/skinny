## MODIFIED Requirements

### Requirement: Unified render-selection flags across every front-end

The render-selection flags SHALL be exposed identically by every front-end — the
windowed app (`skinny`), the Qt GUI (`skinny-gui`), the web server
(`skinny-web`), and the headless renderer (`skinny-render`) — defined from a
single shared source so they cannot drift. The flags are:

- `--backend {auto,metal,vulkan}` — default `auto`, with a `SKINNY_BACKEND`
  environment fallback; selects the GPU backend, fixed for the session.
- `--integrator {path,bdpt,sppm}` — default `path`. `sppm` selects the
  Stochastic Progressive Photon Mapping integrator, which runs under
  `--execution-mode wavefront`.
- `--execution-mode {auto,megakernel,wavefront}` — default `auto`, with a
  `SKINNY_EXECUTION_MODE` environment fallback. `auto` (the default) SHALL derive
  the execution mode from the startup integrator — `path` → `megakernel`, `bdpt`
  → `megakernel`, `sppm` → `wavefront` — via a single shared resolver used by
  every front-end. An explicit `megakernel` or `wavefront` (flag or environment
  variable) SHALL override the derived default and pin the mode for the session.
  Precedence: explicit mode > integrator-derived default. The resolution SHALL
  happen once at startup, from the integrator active at launch (explicit
  `--integrator`, else the persisted integrator on interactive front-ends, else
  `path`), and the mode SHALL remain fixed for the session.
- `--bdpt-walk {fused,eye,eye_light}` — default `fused`, with a
  `SKINNY_BDPT_WALK` environment fallback; affects only `wavefront` + `bdpt`.

On the interactive front-ends (`skinny`, `skinny-gui`, `skinny-web`)
`--integrator` SHALL set the **initial** integrator and the integrator SHALL
remain runtime-cycleable; the backend, execution mode, and walk SHALL remain
fixed for the session.

#### Scenario: Same flags on every front-end

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-web`, `skinny-render` is run
  with `--help`
- **THEN** all of `--backend`, `--integrator`, `--execution-mode`, `--bdpt-walk`
  are present with identical choices and defaults, `--integrator` lists `path`,
  `bdpt`, and `sppm`, and `--execution-mode` lists `auto`, `megakernel`, and
  `wavefront` with default `auto`

#### Scenario: Interactive front-end launches into the chosen integrator

- **WHEN** an interactive front-end is launched with `--integrator bdpt`
- **THEN** the renderer starts on the `bdpt` integrator, and the integrator can
  still be cycled at runtime

#### Scenario: Execution mode is auto-derived from the integrator

- **WHEN** a front-end is launched with `--integrator sppm` and no explicit
  `--execution-mode` (and no `SKINNY_EXECUTION_MODE`)
- **THEN** the execution mode resolves to `wavefront`, and with `--integrator
  path` or `--integrator bdpt` (and no explicit mode) it resolves to `megakernel`

#### Scenario: Explicit execution mode overrides the derived default

- **WHEN** a front-end is launched with `--integrator path --execution-mode
  wavefront` (or `SKINNY_EXECUTION_MODE=wavefront`)
- **THEN** the execution mode is `wavefront` for the session — the explicit value
  wins over the `path`-derived `megakernel` default

#### Scenario: Headless renders in the chosen execution mode

- **WHEN** `skinny-render` is run with `--execution-mode wavefront` on a Vulkan
  backend
- **THEN** the headless render uses the wavefront backend for the run

#### Scenario: Backend is selected on the command line

- **WHEN** any front-end is launched with `--backend vulkan` (or
  `SKINNY_BACKEND=vulkan`)
- **THEN** the renderer runs on the Vulkan backend for the whole session, and the
  backend is resolved through the single shared resolver used by every front-end

### Requirement: Reject impossible render-flag combinations at startup

The CLI front-ends SHALL validate the parsed render flags at startup and SHALL
exit with a clear error when a combination cannot work, rather than crashing
later or silently doing nothing. Specifically:

- When the integrator is explicitly `bdpt` AND either `--online-training` is set
  or the directional-proposal mixture includes the neural proposal, the
  front-ends SHALL raise a usage error naming the incompatible option and the fix
  (`--integrator path`), because BDPT does not consume the neural directional
  proposal on any backend or execution mode.
- When the integrator is `sppm` AND the **resolved** execution mode is
  `megakernel`, the front-ends SHALL raise a usage error stating that SPPM has no
  megakernel path and naming the fix (drop the explicit `--execution-mode
  megakernel`, or use `--execution-mode wavefront`). Because the execution mode
  defaults to `auto` and `sppm` auto-derives to `wavefront`, this error SHALL
  trip only when the user has **explicitly** forced `--execution-mode megakernel`
  (via flag or `SKINNY_EXECUTION_MODE`).

The validation SHALL run after the execution mode is resolved, SHALL be shared
across all front-ends (`skinny`, `skinny-gui`, `skinny-web`, `skinny-render`),
and SHALL NOT trip when the integrator is the default/persisted path (not
explicitly `bdpt` or `sppm`).

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

#### Scenario: sppm + explicit megakernel exits with an error

- **WHEN** a front-end is launched with `--integrator sppm --execution-mode
  megakernel`
- **THEN** it prints a clear error stating that SPPM has no megakernel path and
  to drop the flag or use `--execution-mode wavefront`, and exits without
  initializing the GPU

#### Scenario: compatible combinations are unaffected

- **WHEN** a front-end is launched with `--integrator path` (with or without a
  neural proposal / `--online-training`), or with `--integrator bdpt` and no
  neural proposal and no `--online-training`, or with `--integrator sppm` (with
  or without an explicit `--execution-mode wavefront`), or with no explicit
  `--integrator`
- **THEN** startup proceeds normally with no error
