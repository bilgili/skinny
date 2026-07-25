# frontend-bringup Delta Specification

## ADDED Requirements

### Requirement: One shared, canonical bring-up sequence for every front-end

The renderer bring-up orchestration SHALL be owned by a single shared bring-up
module — startup-integrator resolution, execution-mode resolution, render-flag
validation, the refusal guards (`sppm`-without-wavefront, MLT envelope,
spectral envelope, MCP support), and GPU backend selection —
and SHALL run in one canonical order for every front-end (`skinny`,
`skinny-gui`, `skinny-render`, `skinny-web`). Front-ends MUST NOT re-implement
or reorder the sequence; they invoke the shared module and keep only
surface-specific wiring (GLFW window, Qt threading, web server/session
lifecycle, MCP flag plumbing). Adding a new refusal guard to the shared
sequence SHALL make it effective on all front-ends without per-front-end edits.

#### Scenario: All four front-ends run the same guard sequence

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-render`, or `skinny-web` is
  launched with a render-flag combination that today is refused at startup
  (for example `--integrator sppm --execution-mode megakernel`, an
  out-of-envelope `--spectral` combination, or an unsupported `--mcp`)
- **THEN** the launch is refused by the shared bring-up sequence with the same
  refusal semantics and byte-identical message text as before this change

Message prefixes are asymmetric and SHALL stay so: the shared refusal guards
print their own fixed `skinny:` prefix on every front-end (the MCP guard prints
none), while a backend-selection failure is prefixed with the invoking
front-end's program name (`skinny:`, `skinny-gui:`, `skinny-render:`,
`skinny-web:`). That is the pre-change behavior on all four; repointing the
guard prefixes at the invoking program would be a user-visible output change
and is out of scope here.

#### Scenario: A guard added to the shared sequence covers every front-end

- **WHEN** a new refusal guard is added to the shared bring-up sequence
- **THEN** the guard is enforced on all four front-ends without any
  front-end-specific bring-up code being modified

### Requirement: Staged bring-up — plan separable from construction

The shared bring-up SHALL be staged in two separable steps: a **plan** step
that performs all resolution and refusal guards and yields a validated plan
(resolved backend, execution mode, startup integrator, and the inputs needed
for construction), and a **create** step that constructs the GPU context and
`Renderer` from that plan. Front-ends that defer context creation (the Qt GUI
constructs on its render thread; the web server constructs per session on a
background thread) SHALL be able to run the plan step at startup and the
create step later, on a different thread, without repeating any guard. The
create step SHALL pass the plan-carried, guard-vetted fields (execution mode,
spectral, bdpt walk, neural build config, backend) to the `Renderer` itself
and SHALL forward front-end-specific constructor inputs (scene path, asset
directories, neural handoff/trainer/precision, …) verbatim as pass-through
keyword arguments; post-construction renderer state (persisted overrides,
integrator/reuse indices, lobe samplers) SHALL remain applied by the
front-ends after `create` returns. The create step SHALL destroy the context
if renderer construction fails.

#### Scenario: Deferred construction reuses the startup plan

- **WHEN** an interactive front-end runs the plan step at startup and later
  invokes the create step on another thread (Qt render thread or a web
  session's background initializer)
- **THEN** the context and renderer are constructed from the already-validated
  plan with no guard re-run, the resolved backend/execution mode match the
  plan, the front-end's own constructor inputs (for example the web session's
  scene path) reach the `Renderer` unmodified via the pass-through keyword
  arguments, and the front-end's post-construction state is applied by the
  front-end afterwards exactly as before this change

#### Scenario: Construction failure tears down the context

- **WHEN** the create step's renderer construction raises after the GPU
  context was created
- **THEN** the context's `destroy()` is invoked before the exception
  propagates

### Requirement: Persisted-settings precedence is preserved per front-end

The plan step SHALL accept an optional persisted-settings input. When it is
provided (the interactive front-ends `skinny` and `skinny-gui`, which persist
settings), the persisted integrator SHALL feed startup-integrator resolution,
the persisted backend SHALL feed backend selection, and the resolution
precedence SHALL remain flag > environment > persisted > auto. When it is
omitted (`skinny-render` and `skinny-web`, which do not persist settings),
resolution SHALL behave exactly as today's non-interactive front-ends: CLI and
environment only, no persisted participation. The shared sequence MUST NOT
introduce persistence to a front-end that does not persist, nor drop it from
one that does.

#### Scenario: Persisted integrator still drives interactive resolution

- **WHEN** `skinny` or `skinny-gui` is launched with no `--integrator` and no
  `SKINNY_EXECUTION_MODE`, and the persisted settings record `sppm` as the
  integrator, under an explicitly forced `--execution-mode megakernel`
- **THEN** the launch is refused (sppm has no megakernel path) exactly as
  before this change, even though the CLI `--integrator` was absent

#### Scenario: Non-persisting front-ends stay persistence-free

- **WHEN** `skinny-render` or `skinny-web` is launched on a machine whose
  `~/.skinny/settings.json` records a persisted backend or integrator
- **THEN** the persisted values do not participate in resolution — the
  resolved backend and execution mode derive from flags, environment, and
  `auto` only, as before this change

### Requirement: Hostless test of the full bring-up sequence

The full bring-up sequence SHALL be testable without a GPU: the create step
SHALL accept an injectable context factory, and a hostless test suite SHALL
exercise the plan and create steps against a stub factory — asserting the
canonical guard order's accept/refuse outcomes across the guard matrix
(integrator × execution mode × spectral × persisted-vs-CLI), the exact refusal
messages including each front-end's program prefix, the persisted-precedence
behavior of both the persisting and non-persisting configurations, and the
destroy-on-failure guarantee.

#### Scenario: Guard matrix verified against a stub context factory

- **WHEN** the hostless bring-up tests run the plan and create steps with a
  stub context factory for each front-end configuration
- **THEN** every guard-matrix combination yields the same accept/refuse
  outcome and refusal message as the pre-change front-ends, and no real GPU
  context is constructed
