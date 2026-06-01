## MODIFIED Requirements

### Requirement: Execution-mode axis orthogonal to the integrator

The renderer SHALL expose an execution mode — `megakernel` or `wavefront` — as
a selection independent of the integrator (`path` / `bdpt`). The execution mode
SHALL be selected on the **command line** (`--execution-mode`, with a
`SKINNY_EXECUTION_MODE` environment fallback), mirroring `--backend`, and SHALL
be **fixed for the session** — it is a constructor argument of the renderer, not
a runtime-switchable, GUI-surfaced, or persisted parameter. The `megakernel`
mode SHALL remain the default and SHALL preserve current behavior exactly. On
the Metal backend the execution mode SHALL be pinned to `megakernel`
(unchanged), and selecting `wavefront` + `bdpt` SHALL follow the existing
capability gate.

#### Scenario: Execution mode is selected on the command line

- **WHEN** the application is launched with `--execution-mode wavefront` (or
  `SKINNY_EXECUTION_MODE=wavefront`)
- **THEN** the renderer runs in wavefront mode for the whole session, and the
  mode is not offered as a runtime toggle in any front-end

#### Scenario: Default and Metal pin

- **WHEN** no execution mode is specified, or the backend is Metal
- **THEN** the execution mode is `megakernel` and behavior is identical to the
  renderer before the wavefront work

#### Scenario: Megakernel default is unchanged

- **WHEN** the execution mode is `megakernel`
- **THEN** the rendered output and the per-frame dispatch behavior are identical
  to the renderer before this change

## ADDED Requirements

### Requirement: Only the selected backend is compiled

The renderer SHALL compile the GPU pipelines for the selected execution mode
only. In `megakernel` mode it SHALL build the megakernel compute pipeline and
SHALL NOT build the wavefront stage pipelines. In `wavefront` mode it SHALL NOT
compile `main_pass.slang` or the megakernel compute pipeline at all, and SHALL
build only the wavefront stage pipelines. The scene plumbing the wavefront
shares with the megakernel — the set-0 descriptor-set layout, per-frame
descriptor sets, `generated_materials` emission, and the per-graph binding map —
SHALL be built independently of the megakernel pipeline so the wavefront
backend stands alone.

#### Scenario: Wavefront mode does not compile the megakernel

- **WHEN** the renderer is constructed in `wavefront` mode and a scene is loaded
- **THEN** the megakernel `main_pass` pipeline is never compiled (no
  `slangc` invocation for `main_pass`, and the renderer exposes no megakernel
  pipeline), and the scene renders via the wavefront stage pipelines

#### Scenario: Megakernel mode does not build wavefront pipelines

- **WHEN** the renderer is constructed in `megakernel` mode and a scene is
  loaded
- **THEN** no wavefront stage pipeline is built, and the scene renders via the
  megakernel pipeline

#### Scenario: Adding a material in wavefront mode never touches the megakernel

- **WHEN** a model introducing a new material graph is added at runtime in
  `wavefront` mode
- **THEN** only that material's wavefront shade pipeline is compiled, and no
  megakernel pipeline is (re)built
