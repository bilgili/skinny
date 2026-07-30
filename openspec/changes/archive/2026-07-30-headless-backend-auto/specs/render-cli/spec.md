## ADDED Requirements

### Requirement: Headless Python API resolves the backend through the shared selector

The direct Python entry point SHALL resolve its backend argument through
`skinny.backend_select.select_backend`, with no persisted-settings input. This
covers `HeadlessRenderer` constructed without a `BringupPlan`. The argument SHALL
accept the same three tokens as the `--backend` flag (`auto`, `metal`,
`vulkan`), and its default SHALL be the unset value that defers to the
environment — the same `None` the four front-ends' `--backend` flag defaults to,
not the literal string `auto`, which as an explicit argument outranks
`SKINNY_BACKEND`. Resolution precedence SHALL be the shared chain: the explicit
argument, then the `SKINNY_BACKEND` environment variable, then `auto`. `auto`
SHALL resolve to native Metal on an Apple-Silicon host where a Metal device
constructs, and to Vulkan everywhere else. The module-level headless wrappers
(`render_to_array`, `render_scene`, `render_animation`) SHALL inherit this
default.

#### Scenario: Default headless render uses native Metal on Apple Silicon

- **WHEN** a Python caller constructs `HeadlessRenderer(width, height)` with no
  backend argument on an Apple-Silicon host where a Metal device constructs
- **THEN** the renderer is built on the native Metal context, with no Vulkan
  runtime required on the dynamic-library path

#### Scenario: Default headless render falls back to Vulkan elsewhere

- **WHEN** a Python caller constructs `HeadlessRenderer(width, height)` with no
  backend argument on a host where no Metal device constructs
- **THEN** the renderer is built on the Vulkan context

#### Scenario: An explicit auto token outranks the environment

- **WHEN** `SKINNY_BACKEND=vulkan` is set and a Python caller passes the literal
  `backend="auto"`
- **THEN** the resolution is `auto`'s — Metal where a Metal device constructs —
  because an explicit argument outranks the environment at every token,
  matching `--backend auto` on the command line

#### Scenario: The auto token is accepted

- **WHEN** a Python caller passes `backend="auto"` to `HeadlessRenderer`
- **THEN** the token is resolved to `metal` or `vulkan` before a context is
  constructed, instead of reaching the context factory unresolved and raising
  `unknown backend 'auto'`

#### Scenario: Environment variable is honoured

- **WHEN** `SKINNY_BACKEND=vulkan` is set and a Python caller constructs
  `HeadlessRenderer(width, height)` with no backend argument
- **THEN** the renderer is built on the Vulkan context

#### Scenario: Explicit argument outranks the environment

- **WHEN** `SKINNY_BACKEND=vulkan` is set and a Python caller passes
  `backend="metal"` on a host where a Metal device constructs
- **THEN** the renderer is built on the native Metal context

#### Scenario: Unavailable explicit Metal is refused, not degraded

- **WHEN** a Python caller passes `backend="metal"` on a host where no Metal
  device constructs
- **THEN** the constructor raises a `RuntimeError` naming the missing
  requirement, before any GPU context exists

#### Scenario: The CLI path resolves the backend exactly once

- **WHEN** `skinny-render` runs and hands `HeadlessRenderer` the `BringupPlan`
  from the shared bring-up sequence
- **THEN** the headless constructor uses the plan's already-resolved backend and
  performs no second resolution

#### Scenario: Resolution is verified without a GPU

- **WHEN** the hostless test suite runs
- **THEN** it proves the default token, an explicit token, the environment
  precedence, and the "plan given ⇒ no resolution" case by observing the
  resolution inputs and the resulting plan, with no GPU device constructed
