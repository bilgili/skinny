## MODIFIED Requirements

### Requirement: Execution-mode axis orthogonal to the integrator

The renderer SHALL expose an execution mode — `megakernel` or `wavefront` — as
a selection independent of the integrator (`path` / `bdpt`). The execution mode
SHALL be selected on the **command line** (`--execution-mode`, with a
`SKINNY_EXECUTION_MODE` environment fallback), mirroring `--backend`, and SHALL
be **fixed for the session** — it is a constructor argument of the renderer, not
a runtime-switchable, GUI-surfaced, or persisted parameter. When no explicit
mode is given (the default, `auto`), the execution mode SHALL be **derived from
the startup integrator**: `path` → `megakernel`, `bdpt` → `megakernel`, `sppm`
→ `wavefront`; an explicit `megakernel` or `wavefront` SHALL override that
derivation. The `megakernel` mode SHALL remain the derived default for `path`
and `bdpt` and SHALL preserve current behavior exactly. On the Metal backend the
execution mode SHALL be pinned to `megakernel` (unchanged), and selecting
`wavefront` + `bdpt` SHALL follow the existing capability gate.

#### Scenario: Execution mode is selected on the command line

- **WHEN** the application is launched with `--execution-mode wavefront` (or
  `SKINNY_EXECUTION_MODE=wavefront`)
- **THEN** the renderer runs in wavefront mode for the whole session, and the
  mode is not offered as a runtime toggle in any front-end

#### Scenario: Default is derived from the integrator

- **WHEN** no explicit execution mode is specified (`auto`)
- **THEN** the execution mode is `megakernel` for `path` and `bdpt` (behavior
  identical to the renderer before this change) and `wavefront` for `sppm`

#### Scenario: Megakernel default is unchanged

- **WHEN** the execution mode resolves to `megakernel`
- **THEN** the rendered output and the per-frame dispatch behavior are identical
  to the renderer before this change
