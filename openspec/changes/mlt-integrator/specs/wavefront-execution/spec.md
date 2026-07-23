# Wavefront Execution — MLT staged sequence

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
→ `wavefront`, `mlt` → `wavefront`; an explicit `megakernel` or `wavefront`
SHALL override that derivation. The `megakernel` mode SHALL remain the derived
default for `path` and `bdpt` and SHALL preserve current behavior exactly. Both
execution modes SHALL run on both backends — the native Metal backend runs the
wavefront mode at parity with Vulkan (`metal-wavefront-parity`), so a
wavefront-only integrator (`sppm`, `mlt`) resolves and runs on Metal. (This
supersedes the pre-`metal-wavefront-parity` clause that pinned Metal to the
megakernel.)

#### Scenario: Execution mode is selected on the command line

- **WHEN** the application is launched with `--execution-mode wavefront` (or
  `SKINNY_EXECUTION_MODE=wavefront`)
- **THEN** the renderer runs in wavefront mode for the whole session, and the
  mode is not offered as a runtime toggle in any front-end

#### Scenario: Default is derived from the integrator

- **WHEN** no explicit execution mode is specified (`auto`)
- **THEN** the execution mode is `megakernel` for `path` and `bdpt` (behavior
  identical to the renderer before this change) and `wavefront` for `sppm` and
  `mlt`

#### Scenario: Megakernel default is unchanged

- **WHEN** the execution mode resolves to `megakernel`
- **THEN** the rendered output and the per-frame dispatch behavior are identical
  to the renderer before this change

## ADDED Requirements

### Requirement: Wavefront drives the MLT chain sequence

The wavefront driver SHALL provide a fourth per-frame staged integrator
sequence for MLT alongside path, BDPT, and SPPM: chain mutation (advance each
chain's primary-sample state), subpath walks and connections through the
existing staged BDPT kernels consuming chain samples, acceptance + splat, and a
b-normalized splat resolve into the accumulation image. Chain and bootstrap
state SHALL live in persistent GPU buffers owned by the driver (allocated
through the existing suballocation path) and survive across accumulation
frames; the sequence SHALL run on both backends through the existing
backend-neutral recorder seam.

#### Scenario: MLT frame executes the staged sequence

- **WHEN** a frame is rendered with `(mlt, wavefront)`
- **THEN** the driver records mutation, walk/connection, acceptance/splat, and
  resolve stages in order, and the accumulation image advances by one
  b-normalized MLT pass

#### Scenario: Chain buffers persist across frames

- **WHEN** two consecutive frames render with no state-hash change
- **THEN** frame two's chains continue from frame one's accepted states (no
  re-bootstrap, no chain reset)
