# Change: frontend-bringup-builder

## Why

All four front-ends (`skinny` → `app.py`, `skinny-gui` → `ui/qt/app.py`,
`skinny-render` → `headless.py`, `skinny-web` → `web_app.py`; entry points in
`pyproject.toml [project.scripts]`) independently repeat the same bring-up
orchestration: `resolve_execution_mode` → refusal guards
(`reject_sppm_without_wavefront` / `reject_mlt_unsupported` /
`reject_spectral_unsupported` / `reject_mcp_unsupported` /
`validate_render_flags`) → `select_backend` → `make_context` → `Renderer(...)`.

The gating *pieces* are deep and shared (`cli_common.py`, `backend_select.py` —
both hostless-tested), but the *sequence* — which guards run, in what order,
with which persisted inputs — is tribal knowledge copied four times, and it has
already drifted:

- `skinny`/`skinny-gui` call `validate_render_flags` **before**
  `resolve_execution_mode`, then re-check the persisted-integrator cases with
  explicit `reject_*` calls; `skinny-render`/`skinny-web` resolve the mode
  **first** and rely on `validate_render_flags` alone (the
  `resolve_execution_mode` docstring documents the second order — half the
  front-ends contradict it).
- Only the interactive pair calls `reject_mcp_unsupported` and passes
  `persisted=` to `select_backend` / `startup_integrator_name`; the
  non-interactive pair silently depends on the CLI-keyed guard subset being
  equivalent when there is no persistence.
- Each front-end hand-rolls the same `try: select_backend … except RuntimeError:
  SystemExit(f"{prog}: {exc}")` wrapper with its own prefix.

The interactive order is also **out of spec today**: the `render-cli` spec
(`openspec/specs/render-cli/spec.md`, "Reject impossible render-flag
combinations at startup") already requires "The validation SHALL run after the
execution mode is resolved" — `skinny`/`skinny-gui` validate before resolving
and only stay refusal-equivalent through the guards' `"auto" != "megakernel"`
string comparison. This change restores conformance without touching the
requirement.

Nothing enforces that a new guard (the next `reject_*`) lands in all four
places, and there is no hostless test of the full sequence. This change is
squarely in the spirit of the standing rule that behavior changes must apply
across **all** front-ends consistently: one bring-up module makes that
structural instead of manual.

## What Changes

- Add one bring-up module (`src/skinny/bringup.py`) owning the canonical
  args → validated `(backend, execution_mode, startup integrator, …)` sequence,
  staged in two steps so front-ends that defer context creation (Qt render
  thread, per-session web) can plan first and construct later:
  1. **plan** — resolve + all refusal guards, one canonical order, persisted
     settings as an optional input, `prog`-prefixed refusal messages;
  2. **create** — `make_context` + `Renderer(...)` from the plan, with
     destroy-on-failure, injectable context factory for hostless tests.
- Migrate the four front-ends onto it, one at a time, refusal-parity checked.
  Front-ends keep only surface-specific wiring (GLFW window, Qt threading /
  `render_session.py`, web server + session lifecycle, MCP flags).
- Add a hostless test of the full gating sequence against a stub context
  factory (guard order, persisted precedence, refusal messages per front-end).
- Behavior-preserving: same refusals, same messages, same startup behavior on
  all four; a future fifth front-end costs one plan + one create call.

## Capabilities

### New Capabilities

- `frontend-bringup` — the shared, staged bring-up sequence: canonical guard
  order, persisted-precedence preservation (flag > env > persisted > auto, only
  where a front-end persists), per-front-end knobs for genuine deltas only, and
  hostless testability via an injectable context factory.

### Modified Capabilities

- None. `render-cli`'s requirements (shared flags, refusal semantics, shared
  resolvers, validation-after-resolution) are unchanged — this change relocates
  *where* the already-specified sequence runs, it does not change what is
  required of it. It in fact restores the interactive front-ends to the
  existing "validation SHALL run after the execution mode is resolved"
  requirement they currently violate.

## Impact

- **New code:** `src/skinny/bringup.py`; hostless tests
  (`tests/test_bringup.py`).
- **Modified code:** `src/skinny/app.py`, `src/skinny/ui/qt/app.py`,
  `src/skinny/ui/qt/viewport.py` (`_build_renderer` — the actual Qt
  context/renderer construction site, moved onto `plan.create`),
  `src/skinny/headless.py`, `src/skinny/web_app.py` — bring-up regions replaced
  by builder calls; `cli_common.py` / `backend_select.py` untouched (the builder
  composes them, it does not absorb them).
- **Not touched:** `render_session.py` (`RenderCommandQueue` /
  `QtRendererConfig`) — Qt-only, sits *above* the builder; other front-ends are
  not forced through it. Renderer internals, shaders, settings-file format:
  unchanged.
- **Risk:** low — behavior-preserving refactor; per-front-end migration is
  independently revertible and gated on refusal-message parity.
