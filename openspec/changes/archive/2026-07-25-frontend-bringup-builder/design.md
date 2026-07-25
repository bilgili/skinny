# Design: frontend-bringup-builder

## Context

Four front-ends run the same bring-up orchestration with hand-copied
sequences. Mapped from the code (worktree `skinny-arch-proposals`):

**`skinny` (`app.py` main, ~495–650):**
parse (`add_render_flags(parser)` — full flag set) →
`validate_render_flags(args)` → `ensure_dirs()` + `load_settings()` →
`startup_integrator_name(args.integrator, persisted integrator_index)` →
`resolve_execution_mode` → `reject_sppm_without_wavefront` →
`reject_mlt_unsupported` → `reject_spectral_unsupported` →
`reject_mcp_unsupported` →
`select_backend(args.backend, persisted=saved["backend"])` wrapped in
`try/except RuntimeError → SystemExit("skinny: …")` → GLFW init + window →
`make_context(backend, window, w, h)` → `Renderer(...)` → per-key persisted
overrides (encoding, neural_handoff, neural_trainer, train_precision,
sppm_glossy_roughness, online_training — each guarded by
`"--flag" not in sys.argv and not env`).

**`skinny-gui` (`ui/qt/app.py` main, ~660–760):**
parse (`add_render_flags(parser, proposals=False)`) → `validate_render_flags`
→ `load_settings()` → same persisted `startup_integrator_name` → same
resolve + four `reject_*` calls, explicitly ordered **before**
`select_backend` (comment: guard before the GPU probe) →
`select_backend(..., persisted=...)` with `"skinny-gui: …"` prefix →
persisted-precedence for encoding + sppm_glossy_roughness only →
`QApplication` + `MainWindow(...)`. Context + `Renderer` construction is
**deferred** into the Qt render thread — the actual construction site is
`ui/qt/viewport.py` `_build_renderer` (~70–104), fed a `QtRendererConfig`
via `render_session.py`.

**`skinny-render` (`headless.py` main, ~380–410):**
parse (`add_render_flags(p, resolution=False, mcp=False)` — owns its own
`--width/--height`) → `resolve_execution_mode(ns.execution_mode,
ns.integrator or "path")` **first** → `validate_render_flags(ns)` →
`reject_spectral_unsupported` only → `select_backend(ns.backend)` (no
persisted) with `"skinny-render: …"` prefix → `HeadlessRenderer(...)`, which
does `make_context(backend, window=None, gpu_preference=gpu)` +
`Renderer(...)` with destroy-on-failure, then applies proposals / reuse /
lobe_samplers post-construction.

**`skinny-web` (`web_app.py` main, ~700–760):**
parse (`add_render_flags(parser, proposals=False, resolution=False,
mcp=False)`) → `resolve_execution_mode` **first** → `validate_render_flags` →
`reject_spectral_unsupported` only → `select_backend(args.backend)` (no
persisted) with `"skinny-web: …"` prefix → stash resolved values in module
globals → per-session, on a **background thread**,
`make_context(_BACKEND, window=None, gpu_preference=...)` + `Renderer(...)`
inside `SkinnySession.initialize()`.

### The genuine deltas (everything else is copy drift)

| Axis | skinny | skinny-gui | skinny-render | skinny-web |
|---|---|---|---|---|
| Persisted settings feed resolution (integrator, backend, encoding, …) | yes | yes | no | no |
| Explicit `reject_sppm/mlt/mcp` re-checks (persisted-integrator cases) | yes | yes | no (CLI-keyed `validate_render_flags` suffices — no persistence) | no |
| `select_backend(persisted=…)` | yes | yes | no | no |
| Backend-failure `SystemExit` prefix (the guards print a fixed `skinny:` on all four) | `skinny:` | `skinny-gui:` | `skinny-render:` | `skinny-web:` |
| Flag-set knobs (`add_render_flags`) | full | `proposals=False` | `resolution=False, mcp=False` | `proposals=False, resolution=False, mcp=False` |
| Context surface | GLFW window, in `main` | deferred to Qt render thread | `window=None` + `gpu_preference`, immediate | `window=None` + `gpu_preference`, deferred per session, background thread |
| Post-construction persisted overrides on `Renderer` | yes (6 keys) | via `MainWindow`/render thread | no (explicit ctor args) | no (globals → ctor args) |
| validate-vs-resolve order | validate → resolve → rejects | validate → resolve → rejects | resolve → validate | resolve → validate |

The last row is pure drift: `resolve_execution_mode`'s docstring — and the
`render-cli` spec itself ("The validation SHALL run after the execution mode is
resolved", `openspec/specs/render-cli/spec.md`) — say resolution runs *before*
`validate_render_flags`; the interactive pair does the opposite (out of spec
today) and compensates with explicit `reject_*` re-checks. The two orders are
currently refusal-equivalent only by a fragile mechanism: when the mode is not
explicit, the interactive pair passes the still-unresolved string `"auto"` into
the guards, which compare `(execution_mode or "megakernel") == "megakernel"`
(`cli_common.py:232`, `:275`) — `"auto"` is truthy and ≠ `"megakernel"`, so the
pre-resolution guard is a silent no-op and the post-resolution `reject_*`
re-checks do the real work. An accident, not a contract (any guard that ever
treats `"auto"` differently breaks it). The builder picks **one** canonical
order (resolve → validate → persisted-aware rejects), which is a superset of
both, provably preserves every refusal, and restores the interactive pair to
`render-cli` conformance.

## Goals / Non-Goals

**Goals**

- One module (`src/skinny/bringup.py`) owning the canonical sequence
  args → validated plan → `(ctx, renderer)`.
- Staged: the validate/resolve step is separable from context/renderer
  construction, so Qt and web can plan at `main()` and construct later
  (render thread / per-session background thread).
- Per-front-end knobs only for the real deltas above: `prog` (refusal
  prefix), `persisted` (settings dict or `None`), context surface
  (`window` / `gpu_preference`), and an injectable context factory.
- Hostless test of the full gating sequence against a stub context factory.
- Behavior-preserving: same refusals, same messages, same startup behavior
  on all four. This is exactly the standing "changes apply across ALL
  front-ends consistently" preference, made structural.

**Non-Goals**

- No changes to `render_session.py` / Qt threading
  (`RenderCommandQueue` / `QtRendererConfig` stay Qt-only; the builder sits
  *below* it — other front-ends are never routed through it).
- No changes to `cli_common.py` guards or `backend_select.py` resolution —
  the builder composes them; their hostless tests stay authoritative for the
  pieces.
- No new persistence, no settings-format change, no flag changes.
- Not absorbing the six per-key persisted `Renderer` overrides in `app.py`
  (neural_handoff/trainer/precision/…) in v1 — they mutate a constructed
  `Renderer` and only `skinny` needs them; folding them in is a follow-up
  once the sequence itself is shared.

## Decisions

**D1 — Two-stage builder: `plan_bringup()` then `BringupPlan.create()`.**

```python
plan = plan_bringup(args, prog="skinny-gui", persisted=saved_settings)
# … later, possibly on another thread / per session:
ctx, renderer = plan.create(window=None, width=1280, height=720,
                            gpu_preference=gpu,
                            context_factory=make_context,  # default
                            **renderer_kwargs)
```

**Renderer-construction seam (kwargs pass-through).** The four sites construct
`Renderer` with four different kwargs sets (`app.py:590–604`,
`web_app.py:117–139`, `ui/qt/viewport.py:70–104` `_build_renderer`,
`headless.py:157–167`). The split is explicit:

- **Plan-carried fields** — inputs the guard sequence resolved or vetted, the
  same on every front-end: `execution_mode`, `spectral`, `bdpt_walk`
  (via `resolve_walk`), `neural_config` (via `neural_config_from_args`),
  `backend`. `create` passes these to `Renderer` itself.
- **`**renderer_kwargs` splat** — front-end-specific constructor inputs the
  builder does not interpret: `usd_scene_path`, `use_usd_mtlx_plugin`,
  `shader_dir`/`hdr_dir`/`tattoo_dir`, `neural_handoff` / `neural_trainer` /
  `train_precision` (only `skinny` passes them at construction). `create`
  forwards them verbatim.
- **Post-construction state stays in the front-ends** — everything applied to
  the constructed `Renderer` today stays at the call sites: `skinny`'s six
  persisted overrides, web's `integrator_index`/`reuse_index`/lobe samplers,
  Qt's `_requested_backend`/`_online_training_requested`/post-hoc
  integrator+reuse, headless's proposals/reuse/lobe_samplers. The builder ends
  at a constructed `(ctx, renderer)`.

- `plan_bringup` runs, in canonical order:
  `startup_integrator_name` (persisted-aware iff `persisted` given) →
  `resolve_execution_mode` → `validate_render_flags` →
  `reject_sppm_without_wavefront` → `reject_mlt_unsupported` →
  `reject_spectral_unsupported` → `reject_mcp_unsupported` →
  `select_backend(..., persisted=persisted.get("backend"))`, wrapping the
  `RuntimeError` as `SystemExit(f"{prog}: {exc}")`. Result is a small
  frozen dataclass (`backend`, `execution_mode`, `startup_integrator`,
  plus the pass-through fields `create` needs).
- `create` calls the context factory then `Renderer(...)`, with the
  existing `HeadlessRenderer` destroy-on-failure pattern
  (`except: ctx.destroy(); raise`).
- *Alternative rejected — single `bring_up(args) -> (ctx, renderer)` call:*
  Qt and web cannot use it (they construct on other threads, per session),
  so two of four front-ends would keep hand-copied sequences — the whole
  point lost. Staging is the minimum that covers all four.
- *Alternative rejected — builder-object with fluent setters:* four call
  sites, fixed sequence; a function + dataclass is smaller and un-driftable.

**D2 — Persistence participation is the caller's one knob, not builder
logic.** `persisted=None` (headless, web) reproduces today's non-interactive
behavior exactly (CLI-keyed guards only, `select_backend` without
`persisted=`); passing the settings dict (skinny, skinny-gui) reproduces the
interactive behavior (persisted integrator feeds `startup_integrator_name`,
persisted backend feeds `select_backend`). Precedence flag > env > persisted >
auto is untouched because it lives inside `select_backend` /
`resolve_execution_mode` already — the builder only decides *whether* the
persisted value is offered, mirroring which front-ends persist today.

**D3 — Canonical guard order is resolve → validate → persisted rejects.**
Matches the `resolve_execution_mode` docstring and the non-interactive pair;
strictly stronger than the interactive pair's order because the persisted
`reject_*` re-checks run unconditionally when `persisted` is given. The
hostless test pins this order (and the refusal messages) per front-end
configuration.

**D4 — Stub-context seam = the `context_factory` parameter on `create`.**
Default `backend_select.make_context`; tests pass a stub recording
`(backend, window, width, height, gpu_preference)` and returning a fake ctx
with `destroy()`. No monkeypatching, no test-only module. `select_backend`'s
GPU probe is avoided in tests the same way it is today in
`tests/` for `backend_select` (env pin / monkeypatched `metal_available`).

**D5 — MCP, GLFW, Qt, server wiring stay in the front-ends.** The builder
ends at `(ctx, renderer)`. `reject_mcp_unsupported(False)` is a no-op, so the
builder can call it unconditionally with the front-end's mcp flag
(defaulting absent attribute → `False`) — no knob needed.

## Risks / Trade-offs

- **[Risk] Silent refusal-message drift during migration** (a reworded
  `SystemExit` breaks users' scripts/greps). → Mitigation: hostless test
  asserts exact messages incl. `prog` prefix per front-end *before* the first
  front-end migrates; each migration is diffed against that pinned baseline.
- **[Risk] Canonical order changes behavior for an edge combo** (e.g.
  explicit `--integrator sppm --execution-mode megakernel` on `skinny`, where
  today `validate_render_flags` fires before resolution). → Mitigation: the
  test enumerates the guard matrix (integrator × mode × spectral × mlt ×
  persisted-vs-CLI) and asserts refusal/acceptance identical to the current
  per-front-end sequences, captured as fixtures from the code *before*
  refactoring.
- **[Risk] Canonical order moves the interactive refusals after
  `ensure_dirs()` / `load_settings()` side effects.** Today `skinny` runs
  `validate_render_flags` (`app.py:515`) *before* `ensure_dirs()` (`:519`), so
  invalid flags refuse before any `~/.skinny` touch; under the builder the
  plan needs the persisted settings first, so `ensure_dirs()`'s mkdir happens
  before a flag refusal. → Mitigation: accepted and named as the one
  deliberate behavior delta — refusal *outcomes* and messages are unchanged,
  and `load_settings()` never raises (`settings.py:49` returns `{}` on a
  missing/corrupt/non-dict file), so the observable difference is only that
  `~/.skinny/` may be created on a refused launch.
- **[Risk] `plan.create()` on a non-main thread (web) behaves differently
  from the inline code.** → Mitigation: `create` is a verbatim relocation of
  `HeadlessRenderer.__init__`'s factory+destroy-on-failure body; web's
  session `initialize()` keeps its own logging/error capture around the call.
- **[Trade-off] `skinny`'s six post-construction persisted overrides stay
  local** — the builder does not fully own `skinny`'s bring-up in v1. Accepted:
  they are one front-end's runtime-mutable state, not sequence; folding them
  in now would grow the plan for one caller (YAGNI, recorded as follow-up).

## Migration Plan

One front-end per step, refusal-parity checked after each; each step
independently revertible.

1. Land `bringup.py` + hostless tests (stub factory, pinned messages, guard
   matrix) with **no** front-end migrated. Tests encode today's behavior.
2. Migrate `skinny-render` (simplest: no persistence, no deferral) —
   `HeadlessRenderer` gains a plan-consuming path; CLI output byte-identical.
3. Migrate `skinny-web` — plan in `main()`, `plan.create(...)` inside
   `SkinnySession.initialize()`; module globals shrink to the plan object.
4. Migrate `skinny` — plan in `main()`, `create` after GLFW window; the six
   persisted `Renderer` overrides stay where they are.
5. Migrate `skinny-gui` — plan in `main()` before `QApplication`; the plan
   (not loose args) is handed to `MainWindow` → render thread, which calls
   `create` where the context is built today. `render_session.py` unchanged.
6. Docs: `docs/Architecture.md` (bring-up section), `README.md` untouched
   unless wording references per-front-end resolution.

Refusal-parity check per step: run each front-end's `--help` and the refusal
matrix (sppm+megakernel, mlt+spectral-off-envelope, spectral out-of-envelope,
mcp on unsupported, bad backend) and diff stderr against the pre-migration
capture.

## Open Questions

- Should the plan dataclass also carry `encoding` resolution (the
  CLI/env/persisted precedence currently duplicated in `skinny` and
  `skinny-gui`)? Leaning yes-in-v1 if it stays a pure-input resolution (it
  is: same guard pattern, two call sites) — decided at implementation review.
- Does `skinny-gui` want the plan threaded through `QtRendererConfig` or
  passed alongside it? Threading through touches `render_session.py`
  signatures (non-goal); passing alongside keeps the non-goal intact —
  default to alongside.
