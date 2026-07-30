## Context

`skinny.headless.HeadlessRenderer` has two construction paths.

1. **The CLI path.** `skinny-render`'s `main()` calls
   `bringup.plan_bringup(ns, prog="skinny-render")`. That runs every refusal
   guard and ends with `select_backend`, so the returned `BringupPlan` carries a
   resolved backend (`"vulkan"` or `"metal"`). `main()` hands the plan in as
   `plan=`, and `HeadlessRenderer` calls `plan.create(...)`.
2. **The direct-API path.** A Python caller (a test, the parity harness, a
   script) passes no plan. `HeadlessRenderer.__init__` then builds a
   `BringupPlan` itself from its keyword arguments, with the signature default
   `backend: str = "vulkan"`.

Path 2 never reaches `select_backend`. Three consequences follow. The default is
Vulkan on every host, so an Apple-Silicon user gets MoltenVK and must export
`VULKAN_SDK` and `DYLD_LIBRARY_PATH`. `SKINNY_BACKEND` is ignored. And
`backend="auto"` is not a legal value: the unresolved string travels to
`backend_select.make_context`, which raises
`unknown backend 'auto' (expected 'vulkan' or 'metal')`.

Callers that already pass an explicit backend are unaffected by any of this.
`pbrt/parity.py` calls `select_backend()` itself and threads the result in; the
Metal parity tests pass `backend="metal"`. The affected callers are the ones
that use the default: `tests/test_headless_api.py`, the module-level
`render_to_array` / `render_scene` / `render_animation` wrappers, and any user
script.

## Goals / Non-Goals

**Goals:**

- The headless Python API resolves its backend through the one shared selector,
  with the same precedence as every other runner.
- `HeadlessRenderer(w, h)` on an Apple-Silicon host with a Metal device renders
  on native Metal.
- `backend="auto"` becomes a legal argument value.
- The CLI path keeps exactly one backend resolution.

**Non-Goals:**

- Removing the module-load `import vulkan` in `renderer.py`. The `vulkan`
  Python package stays an import-time requirement on every host; only the
  Vulkan *runtime* (`DYLD_LIBRARY_PATH`) stops being needed for a render.
- New keyword arguments on the module-level wrapper functions. They construct a
  `HeadlessRenderer` with the default, so the new default reaches them without a
  signature change.
- Any change to guard order, to `bringup.plan_bringup`, or to the persisted
  settings. The headless front-end stays persistence-free (`persisted=None`).
- Any change to the rendered image. Backend choice does not change the
  estimator; cross-backend image equality is the parity harness's subject, not
  this change's.

## Decisions

### Decision 1 — Resolve at the plan-construction site, not at `create`

`HeadlessRenderer.__init__` calls
`select_backend(backend, persisted=None)` inside the `plan is None` branch, and
puts the resolved name in the `BringupPlan`.

The alternative is to resolve inside `BringupPlan.create`. Rejected: `create`
must stay a pure construction step that re-runs no guard, because two
front-ends call it later and on another thread (`frontend-bringup`, requirement
"Staged bring-up"). `select_backend` probes and closes a GPU device and can
raise a refusal, so it belongs to the plan step. Resolving in `create` would
also resolve twice on the CLI path.

A second alternative is to keep the raw string in the plan and let
`make_context` resolve. Rejected: `make_context` is the construction seam for
both backends and takes an already-resolved name on every other path; teaching
it to resolve would put two resolution points in the tree.

### Decision 2 — The default is `None`, which resolves to `auto`, not a preserved `vulkan`

Matching the other runners means matching their default. `skinny`,
`skinny-gui`, `skinny-web`, and `skinny-render` all default to `auto`, which is
what "similar to other skinny runners" asks for. Keeping `vulkan` as the
default and only *accepting* `auto` would leave every existing caller — the
wrappers, the tests, user scripts — on MoltenVK unless it opts in, which is the
problem this change exists to fix.

This is a breaking change for a caller that depends on the implicit default. The
migration is one keyword argument, `backend="vulkan"`, and the proposal records
it.

The signature default is `None`, not the string `"auto"`, and the distinction is
load-bearing. `select_backend` reads `prefer or env or persisted or "auto"`, so a
literal `"auto"` in the `prefer` position is a truthy explicit choice that
**outranks `SKINNY_BACKEND`**. `None` is what makes the environment participate,
and it is exactly what argparse hands the four front-ends
(`--backend … default=None` in `cli_common.add_render_flags`). A caller who
passes `backend="auto"` explicitly still gets auto resolution overriding the
environment — the same thing `--backend auto` does on the command line. Both
behaviours are recorded as scenarios.

### Decision 3 — One call, one precedence chain

`select_backend(prefer, persisted=)` already owns the precedence
`prefer > SKINNY_BACKEND > persisted > auto`. Passing the keyword argument as
`prefer` and `persisted=None` therefore gives the headless API the documented
chain with no local re-implementation: an explicit argument wins, otherwise the
environment, otherwise `auto`.

`persisted=None` is deliberate and matches `bringup`'s treatment of
`skinny-render` — a non-interactive front-end reads no `~/.skinny/settings.json`
backend. The direct Python API is non-interactive by construction, so it
follows the same rule.

### Decision 4 — Failure behaviour is `select_backend`'s, unchanged

An explicit `backend="metal"` on a host with no Metal device raises the existing
`RuntimeError` naming the missing requirement. An unknown token raises the
existing `ValueError`. Both surface from the constructor, before any GPU context
exists, so there is nothing to tear down. `HeadlessRenderer` adds no wrapper and
no message of its own; the `SystemExit(f"{prog}: …")` wrapper stays in `bringup`
where the CLI needs it.

### Decision 5 — The hostless test stubs the selector and the construction step

The resolution is verifiable with no GPU. The test monkeypatches
`skinny.headless.select_backend` to record its arguments and return a fixed
name, and monkeypatches `BringupPlan.create` to capture the plan and return a
`(ctx, renderer)` pair of stubs. That covers the default, an explicit value, the
`SKINNY_BACKEND` path, and the "plan given ⇒ no resolution" case without a
device.

The GPU behaviour itself needs no new test: the existing headless GPU tests run
on the resolved backend after this change, which is the observable outcome.

## Risks / Trade-offs

- **A caller depends on the implicit Vulkan default and silently moves to
  Metal.** → The proposal marks the default change **BREAKING** and names the
  one-argument migration. The in-tree callers are enumerated above: every
  parity-harness and Metal-test call site already passes an explicit backend, so
  only `tests/test_headless_api.py` and the wrappers move, and both want the
  resolved backend.
- **`select_backend` probes a Metal device on every `HeadlessRenderer`
  construction under `auto` or `metal`.** → The probe creates and immediately
  closes a device; it already runs once per launch on all four front-ends. A
  caller that builds many contexts in a loop can resolve once itself and pass
  the resolved name, which is what `pbrt/parity.py` does.
- **A headless script written against Vulkan-only behaviour hits a Metal
  envelope limit** (UsdSkel GPU skinning falls back to CPU on Metal; wavefront
  indirect dispatch uses the readback fallback). → These are recorded
  compatibility-matrix rows, not failures, and both backends run the full
  megakernel and wavefront renderer. A script that needs the Vulkan-only path
  passes `backend="vulkan"`.
- **A GPU test that used to skip for a missing Vulkan runtime now runs.** → That
  is the intent. A silent skip reads as a pass, and this repo has been bitten by
  exactly that (`DYLD_LIBRARY_PATH` stripped ⇒ headless tests skipped silently).

## Migration Plan

1. Change the default and add the `select_backend` call in
   `HeadlessRenderer.__init__`.
2. Add the hostless resolution test.
3. Update `docs/PythonAPI.md` and `docs/FrontEnds.md`.
4. Run the hostless suite, then the headless GPU tests on this host (Metal) to
   confirm the default now resolves to Metal and renders.

Rollback is the one-line default and the one call — no data, no format, and no
persisted state is involved.

## Open Questions

None.
