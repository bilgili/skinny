## Why

The headless Python API selects Vulkan, and only Vulkan. Every other runner
resolves the GPU backend through `backend_select.select_backend`, so
`--backend auto` (and the `SKINNY_BACKEND` environment variable) gives them
native Metal on an Apple-Silicon host. A direct call to
`HeadlessRenderer(w, h)` keeps the hard-coded default `backend="vulkan"`, so an
Apple-Silicon user gets the MoltenVK path, needs `VULKAN_SDK` and
`DYLD_LIBRARY_PATH` in the environment, and cannot ask for `auto` at all —
`backend="auto"` reaches `make_context` unresolved and raises
`unknown backend 'auto'`.

## What Changes

- `HeadlessRenderer.__init__` resolves its `backend=` argument through
  `backend_select.select_backend(backend, persisted=None)` before it builds the
  internal `BringupPlan`. The argument now accepts `auto`, `metal`, and
  `vulkan`, and it honours the `SKINNY_BACKEND` environment variable at the
  same precedence every other runner uses.
- The default becomes `backend=None` — unset, deferring to the environment and
  then to `auto`, exactly as argparse's `--backend default=None` does on the
  four front-ends. (Not the literal string `"auto"`, which as an explicit
  argument would outrank `SKINNY_BACKEND`.) On an Apple-Silicon host with a
  Metal device the headless API therefore renders on native Metal. **BREAKING**
  for a caller that relied on the implicit Vulkan default; such a caller passes
  `backend="vulkan"` to keep the old context.
- The three module-level wrappers (`render_to_array`, `render_scene`,
  `render_animation`) inherit the new default with no signature change, because
  they construct a `HeadlessRenderer` with the default backend.
- An explicit `backend="metal"` on a host with no Metal device raises the
  existing clear `RuntimeError` from `select_backend` rather than degrading.
- The `plan=` path (the `skinny-render` CLI) is untouched: `plan_bringup`
  already ran `select_backend`, so the plan carries a resolved backend and no
  second resolution happens.

## Capabilities

### New Capabilities

None. The behaviour belongs to an existing capability.

### Modified Capabilities

- `render-cli`: the headless Python API resolves its backend through the shared
  selector, accepts `auto`, honours `SKINNY_BACKEND`, and defaults to `auto`.
  A new requirement records the resolution and its precedence; the existing
  headless requirements are unchanged.

## Impact

- Code: `src/skinny/headless.py` (`HeadlessRenderer.__init__` only).
- Tests: a hostless test of the resolution and its precedence; the GPU headless
  tests that use the default backend now run on the resolved backend.
- Docs: `docs/PythonAPI.md` (the `HeadlessRenderer` signature and the Vulkan-SDK
  note), `docs/FrontEnds.md` (the headless entry description).
- Not in scope: `renderer.py` still imports the `vulkan` module at module load,
  so the `vulkan` Python package stays an import-time requirement on every
  host. Removing that import is separate work.
