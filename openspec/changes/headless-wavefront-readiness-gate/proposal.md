# Headless wavefront readiness gate

## Why

`skinny-render --execution-mode wavefront` always fails with
`render pipeline failed to build — scene has no usable materials`, even when the
scene built fine. `HeadlessRenderer.render_to_array` / `render_scene` gate
readiness on `renderer.pipeline is None`, but in wavefront execution mode the
megakernel pipeline is intentionally never built (`scene_bindings_only` build,
`renderer.py` `_build_scene_pipeline`), so `pipeline` is `None` by design and
the gate misfires on every wavefront invocation. The parity harness
(`tests/pbrt/parity.py`) only works because it bypasses these methods and calls
`_prepare`/`_accumulate` directly.

## What Changes

- `HeadlessRenderer.render_to_array` and `render_scene` gate on
  `renderer._backend_render_ready` — the backend- and execution-mode-aware
  readiness signal already used by the interactive front-ends (`app.py`,
  `ui/qt/viewport.py`) — instead of `renderer.pipeline is None`.
- Hostless regression tests: a wavefront-shaped renderer (no megakernel
  pipeline, backend ready) renders through both methods; an unready renderer
  still raises the materials error.

## Impact

- Affected specs: `render-cli` (headless wavefront scenario now reachable via
  `render_scene`/`render_to_array`, not just the parity harness's internal
  path).
- Affected code: `src/skinny/headless.py` (two checks),
  `tests/test_headless_api.py` (new hostless regression class).
- No behavior change for megakernel mode: `_backend_render_ready` reduces to
  the pipeline check there.
