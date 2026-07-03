# Tasks

## 1. Regression test first

- [x] 1.1 Add hostless `TestHeadlessReadinessGate` to `tests/test_headless_api.py`:
      wavefront-shaped renderer (`pipeline is None`,
      `_backend_render_ready True`) must render via `render_to_array` and
      `render_scene`; unready renderer must still raise the
      "no usable materials" `RuntimeError`. Confirm the wavefront cases FAIL
      before the fix.

## 2. Fix the gate

- [x] 2.1 Replace both `if self.renderer.pipeline is None:` checks in
      `src/skinny/headless.py` (`render_to_array`, `render_scene`) with
      `if not self.renderer._backend_render_ready:`.

## 3. Verify

- [x] 3.1 Hostless pytest green (`tests/test_headless_api.py` + default sweep).
- [x] 3.2 `openspec validate headless-wavefront-readiness-gate` passes.
