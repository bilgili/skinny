## 1. Fix the enable gate

- [x] 1.1 `src/skinny/app.py` (GLFW): gate `enable_online_training()` on `renderer._backend_render_ready` instead of `renderer.descriptor_sets is not None`.
- [x] 1.2 `src/skinny/ui/qt/viewport.py` (`_maybe_online_training`): defer on `not self.renderer._backend_render_ready` instead of `descriptor_sets is None`.

## 2. Regression test

- [x] 2.1 `tests/` assert `_backend_render_ready` is true for a Metal wavefront renderer stand-in with `_scene_bindings` set and `descriptor_sets=None` (the case the old gate failed), false without scene bindings, and still requires `descriptor_sets` on a Vulkan stand-in.

## 3. Verification

- [x] 3.1 `ruff check` clean on the touched files; the new + existing online-training tests pass; `openspec validate online-training-metal-enable-gate --strict` passes.
