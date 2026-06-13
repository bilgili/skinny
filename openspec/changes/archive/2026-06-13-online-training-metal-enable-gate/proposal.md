## Why

Online training never enables on the native Metal backend. Both interactive
front-ends defer enabling until "the scene is built", but they detect that with
`renderer.descriptor_sets` — a Vulkan-only object allocated by
`vkAllocateDescriptorSets`. The native Metal backend binds resources by name and
never allocates Vulkan descriptor sets, so `descriptor_sets` is `None` for the
whole session. The gate is therefore never satisfied on Metal:

- `app.py` (GLFW): `… and renderer.descriptor_sets is not None` is never true.
- `ui/qt/viewport.py` (`skinny-gui` worker): `if renderer.descriptor_sets is
  None: return` returns every frame.

So the documented Mac combo (`--backend metal --execution-mode wavefront
--online-training`) arms but never calls `enable_online_training()`: no trainer,
no `ACTIVE` log, no STOPPED summary, an empty GUI status segment — and the new
configuration matrix misleadingly shows `APPROVED`, because `can_online_train()`
checks the execution mode and the active neural proposal, not scene readiness.
This is a latent bug from `online-training-trigger` (whose live drain targeted
the Vulkan/NVIDIA box) exposed now that Metal record-drain + Metal training ship.

## What Changes

- Gate enabling on the **backend-aware** readiness signal `_backend_render_ready`
  instead of `descriptor_sets`, in both front-ends. On Vulkan it still requires
  the descriptor sets (behavior unchanged); on Metal it is the scene-bindings
  readiness both backends set — the same signal `online_training_tick()` already
  self-guards on, so enable and drain now agree.

## Capabilities

### Modified Capabilities
- `online-training-control`: the "scene is built" precondition for *enabling*
  the loop is specified as a backend-aware readiness check, so enabling works on
  the native Metal backend (which never allocates Vulkan descriptor sets), not
  only on Vulkan.

## Impact

- **Code:** `src/skinny/app.py` and `src/skinny/ui/qt/viewport.py` — swap the
  `descriptor_sets` readiness test for `renderer._backend_render_ready`.
- **Tests:** a regression test asserting `_backend_render_ready` is true for a
  Metal wavefront renderer with scene bindings but no descriptor sets (the case
  the old gate failed), and still requires descriptor sets on Vulkan.
- **Behaviour:** online training now actually starts on `--backend metal`; the
  `ACTIVE`/STOPPED logs and GUI status appear. Vulkan is unchanged.
- **Risk:** low — `_backend_render_ready` is the existing canonical readiness
  property; the change only widens the enable gate to match the drain guard.
