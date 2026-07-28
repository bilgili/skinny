# Change: renderer-gpu-resource-set

## Why

GPU resource ownership is the largest cluster left in `renderer.py` and the
only one with **zero external callers** — 1,342 lines that no other module
imports, spread over three non-adjacent regions:

- `_init_gpu` (`renderer.py:3671-4235`, 565 lines) allocates ~70 buffers and
  images onto `self` by name, with 4 `is_metal` branches, including a local
  second copy of the backend predicate (`_is_metal`, `_ext_neural`,
  `_shared_neural` at `:3842-3844`).
- `_create_descriptors` + the five `_rebind_*_descriptors` methods
  (`:4236-4912`, 676 lines) write Vulkan descriptor sets; five of them
  early-return on Metal (`:4709`, `:4760`, `:4793`, `:4829`, `:4906`), and the
  Metal equivalent is a separate 80-line bind-by-name table
  (`_build_metal_binds`, `:9745`).
- `cleanup` (`:11505-11605`, 101 lines) destroys ~35 of those resources by
  name, 7,800 lines away from where they were allocated.

Allocation and destruction are a hand-maintained pair. Adding a GPU resource
means editing three regions of one 11,604-line module and remembering the
third; a missed line is a leak that no test can see, because none of this is
reachable without a device. `_ensure_mesh_buffer_capacity`, the volume-grid
rebind, the texture-pool rebind, and `_rewrite_size_dependent_descriptors`
(`:11022`) add four more places where "this resource grew" must be reflected.

The cluster is the textbook deepening candidate: large implementation, no
external interface at all today, and by the deletion test its complexity is
real — deleting it concentrates the inventory rather than moving it.

## What Changes

- Add one module owning the renderer's GPU resource inventory: a
  `SceneResourceSet` constructed from a context plus the size/capacity inputs
  it needs, exposing a small interface — construct, `rebind(scene)`, `close()`
  — and, per resource, one declaration that carries its allocation, its
  binding (descriptor write on Vulkan, name entry on Metal), and its
  destruction together.
- Move `_init_gpu`, `_create_descriptors`, the five `_rebind_*_descriptors`,
  `_rewrite_size_dependent_descriptors`, `_ensure_mesh_buffer_capacity`,
  `_build_metal_binds`, and the destroy list in `cleanup` into it. The renderer
  keeps attribute access to the individual resources (`self._gpu_set.accum`
  etc. or re-exported properties) so no call site outside changes.
- The ~10 `is_metal` branches inside these regions become one branch at the
  set's binding step — the Vulkan adapter writes descriptor sets, the Metal
  adapter fills the by-name table, from the **same** declaration list.
- Add a hostless test that constructs the set against a recording fake context
  and asserts the inventory: every declared resource is allocated, every
  allocated resource is destroyed by `close()`, and the Vulkan binding numbers
  and the Metal binding names cover the same declaration set.
- Pure refactor. Same resources, same sizes, same binding numbers, same
  descriptor writes, same order. No shader change, no behaviour change.

## Capabilities

### Modified Capabilities

- `renderer-module-structure`: a further carve-out stage — GPU resource
  allocation, binding and destruction become one module with a paired
  declaration per resource, under the existing bit-identity requirement for
  carve-out stages.

## Impact

- New: `src/skinny/gpu_resources.py`, one hostless test module.
- Modified: `src/skinny/renderer.py` (three regions removed, one member added;
  ~1,340 lines net out), `src/skinny/metal_compute.py` and
  `src/skinny/vk_compute.py` only if a binding helper moves.
- Unchanged: every call site outside `renderer.py`, all descriptor binding
  numbers, `docs/Architecture.md`'s binding map contents (its *location* gains
  a pointer to the new module).
- Docs: `docs/Architecture.md` module map + a line in the carve-out section.
- **Ordering hazard**: this change and `renderer-pure-core-extraction` and
  `frame-plan-split` all edit `renderer.py`; they touch disjoint line ranges
  but must be sequenced (see design). `renderer-pure-core-extraction` is
  cheapest and should land first.
