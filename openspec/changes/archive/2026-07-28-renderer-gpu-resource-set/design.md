# Design: renderer-gpu-resource-set

## Context

Three regions of `renderer.py` describe the same inventory three times.

**Allocation** — `_init_gpu` (`:3671-4235`): ~70 `StorageBuffer` /
`StorageImage` / `SampledImage` / `UniformBuffer` / `HostStorageBuffer`
constructions assigned to `self.<name>`. Four backend branches: semaphores and
fences only on Vulkan (`:4202`, `:4224`), neural handoff kind (`:3843-3844`),
mtlx skin slot stride 256 on Metal (`:3707`).

**Binding** — `_create_descriptors` (`:4236-4700`, 465 lines) plus five
`_rebind_*` methods, each of which begins with an `is_metal` (or
`descriptor_sets is None`) early return. The Metal counterpart is
`_build_metal_binds` (`:9745`, ~80 lines) mapping 40+ shader global names to
the same resources. Two expressions of one fact: *which resource sits at which
slot*.

**Destruction** — `cleanup` (`:11505-11605`): ~35 `.destroy()` calls by
attribute name, guarded by `getattr(...)`. Resources allocated but not listed
here leak; nothing detects it.

**Growth** — `_ensure_mesh_buffer_capacity` (`:4877`),
`_rewrite_size_dependent_descriptors` (`:11022`), `_sync_volume_grid` /
`_rebind_volume_descriptor` (`:8484-8548`), `_update_texture_pool_descriptors`
(`:8432`): four more sites where a reallocation must be followed by a rebind.

## Goals / Non-Goals

**Goals**
- One declaration per resource carrying allocation, binding and destruction.
- One backend branch, at the binding step, consuming one declaration list.
- Hostless assertion of the inventory and of alloc/destroy pairing.
- Byte-identical GPU behaviour: same buffers, sizes, binding numbers, order.

**Non-Goals**
- Designing the general GPU backend interface. That is `gpu-backend-adapter`;
  this change is its first honest consumer, not its author.
- Changing descriptor binding numbers, or the binding map in
  `docs/Architecture.md`.
- Touching per-frame dispatch, `_pack_uniforms`, or the wavefront passes.

## Decisions

### D1 — Declaration list, not a class per resource

A resource is a small record: name, kind, size/format inputs, Vulkan binding
number (or `None`), Metal global name (or `None`), and a per-frame vs
per-scene lifetime tag. The set is a list of these plus the code that walks
it. Rejected: one class per resource (70 classes for no leverage) and a
generic registry keyed by string (loses the "declared once" property that the
whole change is for).

### D2 — Binding numbers stay literal in the declaration

`docs/Architecture.md` holds the authoritative binding map and the shader
declares the same numbers in `bindings.slang`. This change does **not** derive
them; it relocates them next to the allocation they belong to. Deriving them
from the shader is `shader-byte-layouts` territory and is a separate change.

### D3 — Backend split is one adapter call at bind time

The set holds one `bind(target)` step. On Vulkan it emits the descriptor
writes currently in `_create_descriptors` / `_rebind_*`; on Metal it fills the
name→resource dict currently built by `_build_metal_binds`. Everything else in
the set is backend-neutral. The five `is_metal` early-returns disappear because
the Metal adapter simply has no descriptor step, rather than each method
opting out.

### D4 — The renderer keeps attribute access

~120 sites inside `renderer.py` read `self.<resource>` directly. Renaming them
all would swamp the diff and break the bit-identity review. The set exposes
its resources as attributes and the renderer either holds the set and reads
`self._gpu_set.<name>`, or keeps thin properties. Prefer the former for new
code, the latter where the diff would otherwise be mechanical noise — decided
per region during implementation, recorded in tasks.

### D5 — Fake-context test, not a device test

The hostless test constructs the set against a recording context whose
resource constructors return sentinels. It asserts: every declaration is
allocated exactly once; `close()` destroys exactly the allocated set; the
Vulkan binding numbers are unique and match the pre-refactor list; the Metal
names are unique and cover the same declarations. This is the first consumer
of the recording adapter idea and can be local to the test until
`gpu-backend-adapter` lands.

## Risks / Trade-offs

- **Risk: a resource silently changes size or format during the move.** Gate:
  capture a pre-refactor golden of (name, kind, size, format, binding) for all
  ~70 resources by instrumenting `_init_gpu` once, and pin it as a permanent
  test fixture — the same "record reality, not the new module's own table"
  discipline used by `shader-variant-key-module`.
- **Risk: descriptor write *order* matters on some driver path.** The Vulkan
  adapter preserves the existing write order verbatim; the declaration list is
  ordered to reproduce it. The golden includes order.
- **Trade-off: the set is large.** ~70 declarations is a big literal. It is
  still one place instead of three, and the deletion test says the complexity
  is real — it does not vanish if the module is removed.
- **Conflict with `frame-plan-split`:** that change edits `render()` /
  `render_headless()`, which read these resources. Land this first; the frame
  plan then consumes a stable set interface.

## Migration Plan

1. Instrument and capture the golden inventory (no code moves).
2. Add the module and declarations; renderer delegates allocation only.
3. Move binding (both adapters) behind `rebind`; delete the five early-returns.
4. Move destruction; assert alloc/destroy pairing in the test.
5. Move the four growth/rebind sites onto the set.
6. GPU smoke: megakernel + wavefront on Metal, megakernel on Vulkan, plus the
   parity matrix self-consistency gate — must be unchanged, not merely close.

## Open Questions

- Do the per-frame sync objects (fences, semaphores, command buffers) belong
  in the same set, or in a separate frame-resources module? Leaning: same set,
  tagged with a per-frame lifetime, because `cleanup` already destroys them
  together — revisit if the frame-plan change wants them separately.
