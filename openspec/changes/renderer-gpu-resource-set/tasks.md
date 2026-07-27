# Tasks: renderer-gpu-resource-set

## 1. Baseline capture (no code moves)

- [ ] 1.1 Instrument `_init_gpu` once to dump every allocated resource:
      attribute name, class, size/format inputs, and allocation order. Capture
      on Vulkan and on Metal (they differ in 4 places). Store as a permanent
      hostless fixture.
- [ ] 1.2 Dump the descriptor-write sequence from `_create_descriptors` and
      each `_rebind_*` (binding number, resource, write order) and the
      `_build_metal_binds` name→resource map. Same fixture.
- [ ] 1.3 Diff the `cleanup` destroy list against the 1.1 inventory and record
      every resource allocated but not destroyed today. These are pre-existing
      leaks; list them in the change and fix them as part of step 4 (do not
      silently preserve them).

## 2. Module with allocation only

- [ ] 2.1 Add `src/skinny/gpu_resources.py` with the declaration record type
      and the full declaration list transcribed from 1.1.
- [ ] 2.2 `SceneResourceSet.__init__` allocates from the declarations; the
      renderer constructs it and reads resources off it. `_init_gpu` becomes a
      call plus the residual non-resource setup.
- [ ] 2.3 Hostless test: recording context, assert inventory equals the 1.1
      fixture including order.

## 3. Binding behind one step

- [ ] 3.1 Move `_create_descriptors` and the five `_rebind_*` bodies into the
      set's Vulkan binding adapter, preserving write order exactly.
- [ ] 3.2 Move `_build_metal_binds` into the Metal binding adapter, fed by the
      same declarations.
- [ ] 3.3 Delete the five `is_metal` / `descriptor_sets is None` early-returns.
- [ ] 3.4 Hostless test: both adapters cover the same declarations; binding
      numbers and names match the 1.2 fixture.

## 4. Destruction paired with declaration

- [ ] 4.1 `close()` destroys from the declaration list; delete the by-name
      destroy body in `cleanup`.
- [ ] 4.2 Fix the leaks found in 1.3; note each in the change's notes.
- [ ] 4.3 Hostless test: allocated set equals destroyed set.

## 5. Growth sites

- [ ] 5.1 Route `_ensure_mesh_buffer_capacity`, `_sync_volume_grid` /
      `_rebind_volume_descriptor`, `_update_texture_pool_descriptors`, and
      `_rewrite_size_dependent_descriptors` through the set.
- [ ] 5.2 Assert no descriptor write remains outside the set (source grep gate,
      same shape as the `shader-variant-key` grep gate).

## 6. Gates

- [ ] 6.1 `ruff check src/` clean; full hostless `pytest` green.
- [ ] 6.2 GPU smoke on Metal: megakernel and wavefront, one scene each.
- [ ] 6.3 GPU smoke on Vulkan: megakernel, one scene.
- [ ] 6.4 Parity matrix self-consistency gate unchanged — identical, not close.
- [ ] 6.5 `tests/test_metal_cleanup.py` including the gpu-marked kill harness
      (context lifecycle changed).
- [ ] 6.6 Docs: `docs/Architecture.md` module map + carve-out section.
- [ ] 6.7 `openspec validate renderer-gpu-resource-set --strict`.
