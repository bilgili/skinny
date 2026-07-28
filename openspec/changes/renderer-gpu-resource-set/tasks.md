# Tasks: renderer-gpu-resource-set

## 1. Baseline capture (no code moves)

- [x] 1.1 Instrument `_init_gpu` once to dump every allocated resource:
      attribute name, class, size/format inputs, and allocation order. Capture
      on Vulkan and on Metal (they differ in 4 places). Store as a permanent
      hostless fixture.
      → `tests/fixtures/gpu_resource_inventory.json`. **33 resources** (the
      proposal's "~70" was an over-estimate), allocated in the *same order* on
      both backends; the only divergence is `mtlx_skin_buffer` (2880 B on
      Vulkan vs 4352 B on Metal, the 256 B MSL slot ceiling). `--spectral`
      interleaves 7 more → 40.
- [x] 1.2 Dump the descriptor-write sequence from `_create_descriptors` and
      each `_rebind_*` (binding number, resource, write order) and the
      `_build_metal_binds` name→resource map. Same fixture.
      → 32 writes per descriptor set (×2 frames in flight). The write order is
      **not** allocation order and not binding order: 26 lands between 11 and
      12, 20 after 24, and 1/30/31 last.
- [x] 1.3 Diff the `cleanup` destroy list against the 1.1 inventory and record
      every resource allocated but not destroyed today. These are pre-existing
      leaks; list them in the change and fix them as part of step 4 (do not
      silently preserve them).
      → **No leaks.** Instrumenting `.destroy()` on every allocated resource
      and running the full construct→`cleanup()` cycle shows allocated ==
      destroyed on all three configurations (Vulkan RGB, Metal RGB, Metal
      spectral), with nothing destroyed twice. Step 4 therefore has no leak to
      fix; it makes the pairing *structural* rather than fixing a live bug.

## 2. Module with allocation only

- [x] 2.1 Add `src/skinny/gpu_resources.py` with the declaration record type
      and the full declaration list transcribed from 1.1.
- [x] 2.2 `SceneResourceSet.__init__` allocates from the declarations; the
      renderer constructs it and reads resources off it. `_init_gpu` becomes a
      call plus the residual non-resource setup.
      → D4 resolved: resource attributes are **read-only properties** forwarding
      to the set, so the ~120 `self.<resource>` reads are unchanged and
      assignment (which would reallocate without rebinding) is refused.
- [x] 2.3 Hostless test: recording context, assert inventory equals the 1.1
      fixture including order. → `tests/test_gpu_resources.py`, 21 tests.

## 3. Binding behind one step

- [x] 3.1 Move `_create_descriptors` and the five `_rebind_*` bodies into the
      set's Vulkan binding adapter, preserving write order exactly.
      The pool tally is now COUNTED from the active declarations instead of the
      hand-maintained "17 fixed + 3 + 2 = 22" comment, and reproduces it exactly.
- [x] 3.2 Move `_build_metal_binds` into the Metal binding adapter, fed by the
      same declarations. The renderer keeps a 2-line `_build_metal_binds` for the
      two globals that are NOT declared resources: `commonSampler` (binding 38)
      and `graphParamsCombined` (scene-graph lifetime).
- [x] 3.3 Delete the five `is_metal` / `descriptor_sets is None` early-returns.
      All five gone, plus `_rewrite_size_dependent_descriptors`. Pinned by
      `test_no_per_method_backend_guard_returns_remain`. The two stale tests in
      `test_metal_foundation.py` that asserted the old guards were rewritten to
      assert the new seam.
- [x] 3.4 Hostless test: both adapters cover the same declarations; binding
      numbers and names match the 1.2 fixture.

## 4. Destruction paired with declaration

- [x] 4.1 `close()` destroys from the declaration list; delete the by-name
      destroy body in `cleanup`.
- [x] 4.2 Fix the leaks found in 1.3; note each in the change's notes.
      1.3 found **no leaks**, so there was none to fix. Pairing did surface a
      real defect on an unexercised path and it is fixed: the Metal record-drain
      aliased `_drain_buffer` INTO the `record_buffer` slot, and `cleanup`
      destroyed **both** — a double free. Ownership now transfers via
      `adopt()`, and `cleanup` asks `_gpu_set.owns()` before freeing.
- [x] 4.3 Hostless test: allocated set equals destroyed set.

## 5. Growth sites

- [x] 5.1 Route `_ensure_mesh_buffer_capacity`, `_sync_volume_grid` /
      `_rebind_volume_descriptor`, `_update_texture_pool_descriptors`, and
      `_rewrite_size_dependent_descriptors` through the set.
      **Reality had three MORE growth sites than the task listed** and all are
      routed too: material-capacity growth (5 buffers at once), instance-capacity
      growth, and the Metal std-surface MSL-stride grow. Growth sites now state
      the new *capacity* and `regrow()` re-evaluates the declaration's own size
      formula, so the `cap * STRIDE + slack` arithmetic is no longer duplicated
      between the initial allocation and each grow.
- [x] 5.2 Assert no descriptor write remains outside the set (source grep gate,
      same shape as the `shader-variant-key` grep gate).
      `renderer.py` contains neither `VkWriteDescriptorSet` nor
      `vkUpdateDescriptorSets`. The two writes for non-declared resources (the
      graph-param buffer, the record-drain buffer) go through
      `SceneResourceSet.write_binding`.

## 6. Gates

- [x] 6.1 `ruff check src/` clean; full hostless `pytest` green.
      NOTE: bare `ruff check src/` checks **0 files** (root `.gitignore` is `*`),
      so it was run over an explicit tracked-file list — clean. Hostless sweep:
      2466 passed, 7 failed, all 7 pre-existing and identical to `main`.
- [x] 6.2 GPU smoke on Metal: megakernel and wavefront, one scene each.
      `suite/mat_diffuse` @128², 24 frames. Both **bit-identical to `main`**
      (maxdiff 0), and mega ≡ wave (maxdiff 0). Mesh buffers grew 352→39712 B,
      exercising `regrow` on a real device. Inventory re-captured post-change on
      Metal RGB matches the golden entry-for-entry (allocation order, sizes,
      Metal bind names, destroy order, alloc==destroy, no double destroy).
- [x] 6.3 GPU smoke on Vulkan: megakernel, one scene.
      **Bit-identical to `main`** (maxdiff 0). Descriptor writes re-captured
      post-change match the golden binding-for-binding, in order. The
      Metal-vs-Vulkan maxdiff of 1 is pre-existing (same on `main`).
- [ ] 6.4 Parity matrix self-consistency gate unchanged — identical, not close.
- [x] 6.5 `tests/test_metal_cleanup.py` including the gpu-marked kill harness
      (context lifecycle changed). 13 hostless + 3 gpu-marked, all pass.
- [x] 6.6 Docs: `docs/Architecture.md` module map + carve-out section.
      New § GPU resource inventory, a pointer from the Descriptor Binding Map,
      the carve-out landed-stages list, and a CLAUDE.md architecture entry.
- [ ] 6.7 `openspec validate renderer-gpu-resource-set --strict`.
