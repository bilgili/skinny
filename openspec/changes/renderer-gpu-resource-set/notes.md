# Implementation notes: renderer-gpu-resource-set

Records the evidence and the decisions the tasks ask to be written down.

## 1.1–1.3 — What the capture actually found

The baseline was taken by **instrumenting the live pre-change renderer**, not by
reading the source: `bs.resource_module` was wrapped in a recorder, every
resource's `destroy` was wrapped, `vk.vkUpdateDescriptorSets` was patched, and a
64×64 sceneless renderer was constructed and torn down on Vulkan RGB, Metal RGB
and Metal spectral. Result: `tests/fixtures/gpu_resource_inventory.json`.

Three findings contradict the proposal and are worth keeping:

1. **33 resources, not "~70".** The proposal estimated ~70 buffers and images in
   `_init_gpu`; the real count is 33 (40 under `--spectral`). Most of
   `_init_gpu`'s 565 lines were scalar state and seeding uploads, not
   allocation. This is why the module is ~600 lines rather than the ~1,300 the
   proposal's line accounting implied.

2. **The two backends allocate the same resources in the same order.** The
   proposal expected four divergences. Only *one* is an allocation difference —
   `mtlx_skin_buffer` is 2880 B on Vulkan and 4352 B on Metal (the 256 B MSL
   slot ceiling). The other three are not `_gpu` resources at all: the Vulkan
   semaphores and fences, and the neural handoff's `external`/`shared` flags,
   which are *kwargs* on a resource both backends allocate.

3. **There are no pre-existing leaks.** Task 1.3 expected to find resources
   allocated but never destroyed. Instrumenting `destroy()` and running the full
   construct → `cleanup()` cycle shows allocated == destroyed on all three
   configurations, with nothing destroyed twice. Step 4 therefore had no leak to
   fix; what it did was make the pairing *structural*, so a future addition
   cannot leak.

### The three orders genuinely differ

Worth stating because it is the one place the module looks redundant. The
descriptor-write order is neither allocation order nor binding order:

```
0 2 3 4 5 6 7 8 9 10 11 26 12 13 15 16 17 18 19 21 22 23 24 20 1 30 31 33 34 35 36 37
```

— binding 26 is written between 11 and 12, binding 20 after 24, and 1/30/31
last. Teardown is a third order again. All three are recorded once each and
pinned to the capture; none is derived from another.

## 4.2 — Defects surfaced by pairing allocation with destruction

No leaks existed, but pairing surfaced **one real double free** on a path the
capture did not exercise:

The Metal record-drain path (`dump_path_records`) destroys the dummy
`record_buffer` and assigns `self.record_buffer = self._drain_buffer`, so the
bind-by-name table routes the drain buffer to the `recordBuf` global. `cleanup`
then destroyed `_drain_buffer` **and** `record_buffer` — by then the same
object. Any Metal session that dumped path records would double-free on
teardown.

Fixed structurally rather than with a guard at the free site: ownership
transfers through `SceneResourceSet.adopt()`, so the set holds the buffer in the
`record_buffer` slot and `close()` destroys it exactly once; `cleanup` asks
`_gpu_set.owns(self._drain_buffer)` before freeing it itself.

## 5.1 — Reality had more growth sites than the task listed

The task named four. There are seven, and all now route through the set:

| Site | What grows |
|------|-----------|
| `_ensure_mesh_buffer_capacity` | vertex / index / BVH |
| `_sync_volume_grid` | the density 3D texture |
| `_update_texture_pool_descriptors` | the bindless pool (rebind only) |
| `resize()` | offscreen output, accumulation, HUD, readback |
| `_upload_flat_materials` (material capacity) | **five** buffers at once: flat materials, material types, mtlx skin, std surface, spectral material emission |
| `_upload_instances` (instance capacity) | the TLAS instance buffer |
| `_upload_std_surface_params` (Metal MSL stride) | the std-surface buffer |

Growth sites now state the new **capacity** and `regrow()` re-evaluates the
declaration's own `cap * STRIDE + slack`. Previously that arithmetic was written
twice per resource — once in `_init_gpu` and once at the grow site — and the
rebind was a separate call to remember. That duplication is what let binding 49
(`spectralEmitters`) keep pointing at a freed buffer while 18 was rewritten.

## Deliberately NOT changed

- **`light_splat_buffer` is not recreated by `resize()`.** It is sized
  `width * height * 3 * 4` and indexed per pixel, but `resize()` recreates only
  the four size-dependent images. Growing the viewport therefore leaves the BDPT
  light-tracer splatting past the end of a stale buffer. This is pre-existing
  and fixing it would change rendered output, which this pure-refactor change is
  gated against — so the declaration deliberately reproduces today's behaviour
  (it is *not* tagged `size_dependent`). Filed separately.
- **Binding numbers are not derived** from `bindings.slang` (design D2). They
  are relocated next to the allocation, not re-sourced.
- **The descriptor pool and sets** stay in `_create_descriptors`: they are
  Vulkan objects, not inventory. Only their *sizes* are now counted from the
  declarations (and reproduce the pre-change tally exactly: 22 storage buffers,
  3 storage images, 6 samplers + the bindless array, 1 UBO).

## Gate evidence

- **Hostless:** `tests/test_gpu_resources.py`, 23 tests — declarations vs the
  captured golden on all three configurations (name, kind, size inputs, kwargs,
  allocation order), the Vulkan write sequence, the Metal bind names, the
  teardown order, alloc == destroy, plus the source gate that fails if a
  `VkWriteDescriptorSet` reappears in `renderer.py` or a deleted rebind helper
  comes back.
- **Post-change re-capture on real devices:** the same instrumentation run
  against the new code reproduces the golden entry-for-entry on Metal RGB,
  Metal spectral and Vulkan RGB — allocation order, descriptor writes, Metal
  bind names, destroy order, no leak, no double destroy.
- **Bit-identity:** `suite/mat_diffuse` @128², 24 frames, maxdiff **0** vs
  `main` on Metal megakernel, Metal wavefront and Vulkan megakernel. (Metal vs
  Vulkan differ by 1 — pre-existing, identical on `main`.)
- **Hostless sweep:** 2466 passed; the 7 failures are pre-existing and identical
  to `main`.
- `ruff` clean. Note that bare `ruff check src/` inspects **0 files** here (the
  root `.gitignore` is `*`), so it was run over an explicit tracked-file list.
