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

- **`light_splat_buffer` is not recreated by `resize()`** — *within this
  change*. It is sized `width * height * 3 * 4` and indexed per pixel, but
  `resize()` recreated only the four size-dependent images, so growing the
  viewport left the BDPT light-tracer splatting past the end of a stale buffer.
  Pre-existing, and fixing it changes rendered output on the resize path, which
  this pure-refactor change is gated against — so the declaration here
  reproduces today's behaviour and is *not* tagged `size_dependent`.

  Fixed immediately afterwards as a **separate commit on this branch**
  (`fix(renderer): resize the BDPT light-splat buffer with the viewport`), which
  is exactly the payoff this change was for: once the inventory is
  declaration-owned, the fix is one flag — `size_dependent=True` — and
  `SceneResourceSet.resize()` recreates and rebinds it with the rest. Before
  this change the same fix would have meant edits in three places. That commit
  carries its own on-device verification (mid-session 128²→256²→192² resize
  under `--integrator bdpt`, Metal and Vulkan × megakernel and wavefront); the
  parity-matrix gate below was measured at this change's own commit, before it.
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
- **Parity matrix:** 20 passed, 1 skipped, 1 xfailed, 0 failed (112 renders).
  No manifest, `baseline` or tolerance edited; `src/skinny/shaders/` is
  byte-identical to `main`. The single skip is pre-existing — Metal spectral
  wavefront BDPT overflows Metal's 31-buffer argument table (`wfBdptWalk`
  declares 41 globals), which is decided by shader source and build defines,
  none of which this change touches.

  Run on a **detached worktree pinned to this change's commit**, tree verified
  clean before and after. An earlier sweep with the same counts was discarded as
  a gate: an unrelated background task edited the worktree at 09:44, inside that
  run's 09:29–09:50 window. It almost certainly still measured the right code
  (pytest imports both modules at collection, ~15 min before the edit, and the
  edit only affects `resize()`, which the harness never calls) — but "almost
  certainly" is not what "identical, not close" asks for.
- **Hostless sweep:** 2466 passed; the 7 failures are pre-existing and identical
  to `main`.
- `ruff` clean. Note that bare `ruff check src/` inspects **0 files** here (the
  root `.gitignore` is `*`), so it was run over an explicit tracked-file list.

## Pre-merge review (codex + an independent architect reviewer)

Both reviewers were given the Principal Software Architect framing and told to
judge ownership, seams and data flow before line-level correctness. They agreed
on the architecture — ownership, the one backend branch, the `write_binding`
boundary, and recording the three orders rather than deriving them — and
between them found five things worth acting on. All are fixed; none required
changing the design.

Note: `codex review` refuses a custom prompt alongside `--base`/`--uncommitted`,
so the architect framing was delivered by running it in a scratch worktree
where the change is the uncommitted state.

| # | Sev | Finding | Fix |
|---|-----|---------|-----|
| 1 | P2 / LOW | `adopt()` closed the *teardown* double free but the Metal record-drain **grow** path reintroduced it: the caller freed `_drain_buffer`, which by then WAS the `record_buffer` slot occupant, and `adopt` then freed it again. | Caller asks `owns()` first; `adopt` is the single destroyer. Gated by `test_metal_record_drain_grow_does_not_double_free`, which drives the real `_ensure_wf_record_drain` and fails with "StorageBuffer destroyed twice" without the guard. |
| 2 | MEDIUM | Capacities lived on **both** the renderer and `_gpu_set.sizes`; three growth sites wrote two adjacent lines. The same two-copy invariant this change removes for buffers, one level up. | Capacities are allocation inputs and now live on `sizes` only, via forwarding properties. Growth sites bump once. |
| 3 | MEDIUM | Bindings 52–57 stated in `gpu_resources.MLT_BINDINGS` **and** `WavefrontMltPass._BINDINGS`; the source gate only greps `renderer.py` and cannot see them drift. | Cross-check test pins the two equal. (Restructuring the MLT pass is out of scope.) |
| 4 | P3 / MEDIUM | `metal_bindless()` had **zero callers** while four dispatch sites hand-built the same comprehension from the private `texture_pool._slots`. An unused method on the owner is worse than none — it reads as covered. | Both megakernel and preview dispatch sites now take it from the set. |
| 5 | LOW | Two `hasattr(self, "light_splat_buffer")` guards went vacuous once resource attributes became properties: always `True`, so the intended skip-if-absent became `None.fill_zero_sync()`. | Both test for `None`. |

**Accepted, not fixed** (recorded so the next reader does not re-litigate):

- `SceneResourceSet.stub()` returns `None` for unstubbed resources rather than
  raising, so a test that under-stubs fails deep in renderer code. Acceptable:
  the alternative is a stricter fake that every future test must fully populate.
- `replace()` nulls the whole group before allocating, so an allocation failure
  mid-group leaves `None` slots. Deliberate — it avoids transiently holding two
  copies of a large mesh — and not a regression (pre-change the same failure
  left dangling destroyed objects).

The reviewers independently confirmed, line by line against `git show main:`,
that `VULKAN_WRITE_SEQUENCE` and `DESTROY_SEQUENCE` reproduce the pre-change
sequences entry for entry, that `pool_sizes()` reproduces the hand-maintained
tally it replaced, that `metal_binds()` produces the identical name→resource
dict, that the allocation-before-seeding reordering reads nothing early, and
that all 67 pre-change resource assignment sites map to a set call.
