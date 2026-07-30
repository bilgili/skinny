# Tasks: gpu-backend-adapter

## 1. Baseline capture

- [x] 1.1 Enumerate the public surface of `vk_compute` and `metal_compute`
      (classes, methods, parameter names, module functions, constants) and
      record the diff as a permanent fixture. This is the drift the
      conformance test will pin.
- [x] 1.2 Enumerate every backend branch outside the adapter modules: the 40
      `is_metal` sites in `renderer.py`, 5 in `debug_viewport.py`, 4
      `compute_queue` probes in `vk_wavefront.py`, 12 `descriptor_sets is
      None` gates, and the `is_metal` parameter of `mlt_chain`. For each,
      record the *reason* it branches. Group into capabilities.
- [x] 1.3 Capture Vulkan `.spv` bytes and `build/spv_cache` keys for all
      kernels — the byte-identity baseline for any define touched by 3.3.

## 2. Stage 1 — capability record, broken probes replaced

- [x] 2.1 Add `src/skinny/gpu_backend.py`: capability record + one-sided
      member table, populated from 1.2.
- [x] 2.2 Replace `hasattr(ctx, "compute_queue")` at all 7 sites. Record that
      this is a behaviour fix (the probe was always true) and measure: run the
      Metal wavefront suite before and after and diff images.
- [x] 2.3 Replace the 12 `descriptor_sets is None` gates.
- [x] 2.4 Hostless test: no attribute-presence backend probe remains.

## 3. Stage 2 — naming and argument domains

- [x] 3.1 Unify `PreviewPipeline` / `PreviewPipelineMetal` under one name.
- [x] 3.2 Unify argument domains: one address-mode and format vocabulary; one
      parameter name per shared method (`rgba_f32` vs `data`). Delete
      `_VKFORMAT_INTS` / `_VK_ADDRESS_INTS` once no caller passes ints.
- [x] 3.3 Fold `BINDLESS_TEXTURE_CAPACITY` into the capability record; make
      the `bindings.slang` "MUST equal" comment a test. Verify 1.3 byte
      identity.
- [x] 3.4 Move the private cross-module reaches (`_make_sampler`,
      `_rgba_f32_to_rgba8`) onto the interface or into their callers.

## 4. Stage 3 — recording adapter

- [x] 4.1 Add the recording adapter: records allocations, bindings, dispatch
      group counts, readbacks; returns zero-filled data.
- [x] 4.2 Conformance test over all three adapters, modulo the one-sided
      table; fails on any drift from the 1.1 fixture that is not declared.
- [x] 4.3 Binding-coverage check: a recorded dispatch whose bindings do not
      cover the reflected shader globals is reported.

## 5. Stage 4 — renderer branch migration

- [x] 5.1 Migrate branches by region, ordered by existing test coverage;
      each region green before the next.
- [x] 5.2 For each of the 15 Metal-only renderer methods, decide and record:
      adapter implementation, or capability-gated path that stays.
- [x] 5.3 Reassess `wavefront_driver.WavefrontRecorder.flush_heavy_eye` — a
      watchdog concept in a backend-neutral protocol; move to the capability
      record if it fits.
- [x] 5.4 Source gate: remaining `is_metal` occurrences are adapter selection
      or genuine two-implementation splits only.

### 5.2 decision record — the Metal-only renderer methods

`renderer-gpu-resource-set` and the readback fold (3.4) already removed most of
them; **7** remain, and all 7 keep their capability-gated / two-implementation
form. None becomes an adapter method:

| Method | Decision | Why |
|--------|----------|-----|
| `_render_windowed_metal` | stays | frame orchestration: acquire a surface image, dispatch, blit, present. The Vulkan twin drives a swapchain with semaphores and in-flight fences (`has_frame_sync_objects`). Two implementations of *a frame*, not of *a resource*. |
| `_render_headless_metal` | stays | same, minus the surface. |
| `_render_megakernel_metal` | stays | binds by name at dispatch and commits in row bands under the watchdog (`needs_watchdog_tiling`); the Vulkan twin records `vkCmdDispatch` into the frame command buffer it owns. |
| `_render_scene_metal` | stays | composes the two above. |
| `_render_wavefront_metal` | stays | drives `metal_wavefront` through one open encoder (`MetalFrameEncoder`, declared one-sided); the Vulkan twin drives `vk_wavefront` pass objects through descriptor sets. |
| `_run_wavefront_mlt_bootstrap_metal` | stays | the shared host round-trip already lives in `mlt_chain.run_bootstrap`; what remains here is the submit shape. |
| `_render_material_preview_metal` | stays | the pipeline class is now one name on both adapters (3.1) and `pack_push` is shared; the record-and-submit vs bind-by-name dispatch is the genuine split, declared as `PreviewPipeline.__init__` in `DIVERGENT_SIGNATURES`. |

The rule that came out of it: **resource construction, binding, readback and
dispatch belong on the adapter; assembling a frame does not.** A method that
only differs in *how it submits* is a two-implementation split, and the
capability record is what tells the shared code which one it is talking to.

### 5.3 decision — `flush_heavy_eye` stays in the driver protocol

Not moved to the capability record. The capability record already owns *whether*
the boundary costs anything (`needs_watchdog_tiling`); hoisting the decision into
`wavefront_driver` would put a watchdog test in the backend-neutral loop **and**
drop the per-scene `bound_heavy_eye` condition the Metal recorder folds in. Like
`barrier()`, it is a point in the recording that each backend interprets. The
protocol docstring now says so.

## 6. Gates

- [x] 6.1 `ruff check src/`; full hostless `pytest`.
- [x] 6.2 `.spv` bytes and `spv_cache` keys unchanged vs 1.3.
- [ ] 6.3 Metal: megakernel + wavefront + preview dock + debug viewport smoke.
- [x] 6.4 Vulkan: megakernel + wavefront smoke.
- [ ] 6.5 Parity matrix dual gate unchanged; both structural and shaded
      Metal↔Vulkan parity tests pass on a dual-device host.
- [x] 6.6 `tests/test_metal_cleanup.py` incl. the gpu-marked kill harness.
- [x] 6.7 Docs: `docs/Architecture.md` backend seam section; CLAUDE.md
      compatibility matrix if any capability changes user-visible behaviour.
- [x] 6.8 `openspec validate gpu-backend-adapter --strict`.

### Gate results

**Run and green:**

- **6.1** `ruff check src/` clean. Hostless `pytest -m "not gpu"`: **17 failed,
  2842 passed** — against a merge-base baseline of **17 failed, 2819 passed**,
  and the failure *sets are identical* (worktree asset holes + known
  pre-existing). No new failures; 23 more tests pass.
- **6.2** The tracked Slang tree digest is unchanged (`0cb4c21e…` over 93 files,
  byte-for-byte vs the merge-base), and the 82 `test_shader_variants.py` /
  `test_spv_cache_hit.py` goldens pass, so no flag tuple or cache key moved.
  (A whole-tree `find` digest differs only because a test run rewrites the
  untracked generated `generated_materials.slang`.)
- **2.2 measurement + 6.3 (megakernel, wavefront) + 6.4** A/B render of 8 cells
  — `mat_diffuse`/`mat_conductor`/`int_indirect_box`/`int_caustic` path
  wavefront, `mat_dielectric` bdpt wavefront, `mat_diffuse` path megakernel,
  `mat_emissive` bdpt megakernel, `int_bleed` path megakernel — rendered on the
  merge-base tree and this tree, **on both backends**. All 16 comparisons are
  **bit-identical**: maxdiff 0.000e+00, relMSE 0, FLIP 0. This is the
  measurement design D3 demands for replacing the always-true `compute_queue`
  probe: the pass factories now refuse on a capability read, and the images did
  not move. The Vulkan run is confirmed to have taken the Vulkan body (its
  config report reads `backend auto → vulkan ON`), which matters because a
  Metal-default sweep never exercises `vk_compute` / `vk_wavefront`.
- **6.6 (hostless half)** `tests/test_metal_cleanup.py -m "not gpu"`: 13 passed.
- **6.7** `docs/Backends.md` gains "The declared seam"; `docs/HostModules.md`
  gains "The backend seam"; `docs/README.md` index updated; CLAUDE.md gains the
  ownership paragraph. `tests/test_docs_links.py` 67/67. No CLAUDE.md
  compatibility-matrix change: no capability alters user-visible behaviour (no
  CLI flag, no envelope combo, no refusal changes).

**Also run and green (second pass, quiet-machine retry):**

- **6.6** `tests/test_metal_cleanup.py -m gpu`: **3 passed** (clean-exit probe,
  SIGKILL-mid-render → GPU-usable probe, atexit teardown), on top of the 13
  hostless. The kill harness is required because this change touches context
  lifecycle and dispatch gating.
- **6.5 (matrix half)** `tests/pbrt/test_parity.py -k matrix` on Metal:
  **12 passed, 0 failed**, 96 renders in 7m49s. The five scenes whose assets
  live only in the primary checkout (`bathroom`, `dragon`, `disney_cloud`,
  `bunny_cloud`, `clouds`) were deselected — they fail at the merge-base too,
  for want of the asset, not the code. The gate asserts each combo against its
  recorded pbrt-truth baseline and the `(Path, wavefront)` self-consistency
  anchor, so passing IS "dual gate unchanged".

**STILL NOT run — host memory, not a result:**

- **6.3 (preview dock, debug viewport)** and **6.5 (structural + shaded
  Metal↔Vulkan parity)**. All four cold-compile Slang for the Metal target and
  are skipped unless `RUN_METAL_PREVIEW_COMPILE` / `RUN_METAL_DEBUG_RASTER` /
  `RUN_METAL_MEGAKERNEL_COMPILE` is set **and** the command runs under
  `scripts/guarded_metal.sh`. The guard aborted at `avail=7.87GB (need>=10)`.

It was **not** bypassed and `MIN_FREE_GB` was **not** lowered: this box runs at
zero swap headroom, and an `MTLCompilerService` spike with nowhere to page locks
or reboots it — which is the exact failure the threshold was chosen to prevent.
Freeing ~3 GB (the guard's metric counts only free + speculative + purgeable
pages, so ordinary app RSS is what stands in the way) and re-running the three
commands is all that remains.
