## 1. Phase 0 — Execution-mode axis (no behavior change)

- [x] 1.1 Add `execution_mode_index` + `execution_modes` to the renderer (megakernel default), `set_execution_mode()` / `cycle_execution_mode()` (delegating to pure `clamp_mode_index` / `next_mode_index` in `params.py`), and include `execution_mode_index` in `_current_state_hash()` so a switch resets accumulation.
- [x] 1.2 Persist via the shared param snapshot — added `_disc("Execution", "execution_mode_index", "execution_modes")` to `STATIC_PARAMS`, so both GLFW (`app.py`) and Qt (`ui/qt/app.py`) `_snapshot_params`/`_apply_saved_params` round-trip it automatically; restore clamps against the live `execution_modes` list.
- [x] 1.3 The same `_disc` entry is the `ALL_PARAMS` wiring; the data-driven `build_app_ui` renders it as a Combo in the Render section for GLFW + Qt + web. Enforced by `test_ui_spec::test_every_param_bound_exactly_once`. (Debug viewport is visualization-only — no param panel — so N/A, consistent with the gizmo-mode precedent.)
- [x] 1.4 Metal pin: when the context has no `compute_queue` (non-Vulkan) the renderer exposes only `["Megakernel"]`, so set/cycle and settings-restore collapse any wavefront request to megakernel. Documented inline like the Vulkan-only skinning/refit passes.
- [x] 1.5 Capability gate implemented as `effective_execution_mode_index` (wavefront+bdpt → megakernel while `WAVEFRONT_BDPT_SUPPORTED=False`) + `execution_mode_fallback_active` property as the UI surfacing hook. NOTE: the *visible* fallback note is deferred to Phase 1 — in Phase 0 no wavefront dispatch exists yet, so there is no functional difference to surface; the property is the contract P1's UI will read.

## 2. Phase 0 — Two-emitter material codegen

- [x] 2.1 Extracted the aggregator string-building from `_emit_generated_materials` into the pure module-level `emit_megakernel_aggregator(fragments, binding_base)` (verbatim move → byte-identical); the method now writes per-graph files + calls it. Megakernel is one of two emitters.
- [~] 2.2 Added `emit_wavefront_material_modules(fragments)` — the wavefront half of the split — emitting one self-contained `evalGraphSurface_<target>(P,N,T,UV, params, inout sp)` per graph from the SAME `GraphFragment` (proves the two-emitter architecture; GPU-free tested). DEFERRED to Phase 1: the `[shader("compute")]` `shadeMaterial_X` entry wrapper and the `evalBSDF_X` connection entry, because their signatures are determined by P1's wavefront path-state struct + descriptor layout, which don't exist yet. Building them now would guess the P1 interface.
- [x] 2.3 Megakernel SPIR-V unchanged: the aggregator text is byte-identical (verbatim extraction, `GRAPH_BINDING_BASE`=25 unchanged, `main_pass.slang` untouched) and `test_materialx_graph` (which compiles the aggregator on-GPU, incl. `test_aggregator_emits_all_graphs`) stays green.

## 3. Phase 0 — Vulkan multi-pipeline + indirect dispatch infra

> DESIGN-UNBLOCKED (do at the start of the Phase 1 GPU session). The interfaces
> these need — path-state struct, queue/counting-sort buffers, indirect-args
> shape, per-stage descriptor layout — are now pinned in **design.md §P1-A..§P1-F**
> (3.1 ↔ §P1-F, 3.2 ↔ §P1-C, 3.3 ↔ §P1-C direct-dispatch fallback). Model the
> N-pipeline manager on `vk_skinning.py` (design.md §7). Not yet implemented:
> needs a live GPU iteration loop, so it lands with §4–§6, not as dead infra.

- [x] 3.1 The Vulkan compute layer owns N pipelines / set layouts / sets dispatched from one command buffer in several passes following the `vk_skinning.py` pattern: `WavefrontPathPass` (generate/bounce/resolve), `WavefrontBdptPass` (walk/connect/resolve), `ShadePassGroup` (per-material shade), `IndirectPaintPass` (per-material-queue dispatch) — each records its pipelines + sets into the frame command buffer.
- [x] 3.2 `vkCmdDispatchIndirect` support: `StorageBuffer(..., indirect=True)` adds INDIRECT_BUFFER usage (the build-args-shaped args buffer); `IndirectPaintPass.record_indirect` dispatches per material slot via `vkCmdDispatchIndirect(indirectArgs, slot*12)`. `StorageBuffer.download_sync` added for compute-output readback.
- [x] 3.3 Conservative direct-dispatch fallback: `IndirectPaintPass.record_direct` dispatches worst-case `ceil(streamSize/GROUP)` groups per slot and the kernel early-outs on `tid.x >= sliceCount`, producing output identical to the exactly-sized indirect dispatch.

## 4. Phase 1 — Wavefront path: state + queues

> Contracts pinned in design.md §P1-A (path-state, AoS-first), §P1-B (queues +
> counting sort), §P1-C (indirect args). Implement on GPU with live iteration.

- [x] 4.1 Added `WavefrontPathState` (`shaders/wavefront/wavefront_state.slang`, 68 B scalar layout per §P1-A) + the GPU-free Python mirror `wavefront_layout.py`. `test_wavefront_state.py` derives the stride from the Slang struct fields and cross-checks the Python layout (field order + size); struct compiles clean under slangpy. Buffer is sized `stream_size * PATH_STATE_STRIDE` by the allocator (4.2).
- [~] 4.2 Sizing math (GPU-free): `wavefront_layout.queue_buffer_sizes(...)` is the single source of truth for the stage buffer byte sizes (`test_wavefront_state.py`). Counting-sort scatter DONE + GPU-verified: `shaders/wavefront/scatter.slang` (`scatterByMaterial`, atomic write cursors) groups lanes into per-material slices (`test_wavefront_scatter.py`, real dispatch+readback). Vulkan allocation DONE: `vk_wavefront.py WavefrontPasses` allocates the stage StorageBuffers via `queue_buffer_sizes`, verified on a real headless device (`test_wavefront_passes.py` — allocate/rescale/idempotent-destroy). REMAINING: pin the hit buffer (HitData stride) alongside the intersect stage.
- [x] 4.3 Added `shaders/wavefront/build_args.slang`: `buildArgs` compute kernel (exclusive prefix-sum counts → `materialOffset`, + one indirect (x,y,z) per material via the shared `wfIndirectGroupCount` ceil-div). Ceil-div GPU-verified across edge cases (`test_wavefront_buildargs.py`, shared definition — no formula drift); entry compiles. Full-buffer dispatch verification lands when `WavefrontPasses` binds real buffers.

> ENV-ONLY MILESTONE DONE + A/B-VERIFIED. `shaders/wavefront/wavefront_env.slang`
> (`wavefrontEnv`) generates the camera ray and writes env radiance into the
> accumulation image, reproducing main_pass.slang's env path. `WavefrontEnvPass`
> (`vk_wavefront.py`) builds its pipeline + descriptor set (binds fc/accumBuffer/
> envMap from the renderer's resources, bindings 0/2/4 per slangc reflection).
> `render_headless()` gates the dispatch on `effective_execution_mode_index`;
> `read_accumulation()` reads the linear-HDR accum image for A/B.
> `test_wavefront_render.py` renders the demo scene in both modes and confirms
> a substantial fraction of pixels (the background) match the megakernel's env
> in linear HDR. Megakernel regression (TestMaterialXGraphDemoRender) still
> passes; no resource leaks; env pass destroyed in cleanup(). This proves the
> integration: the execution-mode flag routes to a distinct GPU dispatch that
> reads the shared bindings and writes the accumulation image correctly.
> REMAINING: gate the windowed `render()` path too (headless done) + a tonemap/
> output write so windowed wavefront displays (accum-only today). Then the full
> staged generate→intersect→logic→shade pipeline supersedes this env kernel.

> INTERSECT VERIFIED (primary-ray, fused). `shaders/wavefront/wavefront_visibility.slang`
> (`wavefrontVisibility`) generates the camera ray and traverses the shared BVH
> via `traceScene`, writing a hit mask. `vk_wavefront.BoundComputePass` (general
> spec-driven pass, reusable for every stage kernel) + `renderer.build_wavefront_
> visibility_pass()` bind the geometry/BVH/instance/material buffers at the
> reflected bindings (0/2/5/6/7/12/13/16). A/B-verified against the demo scene's
> known geometry (`test_wavefront_render.py`): hits are centred on the spheres,
> corners miss, hit fraction sane. Proves wavefront BVH traversal matches the
> megakernel's hit/miss boundary. Hit *data* verified too:
> `wavefront_normal.slang` writes per-hit shading normals; the decoded normals at
> sphere hits are unit-length — the shade stage gets a valid surface frame. Both
> built via the generic `build_wavefront_trace_pass(module, entry)`. REMAINING
> for the staged pipeline: write hits into the path-state + per-material queues
> (vs. the fused mask) and the logic/shade stages (material eval via
> evalGraphSurface + lighting + MIS).
>
> SHADE — lighting step DONE: `wavefront_diffuse.slang` (`wavefrontDiffuse`)
> lights the hit with a fixed albedo under the scene's directional light
> (binding 20) + a normal-sampled env ambient; env on miss. Verified: the
> geometry comes out shaded (normal-dependent gradient) and distinct from the
> background (`test_wavefront_render.py`). Proves the wavefront lights a surface
> with the hit normal. REMAINING for a megakernel-A/B shade: real material
> albedo (evalGraphSurface + graph-param SSBOs / bindless textures), MIS, and
> multi-bounce — threaded through the path-state + queue buffers.
>
> SHADE — material eval scaffolded + blocker identified. `wavefront_material.slang`
> (`wavefrontMaterial`) evaluates the hit material's base colour via the shared
> `evalSceneGraphBaseColor` (per-scene generated_materials) and compiles clean.
> Its reflected bindings are 0/2/4/5/6/7/12/13/16 + graph-param SSBOs (25+, e.g.
> graphParams at 26) + **binding 14, the bindless texture array (128
> combined-image-samplers, PARTIALLY_BOUND descriptor indexing)**. That last is
> the blocker: `BoundComputePass` only does descriptorCount=1 — it needs
> descriptor-indexing support (per-binding descriptorCount + a
> VkDescriptorSetLayoutBindingFlagsCreateInfo with PARTIALLY_BOUND for 14, pool
> sized 128, and per-filled-slot writes mirroring `_update_texture_pool_descriptors`).
> DONE: `BoundComputePass` now supports bindless descriptor arrays (per-binding
> descriptorCount + UPDATE_AFTER_BIND|PARTIALLY_BOUND flags + per-filled-slot
> writes, mirroring the megakernel). `renderer.build_wavefront_material_pass()`
> binds the traceScene set + env + every per-graph param SSBO + the 128-slot
> texture pool. `test_wavefront_render.py` confirms the geometry renders in
> material-driven colours that vary across the surface set (per-channel std >
> 0.03) — the wavefront drives per-material evaluation. This is the original
> per-material-partition motivation, running in the wavefront. REMAINING for a
> megakernel-A/B shade: full BSDF response (not just base colour) + lighting +
> MIS + multi-bounce, threaded through the path-state + queue buffers.

## 5. Phase 1 — Wavefront path: stage kernels

> STAGED WAVEFRONT PATH TRACER DONE + A/B-VERIFIED. `shaders/wavefront/
> wavefront_path.slang` (`wfPathGenerate` / `wfPathBounce` / `wfPathResolve`)
> tears the megakernel's in-register path loop into compute dispatches over a
> per-lane `WavefrontPathState` buffer in VRAM. The per-bounce body is a verbatim
> relocation of `integrators/path.slang`'s `estimateRadiance` (+ runFrame's
> primary-miss): it calls the SAME `evaluateBounce` (NEE over all lights + env,
> MIS, BSDF sample, flat/graph/skin/python material dispatch) and the SAME env /
> sphere-light MIS, so it is an unbiased estimator of the same integral.
> `vk_wavefront.WavefrontPathPass` owns the 3 pipelines + path-state buffer and a
> bounce loop (generate → bounce ×MAX with a path-state barrier between →
> resolve); set 0 reuses the megakernel scene descriptor set (no binding
> re-enumeration), set 1 binding 0 is the path-state buffer. A/B
> (`test_wavefront_path_ab`): wavefront-vs-megakernel pixel match 0.960 vs the
> megakernel-vs-itself MC noise floor 0.962, mean |Δ| 0.002 — i.e. the wavefront
> matches the megakernel as well as two megakernel renders match each other.

- [x] 5.1 `wfPathGenerate` seeds the camera ray + path state (β, L=0, rng, depth, alive, bsdfPdf=-1) for every pixel lane.
- [~] 5.2 Intersection: `wfPathBounce` traverses the shared BVH (reused `traceScene` include) per alive lane, including the cutout-transparent skip + the BSDF-sampled sphere-light MIS trace. REMAINING (coherence/perf): a *separate* intersect.slang that appends hits to per-material queues (counting-sort `build_args`/`scatter` primitives are GPU-verified; wiring them into the loop is the per-material-dispatch path).
- [~] 5.3 Integration logic: `wfPathBounce` does NEE (all lights + env) + MIS + Russian roulette + next-bounce ray generation; terminated lanes write radiance + clear ALIVE (flushed by `wfPathResolve`). Reuses the §1–§6 estimator + light/sampler includes verbatim. REMAINING: split into a standalone logic.slang with live-lane compaction (6.2).
- [x] 5.4 Per-material shade *dispatch* over each material queue. Mechanism GPU-verified — counting-sort (`build_args`+`scatter`) → per-material `vkCmdDispatchIndirect` (`IndirectPaintPass` == direct fallback, test 9.3) + the per-material shade *pipelines* + compile cache (6.4). PATH-LOOP INTEGRATION (incremental): **step A DONE** (`86d4e1e`) — `wfPathBounce` split into `wfPathIntersect` (material-free trace, ~250 KB) + shade via a per-lane hit buffer. **step A2 DONE** (`9d697ef`) — shade split per material type: `nee.slang` (NEE extracted from `path.slang`, material-free; megakernel unchanged) + `wf_shade_common.slang` (shared bindings + `wfFinishShade`) + `flat_bounce.slang` (`evaluateFlatBounce`); `wfPathShadeFlat` (~1.4 MB, flat/graph, no skin/python) + `wfPathShade` (catch-all, ~2.8 MB) gated on material type; `WavefrontPathPass.build_catchall` compiles + dispatches the heavy catch-all only when the scene has a non-flat material — flat scenes compile only the small kernel (the MoltenVK flaky-compile fix). A/B holds (`test_wavefront_path_ab` asserts flat demo → `build_catchall is False`). **step B DONE** (`119b090`) — full counting-sort path loop: intersect classifies each hit by material slot + atomically counts it; `wfBuildArgs` prefix-sums the counts into queue offsets + one `VkDispatchIndirectCommand` per slot; `wfScatter` places lanes into per-slot queue slices; the shade kernels are queue-indexed (`wfQueueLane(shadeSlot, tid.x)`) and dispatched via `vkCmdDispatchIndirect` over only their slot's lanes (flat always; catch-all when present). Also fixed a latent OOB (tile dispatch rounds to a 64-multiple → tail threads ran past `stream_size`; added a `streamSize` push-constant guard to every kernel). A/B holds (full-frame + tiled path + tiled bdpt). 5.4 COMPLETE; the per-graph (vs per-type) queue granularity + a profile-driven perf tune are possible future refinements.

## 6. Phase 1 — Wavefront path: streaming dispatch loop

- [~] 6.1 `vk_wavefront.py WavefrontPasses` created (owns the stage buffers; verified on a real device). REMAINING: own the stage compute pipelines + the bounce-loop `dispatch()` (intersect → logic → shade ×N until queues drain or max depth), modeled on `SkinningPasses.dispatch()`.
- [x] 6.2 Tiled streaming: both `WavefrontPathPass` and `WavefrontBdptPass` process the frame in fixed-size streams (`stream_size = min(num_pixels, STREAM_CAP)`), looping `streamBase` over the frame. The stage kernels are 1D slot-indexed (`slot = tid.x`, pixel = `streamBase + slot`), a 4-byte `streamBase` push constant drives generate/walk, the per-slot `pixelIndex`/`aux.pixel` decouples slot from pixel for resolve, and out-of-frame tail slots are flagged with a sentinel so bounce/connect/resolve skip them. Path-state (path) and eye/light/aux subpath (bdpt) VRAM is bounded by `stream_size`, not the pixel count. `_wf_stream_cap` overrides the cap for tests. Verified: `test_wavefront_path_tiled_streaming` (10 tiles incl. a partial tail, buffer == cap·stride, A/B holds) + the bdpt A/B runs tiled (cap 1000, 10 tiles) and still matches. Terminated-lane *compaction* within a stream (the `compactAlive` primitive) remains an occupancy optimization on top — not required for the memory bound.
- [x] 6.3 The renderer per-frame gate (`render_headless`/`render`) dispatches `WavefrontPathPass` (staged path tracer) for `wavefront` mode vs the megakernel `vkCmdDispatch` otherwise, keyed on `effective_execution_mode_index`. The path pass reuses the per-frame megakernel scene descriptor set as set 0; a `_wavefront_debug_pass` override remains for the stage-kernel verification tests.
- [~] 6.4 Per-material shade compile-win DONE + GPU-verified: `build_wavefront_shade_passes` compiles each graph's `shadeSurface_<name>` via `vk_wavefront.compile_shade_module_cached` — a per-module content-hash SPIR-V cache keyed by (this module's source + its one generated-graph dep + the shared shader tree), so adding a material misses only its own key and resident materials are cache hits (`test_wavefront_shade_cache::test_shade_pipelines_cache_per_material`: rebuild compiles nothing; evicting one module recompiles exactly that one). REMAINING: skip the *megakernel* pipeline rebuild on a wavefront-mode add — the megakernel is still maintained as the shared graph-binding/param source + the bdpt fallback, so a wavefront add currently still re-emits it. Decoupling that is part of the wavefront-supersedes-megakernel work (gated on 6.3 + the bdpt fallback removal, 8.3).

## 7. Phase 2 — Geometry suballocator (mode-independent)

- [x] 7.1 `slab_allocator.SlabAllocator` (GPU-free) owns per-mesh slabs over the shared vertex/index/BVH element spaces, keyed by (prim_path, sub-index); offsets are stable for a slab's lifetime. `renderer._upload_meshes_suballocated` replaces `_upload_meshes_concatenated`, sizing buffers via the slab high-water and re-uploading per-slab through the new `StorageBuffer.upload_range`. Grow appends only (`_ensure_mesh_buffer_capacity` reused), so existing offsets never shift. `StorageBuffer.upload_range(data, dst_offset)` writes one slab without touching neighbours.
- [x] 7.2 Free-list with best-fit reuse (no-split, coupled three-space fit) in `SlabAllocator`; `remove_node`'s resync calls `retain_only(current_keys)` → departed meshes are freed and reused by later adds, with zero re-upload of survivors (GPU-verified: `test_suballocator_gpu::test_remove_keeps_survivor_offsets_and_skips_reupload`).
- [x] 7.3 `_upload_meshes_suballocated` makes add/`_resync_geometry_from_stage` incremental: content-fingerprint key (cached on the Mesh) → resident, unchanged meshes are not re-uploaded; only new/changed slabs write; resident offsets unchanged (GPU-verified: `test_add_keeps_resident_offsets_stable`). Bake stays content-hash cached upstream.
- [x] 7.4 `renderer.compact_geometry()` — opt-in: `SlabAllocator.compact()` packs live slabs, the renderer moves their bytes via `upload_range` and `_upload_usd_scene` rewrites every TLAS BLAS offset; safe to skip (free-list tolerates fragmentation) and idempotent. Rendered output unchanged within the renderer's own re-render noise floor (GPU-verified: `test_compaction_preserves_rendered_output`).
- [x] 7.5 Both modes read the suballocated buffers: the megakernel demo render and every wavefront pass (visibility/material/staged-shade) bind the same vertex/index/BVH buffers and stay green after the suballocator swap (`test_headless` + `test_wavefront_render`).

## 7b. Renderer integration map (for the env-only milestone wiring)

Captured this session so the descriptor/render-loop wiring is turn-key (not
re-derivable without reading renderer.py). Megakernel dispatch + descriptor refs:

- `render()` records the frame: `renderer.py:6488` cmd; `:6512` `vkCmdBindPipeline(... self.pipeline.pipeline)`; `:6513` `vkCmdBindDescriptorSets(... self.pipeline.pipeline_layout, 0, 1, [self.descriptor_sets[f]], ...)`; `:6522` `vkCmdDispatch(cmd, groups_x, groups_y, 1)` (groups = ceil(dim/WORKGROUP_SIZE)); `:6626` submit with `in_flight_fences[f]`. Accum cross-frame barrier `:6497-6506`.
- Main pipeline: `self.pipeline` (ComputePipeline), built `renderer.py:2249-2255` (entry_module="main_pass", entry_point="mainImage"). Per-frame descriptor sets: `self.descriptor_sets` (`:2705`).
- Shared bindings (set 0): **0** = `fc` FrameConstants UBO (`self.uniform_buffers[f]`, written `:2820`); **1** = output image; **2** = `accumBuffer` `RWTexture2D<rgba32f>` = `self.accum_image` (StorageImage, GENERAL layout, written `:2828`); **4** = `envMap` combined sampler = `self.env_image` (SampledImage, written `:2844`); **31/32** = env CDFs.
- `accum_frame` feeds the shader via FrameConstants (packed `:6093`); accumulation is read-modify-write of binding 2 (main_pass.slang:157-169) — `wavefront_env.slang` mirrors it.
- Single integrated megakernel (no separate tonemap pass); offscreen→swapchain blit `:6571-6579`.

Wiring steps (the env-only milestone remainder):
1. WavefrontPasses builds the `wavefrontEnv` pipeline (compile `wavefront/wavefront_env.slang` → SPIR-V → `vkCreateComputePipelines`, vk_skinning style) + its own descriptor-set-layout (reflected: 0/2/4) + a descriptor set written with `self.uniform_buffers[f]`, `self.accum_image`, `self.env_image`.
2. `WavefrontPasses.dispatch(cmd, frame)` records bind+`vkCmdDispatch(ceil(w/8), ceil(h/8), 1)`.
3. In `render()`, gate: `effective_execution_mode_index == EXECUTION_WAVEFRONT` → dispatch WavefrontPasses instead of `self.pipeline` (reuse the same accum barrier + blit).
4. Build WavefrontPasses only on Vulkan; rebuild its per-frame descriptor sets when accum/env/uniform resources are (re)created.
5. A/B: headless render a geometry-free scene in both modes; compare the accumulation image (linear HDR) within tolerance.

## 7c. RESUME HERE — staged per-material shade pipelines (the compile-win)

Per-material *evaluation* already works (fused `wavefront_material`). What
remains is making each material its OWN compiled pipeline (add material =
compile one kernel, not the megakernel switch — the original motivation).
`vk_compute.emit_wavefront_shade_module(gf, graph_id, binding)` is done +
compile-verified: it emits a per-graph entry `shadeSurface_<name>` importing
only `generated.<name>_graph` (independent compilation unit), tracing + shading
only pixels whose material maps to `graph_id`, writing the base colour.

Resume steps:
1. [x] **`renderer.build_wavefront_shade_passes()`** DONE — for each `gf` in
   `self._scene_graph_fragments` it writes `emit_wavefront_shade_module(gf, gid,
   GRAPH_BINDING_BASE)` to `shaders/wavefront/shade_<name>.slang` and builds a
   `BoundComputePass` for `shadeSurface_<name>` (bindings 0/2/5/6/7/12/13/16 + 25
   `_graph_param_buffers[target]` + over-provided 14 texture pool). Returns a
   `vk_wavefront.ShadePassGroup` (record_dispatch runs all members with an
   accum-image barrier between; drops into the `_wavefront_debug_pass` seam).
2. [x] **Verify (staged == fused at hit pixels)** DONE —
   `test_wavefront_render.py::test_wavefront_staged_shade_matches_fused`: warmup
   megakernel; visibility → hit mask; fused `build_wavefront_material_pass` at
   frame F; staged `ShadePassGroup` at the SAME frame F (no `update()` between, so
   the camera jitter is identical); assert `staged[mask] ≈ fused[mask]` (≥90% of
   hit pixels within 5e-3). Each shade pass OVERWRITES its material's hits, so the
   match measures the staged pipelines, not fused leftovers. Proves the
   per-material pipelines render correctly AND are separate compilation units.
3. [x] **Compile-win check** DONE — `vk_wavefront.compile_shade_module_cached`
   gives each `shade_<name>.slang` a per-module content-hash SPIR-V cache (key =
   module source + its one generated-graph dep + the shared shader tree, via
   `shared_shader_hash`). Adding a graph misses only its own key; resident graphs
   are cache hits. `build_wavefront_shade_passes` records `shade_compiles`
   (per-material hit/miss) + `shade_keys`; `test_wavefront_shade_cache` asserts
   rebuild compiles nothing and evicting one module recompiles exactly that one
   (task 6.4 / test 9.5). NOTE: still maintains the megakernel pipeline (6.4
   remaining note).
4. [ ] **Queue-sorted dispatch**: feed intersect hits into the per-material queues
   (build_args + scatter, all GPU-verified), dispatch each shade pass over its
   queue slice via `vkCmdDispatchIndirect` — the real staged perf path.
5. [ ] **Full A/B shade**: BSDF response (not just base colour) + NEE + MIS +
   multi-bounce, threaded through the path-state + queue buffers → exact
   megakernel A/B via `test_headless`.

Carry-over facts (avoid re-investigating):
- GPU tests: `export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS;
  export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib; ./bin/python3.13 -m pytest …`
- A/B seam: set `renderer._wavefront_debug_pass = <pass>`,
  `renderer.set_execution_mode(1)`, `render_headless()`, then
  `renderer.read_accumulation()` (linear HDR). Pass destroyed in `cleanup()`.
- slangpy buffer dispatch: pass NDBuffer `.storage` for StructuredBuffer params
  (see `tests/helpers.dispatch_uint_kernel`).
- `assign_graph_ids(frags) → {target → graphId}`; `_graph_param_buffers[target]`
  (StorageBuffer); `pipeline.graph_bindings[target]` (binding number).
- `BoundComputePass` bindless: a `{"array_count", "slots"}` spec drives the
  PARTIALLY_BOUND descriptor-indexing path (binding 14 texture pool).

## 8. Phase 3 — Wavefront bdpt (limited scope)

> STAGED WAVEFRONT BDPT DONE + A/B-VERIFIED. `shaders/wavefront/
> wavefront_bdpt.slang` (`wfBdptWalk` / `wfBdptConnect` / `wfBdptResolve`) tears
> the megakernel `BDPTIntegrator.estimateRadiance` into compute dispatches that
> store the camera + light subpath vertices in VRAM between stages.
> `vk_wavefront.WavefrontBdptPass` owns the per-lane eye/light `BDPTVertex`
> buffers + aux buffer and the walk→connect→resolve loop; set 0 reuses the
> megakernel scene set (incl. lightSplatBuffer for the s=1 splat), set 1 holds
> the subpath/aux buffers. Correctness by reuse: `randomWalk`,
> `sampleLightOrigin`, `connectT1`, `connectGeneric`, `misWeight`,
> `splatLightWalk`, `bdptEnvNEE` are imported verbatim from integrators/bdpt.slang
> — the rng cursor is carried across the walk→connect split so the sequence
> matches. A/B (`test_wavefront_bdpt_matches_megakernel`): match 0.961 vs the
> megakernel-vs-itself floor 0.957, mean |Δ| 0.002.

- [x] 8.1 `wfBdptWalk` builds the eye subpath (z1 + `randomWalk`) and the light subpath (`sampleLightOrigin` + `randomWalk`) per lane and stores both `BDPTVertex` arrays (+ lengths, escaped radiance, lensWeight, rng cursor) to VRAM — bounded by stream × BDPT_MAX_VERTS × 2 (the demo's directional light yields a 1-vertex light subpath, so connections are NEE-dominated, as in the megakernel).
- [x] 8.2 `wfBdptConnect` reloads both subpaths and runs the connection strategies verbatim — t=0 emissive eye hits, t=1 NEE (`connectT1`, visibility + real-BSDF eval), t≥2 generic connections (`connectGeneric` + the `misWeight` recurrence with the rev-pdf terms). The s=1 light-tracer splat runs in `wfBdptWalk` (`splatLightWalk`). Matches the flat-first-hit bdpt scope (pinhole camera).
- [x] 8.3 `bdpt`+`wavefront` fallback removed: `WAVEFRONT_BDPT_SUPPORTED = True`, so `effective_execution_mode` keeps wavefront for bdpt and the render gate dispatches `WavefrontBdptPass` (integrator==1) vs the path pass. `test_execution_mode` still covers the gate logic for both flag values.

## 9. Tests + verification

- [x] 9.1 Headless A/B parity: `tests/test_wavefront_path_ab.py` renders the demo with the `path` integrator in both modes and asserts the wavefront matches the megakernel no worse than two megakernel renders match each other (MoltenVK is not bit-reproducible). Measured: match(wave,mega) 0.960 vs noise floor 0.962, mean |Δ| 0.002. GPU CI gate (`pytest.mark.gpu`).
- [x] 9.2 Headless A/B parity for `bdpt`: `test_wavefront_path_ab.py::test_wavefront_bdpt_matches_megakernel` — megakernel bdpt vs staged wavefront bdpt, same noise-floor-relative criterion as 9.1. Measured: match 0.961 vs floor 0.957, mean |Δ| 0.002. GPU CI gate.
- [x] 9.3 Indirect-dispatch correctness: `tests/test_wavefront_indirect_dispatch.py` — a per-material-queue paint dispatched via `vkCmdDispatchIndirect` (exact group counts from the build-args-shaped buffer, incl. the empty-material 0-group case) produces output byte-identical to the conservative direct dispatch (worst-case groups + early-out), and both honour the counting-sort queue routing.
- [x] 9.4 Suballocator unit tests: `test_slab_allocator.py` (11, GPU-free) — stable offsets across unrelated add/remove, free-list best-fit reuse, append-only growth preserves offsets, compaction packs + reports moves. `test_suballocator_gpu.py` (3, GPU) — remove keeps survivor offsets + re-uploads nothing; add keeps resident offsets stable; compaction leaves rendered output within the re-render noise floor.
- [x] 9.5 Incremental-add — both halves verified. GEOMETRY: `test_suballocator_gpu` proves an add re-uploads only the new slab (resident offsets stable, survivors not re-written) and a remove frees without re-uploading survivors. MATERIAL: `test_wavefront_shade_cache` proves rebuilding the same material set compiles nothing and a previously-unseen material (simulated by evicting its cache entry — the deterministic equivalent of a new graph, free of the megakernel-rebuild noise) compiles exactly one shade pipeline while resident materials stay cache hits.
- [x] 9.6 Front-end parity test (`tests/test_ui_spec.py`): the execution-mode toggle is bound in the single data-driven UI tree (`build_main_ui`) that every render-surface front-end (GLFW, Qt, web) renders — `test_execution_toggle_reachable_in_shared_ui` asserts it's a Combo over the live `execution_modes`; `test_execution_toggle_pins_to_megakernel_on_metal` asserts a single-mode (Metal) backend offers only Megakernel. Complements `test_execution_mode`'s pure clamp/restore Metal-pin coverage.
- [x] 9.7 Final gate. `ruff check src/` clean. Full suite under the 3.13 build venv: 460 passed; the failures/errors are **pre-existing infra/env**, not this change — `test_skin_optics` / `test_integration` harness + `test_sampling`/`test_mis` fail at the slangpy `load_from_file` include path (`cannot open 'skin_bssrdf.slang'`), `test_web`/GPU-enum fail with `VkErrorIncompatibleDriver`, all reproducing identically in isolation in ~1.5 s with the same errors, and this session's 6 commits touch none of their code (conftest/helpers/skin/web/sampling untouched). The change's OWN tests (`test_wavefront_*`, `test_suballocator_gpu`, `test_slab_allocator`, `test_execution_mode`, `test_ui_spec`, headless demo, materialx-graph aggregator) all pass in isolation. CAVEAT: `test_wavefront_bdpt_matches_megakernel` + `test_shade_pipelines_cache_per_material` flaked only inside the full back-to-back GPU suite — MoltenVK occasionally fails the Metal compile of the large `wfPathBounce`/`wfBdptWalk` kernels (they import the whole material/integrator tree) under sustained load; both pass deterministically when run alone (fix path = the deferred per-material-queue shade dispatch, which shrinks each kernel). SPIR-V: the wavefront `.slang` modules are runtime-generated `.spv` (no checked-in change); `main_pass.slang` (megakernel) is byte-unchanged — never edited this branch, `test_aggregator_emits_all_graphs` + the megakernel headless demo stay green, so megakernel output is unchanged.
