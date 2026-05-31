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

- [ ] 3.1 Extend the Vulkan compute layer to own N pipelines / descriptor set layouts / sets and dispatch them from one command buffer, following the `vk_skinning.py` (`SkinningPasses`) pattern.
- [ ] 3.2 Add `vkCmdDispatchIndirect` support and an indirect-args buffer abstraction to `vk_compute.py` / `vk_context.py`.
- [ ] 3.3 Add a conservative direct-dispatch fallback (worst-case groups + empty-lane early-out) to A/B against indirect dispatch.

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

- [ ] 5.1 `wavefront/generate.slang`: seed camera rays for the current stream into the ray queue.
- [ ] 5.2 `wavefront/intersect.slang`: traverse the shared BVH for queued rays (reuse the megakernel traversal include), write hits, and append each to its material queue.
- [ ] 5.3 `wavefront/logic.slang`: light sampling + MIS + Russian roulette + next-bounce ray generation; terminated lanes accumulate radiance and free their slot (reuse the §1–§6 estimator and light/sampler includes).
- [ ] 5.4 Per-material `shadeMaterial_X` shade dispatch: BSDF sample + throughput update over each material queue.

## 6. Phase 1 — Wavefront path: streaming dispatch loop

- [~] 6.1 `vk_wavefront.py WavefrontPasses` created (owns the stage buffers; verified on a real device). REMAINING: own the stage compute pipelines + the bounce-loop `dispatch()` (intersect → logic → shade ×N until queues drain or max depth), modeled on `SkinningPasses.dispatch()`.
- [ ] 6.2 Implement stream refill / compaction of terminated lanes to keep occupancy high; advance through all pixels/samples for the frame. (Compaction primitive DONE + GPU-verified: `shaders/wavefront/compaction.slang` `compactAlive`, `test_wavefront_compaction.py` — remaining is calling it from the bounce loop in `WavefrontPasses`.)
- [ ] 6.3 Wire the renderer per-frame path to dispatch wavefront vs the megakernel `vkCmdDispatch` based on the execution mode.
- [ ] 6.4 Make adding a new material in wavefront compile only its shade/eval pipeline and register it (no megakernel rebuild); reuse existing materials' pipelines.

## 7. Phase 2 — Geometry suballocator (mode-independent)

- [ ] 7.1 Replace `_upload_meshes_concatenated` / `_ensure_mesh_buffer_capacity` with a slab allocator over the shared vertex/index/BVH buffers: per-mesh slabs, stable offsets, grow preserves layout.
- [ ] 7.2 Add a free-list with best-fit reuse; `remove_node` frees the slab and drops instance record(s) with no re-upload.
- [ ] 7.3 Make `add_model` / `_resync_geometry_from_stage` incremental: bake (content-hash cached) + write one slab + add instance record(s); resident meshes untouched.
- [ ] 7.4 Add opt-in compaction that moves slabs and rewrites every referencing instance/TLAS offset; safe to skip.
- [ ] 7.5 Confirm both execution modes read the suballocated buffers correctly.

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

## 8. Phase 3 — Wavefront bdpt (limited scope)

- [ ] 8.1 Camera-subpath and light-subpath wavefront walks that store vertices (pos, normal, throughput, fwd/rev pdf, matId) up to max depth, bounded by stream size × maxdepth × 2.
- [ ] 8.2 Connection+shadow stage: connect prefix pairs with a visibility ray, `evalBSDF_X` at both endpoints, geometry term, and the MIS recurrence; match the current flat-first-hit bdpt scope.
- [ ] 8.3 Remove the `bdpt`+`wavefront` → megakernel fallback once parity holds; update the capability-matrix gate.

## 9. Tests + verification

- [ ] 9.1 Headless A/B parity (extend `tests/test_headless.py`): megakernel vs wavefront for the `path` integrator within tolerance; add as a CI gate.
- [ ] 9.2 Headless A/B parity: megakernel vs wavefront for `bdpt` (after Phase 3).
- [ ] 9.3 Indirect-dispatch correctness: indirect vs conservative direct-dispatch fallback produce equivalent images.
- [ ] 9.4 Suballocator unit tests: offsets stable across unrelated add/remove; free-list reuse; grow preserves layout; compaction leaves rendered output unchanged.
- [ ] 9.5 Incremental-add test: adding a model with a new material in wavefront compiles exactly one shade pipeline and re-uploads only the new slab; adding a model that reuses materials compiles none.
- [ ] 9.6 Front-end parity test: each render-surface front-end reaches the execution-mode toggle; Metal pins to megakernel.
- [ ] 9.7 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest`; recompile any changed SPIR-V with `slangc`; confirm `main_pass.slang` (megakernel) output is unchanged.
