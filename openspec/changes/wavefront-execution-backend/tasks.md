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

> DEFERRED to the start of Phase 1. The descriptor-set layouts, indirect-args
> buffer shapes, and per-stage pipeline signatures here are determined by the
> wavefront path-state struct + queue design (Phase 1, §4–§6), which does not
> exist yet. Building this infra now would guess those interfaces and produce
> untested Vulkan plumbing with no consumer. The `vk_skinning.py` multi-pipeline
> precedent is documented in design.md §7 and stands ready to model from.

- [ ] 3.1 Extend the Vulkan compute layer to own N pipelines / descriptor set layouts / sets and dispatch them from one command buffer, following the `vk_skinning.py` (`SkinningPasses`) pattern.
- [ ] 3.2 Add `vkCmdDispatchIndirect` support and an indirect-args buffer abstraction to `vk_compute.py` / `vk_context.py`.
- [ ] 3.3 Add a conservative direct-dispatch fallback (worst-case groups + empty-lane early-out) to A/B against indirect dispatch.

## 4. Phase 1 — Wavefront path: state + queues

- [ ] 4.1 Define the SoA path-state buffers (ray O/D, throughput, radiance, rng, depth, pixelIdx, MIS pdf, flags) sized to a configurable stream size.
- [ ] 4.2 Define the ray queue(s) and per-material append queues with atomic counters (matId → queue).
- [ ] 4.3 Add the build-indirect-args kernel that turns queue counts into shade-dispatch dimensions.

## 5. Phase 1 — Wavefront path: stage kernels

- [ ] 5.1 `wavefront/generate.slang`: seed camera rays for the current stream into the ray queue.
- [ ] 5.2 `wavefront/intersect.slang`: traverse the shared BVH for queued rays (reuse the megakernel traversal include), write hits, and append each to its material queue.
- [ ] 5.3 `wavefront/logic.slang`: light sampling + MIS + Russian roulette + next-bounce ray generation; terminated lanes accumulate radiance and free their slot (reuse the §1–§6 estimator and light/sampler includes).
- [ ] 5.4 Per-material `shadeMaterial_X` shade dispatch: BSDF sample + throughput update over each material queue.

## 6. Phase 1 — Wavefront path: streaming dispatch loop

- [ ] 6.1 Add `vk_wavefront.py` (`WavefrontPasses`) owning the stage pipelines and the bounce loop (intersect → logic → shade ×N until queues drain or max depth), modeled on `SkinningPasses.dispatch()`.
- [ ] 6.2 Implement stream refill / compaction of terminated lanes to keep occupancy high; advance through all pixels/samples for the frame.
- [ ] 6.3 Wire the renderer per-frame path to dispatch wavefront vs the megakernel `vkCmdDispatch` based on the execution mode.
- [ ] 6.4 Make adding a new material in wavefront compile only its shade/eval pipeline and register it (no megakernel rebuild); reuse existing materials' pipelines.

## 7. Phase 2 — Geometry suballocator (mode-independent)

- [ ] 7.1 Replace `_upload_meshes_concatenated` / `_ensure_mesh_buffer_capacity` with a slab allocator over the shared vertex/index/BVH buffers: per-mesh slabs, stable offsets, grow preserves layout.
- [ ] 7.2 Add a free-list with best-fit reuse; `remove_node` frees the slab and drops instance record(s) with no re-upload.
- [ ] 7.3 Make `add_model` / `_resync_geometry_from_stage` incremental: bake (content-hash cached) + write one slab + add instance record(s); resident meshes untouched.
- [ ] 7.4 Add opt-in compaction that moves slabs and rewrites every referencing instance/TLAS offset; safe to skip.
- [ ] 7.5 Confirm both execution modes read the suballocated buffers correctly.

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
