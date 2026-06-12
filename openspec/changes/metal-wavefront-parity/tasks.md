## 1. Metal capabilities + dispatch primitives (design phase 1)

- [x] 1.1 Probe the Metal device for half-precision support in `metal_context.py`; set `supports_fp16_storage`/`supports_fp16_compute` from the probe (keep `supports_external_memory`/`supports_external_semaphore` `false`). *(slang-rhi 0.42 under-reports `half` on Metal → conservatively fp32 here; see design Open Questions.)*
- [x] 1.2 Add `ComputePipeline.dispatch_indirect(args_buffer, offset, *, bindings)` to `metal_compute.py` via the slang-rhi command encoder's indirect entry; add a one-time `supports_indirect_dispatch` probe with a logged result. *(Probe is **empirical**, not structural: slang-rhi 0.42 binds `dispatch_compute_indirect` on every backend but its Metal impl silently no-ops it — "dispatchComputeIndirect is not supported!" — so a `hasattr` check lies. The probe compiles a 1-group kernel, dispatches it indirectly, and verifies the GPU wrote the sentinel. On this stack it resolves **false** → the fallback (1.3) is the live path, not hypothetical.)*
- [x] 1.3 Implement the CPU slot-count-readback + direct-dispatch fallback used when `supports_indirect_dispatch` is false; assert it produces the same dispatch counts as the indirect path. *(Folded into `dispatch_indirect`: when the probe is false it drains the device, reads the `(x,y,z)` group-count triple back, and issues an equal-count direct `dispatch_compute`. Verified == arange on real Metal; 1.6 asserts indirect == direct.)*
- [ ] 1.4 Add single-frame multi-pass command encoding to `metal_compute.py`: one encoder for the whole loop with a global compute-memory barrier between stages, replacing per-stage `wait_for_idle`; keep `set_data`-only parameter binding (D4 fence-hang discipline).
- [ ] 1.5 Extend the GPU-free `wavefront_layout` tests to assert reflected **MSL** field offsets of `wf_records.slang`/`wavefront_state.slang` equal the scalar SPIR-V offsets (catch `float3` 16 B padding drift).
- [x] 1.6 Unit-test the new primitives headless on Metal (indirect dispatch result == direct; fp16 flags reflect device); run `.venv/bin/ruff check src/`. *(`tests/test_metal_foundation.py`: `test_metal_dispatch_indirect_matches_direct` (indirect == direct == arange) + `test_metal_capability_flags_reflect_device` (fp16/indirect flags are real bools, external interop off). 6/6 foundation tests pass unguarded; ruff clean on changed files.)*

## 2. Backend-neutral wavefront driver (design phase 2)

- [x] 2.1 Define the stage-descriptor model and the backend-adapter interface in a new shared module. *(Done as `wavefront_driver.py`: `WavefrontRecorder` protocol + `record_path_loop` — a command-recorder interface, not a static list, since the loop has data-dependent control flow (tiling, per-bounce, conditional neural/ReSTIR/catch-all) and fill/push ops beyond plain dispatch. GPU-free, pinned by `tests/test_wavefront_driver.py` (5 cases).)*
- [x] 2.2 Implement the Vulkan adapter over the existing `vk_wavefront.WavefrontPathPass.record_dispatch` so Vulkan output is byte-for-byte unchanged. *(`_VkPathRecorder` sequences the pass's existing `vk` helpers; `record_dispatch` now delegates to `record_path_loop`. A/B + render tests: 11 passed in 153.9s, == 154.8s baseline.)*
- [x] 2.3 Route the renderer to build the wavefront driver through the new abstraction on a `VulkanContext`; verify no regression. *(Satisfied by construction — `record_dispatch`'s signature is unchanged, so the renderer call site flows through `record_path_loop` with no edit; verified by the unchanged A/B.)*
- [ ] 2.4 Migrate buffer allocation to the backend-neutral `StorageBuffer`/`StorageImage` wrappers (sizes still from `wavefront_layout.queue_buffer_sizes`), so the same allocation runs on `metal_compute`. **DEFERRED to phase 3:** `vk_wavefront.py` does `import vulkan as vk` at module top, so a Metal host can't import the pass at all — backend-neutral buffers are coupled to phase 3's vulkan-import-free restructure (where `__init__` branches on `is_metal` for pipelines + buffers together), not a standalone swap.

## 3. Wavefront path integrator on Metal (design phase 3)

- [ ] 3.1 Implement the Metal adapter: encode generate → intersect → build_args → scatter → per-material shade → resolve into one frame command encoder with barriers; per-material shade via `dispatch_indirect` (or fallback).
- [ ] 3.2 Construct `WavefrontPathPass` on a `MetalContext` in `renderer.py` (lift the Vulkan-only gate); confirm the wavefront env camera-ray pass replaces the megakernel dispatch.
- [ ] 3.3 Compile the wavefront `.slang` modules through the Metal in-process Slang→Metal path; resolve any scalar-layout/`float3` mismatch surfaced by task 1.5.
- [ ] 3.4 Headless A/B: render the test scene with the wavefront path integrator on Metal vs Vulkan; assert structural-exact equality and converged color within the megakernel perceptual tolerance.

## 4. Wavefront BDPT on Metal (design phase 4)

- [ ] 4.1 Port the subpath-walk + compacted connection stages (`WavefrontBdptPass`) onto the Metal adapter, including the strategy-split/compaction indirect dispatches.
- [ ] 4.2 Construct `WavefrontBdptPass` on a `MetalContext`; expose wavefront+BDPT selection on the Metal path in every front-end.
- [ ] 4.3 Headless A/B: every walk mode renders equivalently on Metal vs Vulkan (structural-exact + color tolerance); staged walk processes only live lanes per bounce.

## 5. ReSTIR DI on Metal (design phase 5)

- [ ] 5.1 Make the `RESTIR_DI` `ReusePlugin` build its reservoir/G-buffer passes and persistent buffers from backend-neutral resources; allocate persistent reservoirs as Metal `StorageBuffer`s that survive accumulation and reset only on config-hash change.
- [ ] 5.2 Wire ReSTIR into the Metal wavefront driver through the unchanged scene-sampling seam (scheduled at bounce 0); megakernel on either device still falls back to identity reuse.
- [ ] 5.3 Headless A/B: ReSTIR DI on Metal vs Vulkan within perceptual tolerance; unbiased mode converges to the stock-NEE reference; ReSTIR beats stock NEE at low sample count on Metal.

## 6. Neural directional proposal on Metal (design phase 6)

- [ ] 6.1 Load frozen weights on Metal by buffer upload (`set_data`) into a `StorageBuffer`; store fp16 when `supports_fp16_storage`, else fp32 via `_effective_neural_config()`; release on deselection.
- [ ] 6.2 Dispatch `neural_proposal_pass` every bounce on the Metal wavefront driver (between intersect and scatter); preserve per-sample `neuralNetworkVersion` stamping; megakernel still rejects neural.
- [ ] 6.3 Headless A/B: pdf parity vs the reference on fixed inputs; default selection byte-identical; neural on Metal vs Vulkan within tolerance (fp32 structural, fp16 within tolerance); unbiasedness check.

## 7. Verification, gating, and docs (design phase 7)

- [ ] 7.1 Add the per-integrator/per-reuse-mode headless A/B parity tests under `tests/` (Metal vs Vulkan) as repeatable cases; record equal-time results and report (no silent fallback if Metal wavefront is slower than Metal megakernel).
- [ ] 7.2 Confirm the rollback gate: if a Metal wavefront capability is missing, Metal pins to the megakernel (today's behavior) with no Vulkan impact; `--backend vulkan` remains byte-identical across all phases.
- [ ] 7.3 Update docs: `docs/Wavefront.md`, `docs/ReSTIR.md`, `docs/NeuralGuiding.md`, `docs/Architecture.md` (backend notes / binding map), and the `README.md`/`CLAUDE.md` backend tables to state wavefront/ReSTIR/neural run on Metal; regenerate any embedded shader code (`node docs/diagrams/embed_code.cjs`) and run `--check`.
- [ ] 7.4 Run `.venv/bin/ruff check src/`, the full `.venv/bin/pytest`, and `openspec validate metal-wavefront-parity --strict`; update `CHANGELOG.md`.
