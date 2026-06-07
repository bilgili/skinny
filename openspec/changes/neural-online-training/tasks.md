## 1. Replay buffer (Mac-runnable)

- [x] 1.1 `src/skinny/sampling/neural_replay.py`: recency-weighted ring buffer over the shipped `PathRecord` layout (`path_records.py`, 64 B/record, AABB header); exponential recency decay
- [x] 1.2 Live drain: `Renderer.drain_path_records_to_replay` reads the GPU record counter (bindings 36/37) into the buffer via the shared `records_from_buffer` reader instead of streaming to `.nrec`. **NVIDIA/Windows seam:** the record entry is a megakernel that device-losts under the 2 s TDR, so the GPU drain runs on the NVIDIA box (7.3); the reader contract is validated off-GPU (`test_neural_online`). Wavefront-native records proposed to remove the megakernel dependency (`openspec/changes/wavefront-native-path-records`)
- [x] 1.3 Eviction/forgetting hook (stale-on-motion policy) — stubbed default, NVIDIA box tunes
- [x] 1.4 Unit test: recency weighting (recent records sampled more often after a simulated scene change)

## 2. Async trainer (CPU/MPS skeleton + CUDA seam)

- [x] 2.1 `src/skinny/sampling/neural_trainer.py`: reuse spline_flow `ConditionalSplineFlow2D(9,6,24,96)` + the `render_records.py` contribution-weighted MLE loss; warm-start from current weights; small step count per cycle
- [x] 2.2 Real loop in `neural_trainer.py`: warm-starts a `ConditionalSplineFlow2D` from the current `NeuralWeights`, runs the `render_records` contribution-weighted MLE for `steps_per_cycle` Adam steps on a recency-weighted batch, bakes back through `export_weights`↔NFW1. Device branch: CPU/MPS for CI, **CUDA + `autocast(fp16)` + `GradScaler`** on the NVIDIA box (Linear GEMMs fp16 on tensor cores, RQ-spline math fp32 — the NF_WT/NF_CT boundary). torch+spline_flow probed lazily; torch-free venv falls back to the placeholder so the loop stays exercisable. **Validated on the RTX 4090** (`tests/test_neural_trainer_torch.py`: warm-start → train → finite, weights-updated, arch-preserved; warm across cycles)
- [x] 2.3 Bake updated weights via the shipped `export_weights` NFW1 format

## 3. Weight-handoff seam (two backends)

- [x] 3.1 `neural_handoff.py`: `NeuralWeightPublisher` interface — `publish(weights)->version`, `current_version()`, `acquire_for_render()`
- [x] 3.2 `neural_handoff_file.py`: trainer writes `.nfw1`; renderer hot-reloads via existing `neural_weights.py` loader; double-buffer swap at frame end. Mac-testable end to end
- [x] 3.3 `neural_handoff_interop.py`: `VK_KHR_external_memory` exported buffer + `cudaImportExternalMemory` write; CUDA + extension capability guard; raises `NotImplementedError` off-CUDA with a clear message
- [x] 3.4 `--neural-handoff file|interop` CLI (`cli_common.add_render_flags`, `SKINNY_NEURAL_HANDOFF` env, default `file`) + settings persistence (app.py snapshot/restore); `Renderer.enable_online_training` instantiates the chosen publisher via `make_publisher`

## 4. networkVersion lifecycle + swap point

- [x] 4.1 Publisher increments `FrameConstants.neuralNetworkVersion` on each successful swap (baseline 0 already in code)
- [x] 4.2 Frame-end double-buffer swap point in `renderer.py` (`_online_frame_end_swap`, called after the frame's fence wait in `render_headless` + after present in `render`): render weights frozen during a frame; `publisher.swap()` + `_apply_render_weights` (re-upload 33/34/35) + version++ at the boundary
- [x] 4.3 Per-sample stamp set to the render buffer's version at draw; `_apply_render_weights` syncs the version into BOTH `FrameConstants.neuralNetworkVersion` and the `WavefrontNeuralProposalPass` push-constant, and the swap fires only at the frame boundary so within-frame weights↔version stay consistent (proven unbiased across a swap by `test_online_file_handoff_swap_unbiased`)

## 5. Interop GPU plumbing (seam only on Mac)

- [x] 5.1 `vk_context.py` enables the win32/fd external-memory device extension when advertised (additive, try/fallback so it never destabilises the render device) + `supports_external_memory`; `StorageBuffer(external=True)` allocates 33/34/35 with `VkExternalMemoryBufferCreateInfo` + `VkExportMemoryAllocateInfo` and exposes `export_handle()` (vkGetMemoryWin32HandleKHR), guarded no-op where unsupported. Renderer allocates the neural buffers external under `--neural-handoff interop`. **Validated on the 4090** (`test_external_memory_export_capability`: external alloc + handle, default device unaffected)
- [x] 5.2 Stubbed interface (NVIDIA box implements the CUDA half): `InteropWeightPublisher` receives the exported buffer, pulls its `export_handle()` + size, and its `_import_external_memory` seam documents the exact `cudaImportExternalMemory`→`cudaExternalMemoryGetMappedBuffer` step + the timeline-semaphore sync contract (CUDA write signals an exported timeline value; the frame-end swap waits it → tear-free). Raises a clear `NotImplementedError` off the implemented path

## 6. Docs

- [x] 6.1 `docs/Architecture.md`: external-memory notes on bindings 33/34/35 (+ 36/37 online drain) in the binding map; new "Online neural training" section (drain→train→publish→swap + the two handoff backends)
- [x] 6.2 `docs/Wavefront.md`: frame-end weight-swap section (frozen-per-frame, version stamp); `README.md`: `--neural-handoff {file,interop}` flag
- [x] 6.3 SVG diagram `docs/diagrams/neural/online_training_loop.svg` (renderer→records 36/37→replay→trainer→publisher{file|interop}→frame-end swap→θ_render + version++), embedded in Architecture.md

## 7. Verification

- [x] 7.1 Headless (wavefront): `test_online_file_handoff_swap_unbiased` — train on records, publish, the frame-end swap increments `networkVersion` (FrameConstants + pass stamp synced), `{bsdf,neural}` stays unbiased vs `{bsdf}` across the swap (rel-mean 0.031 < 0.06). Replay fed synthetically (the GPU megakernel drain device-losts under the 2 s TDR on this box; reader covered off-GPU)
- [x] 7.2 `py_compile` + `ruff check src/` green; interop module imports but raises off-CUDA
- [~] 7.3 Benchmark harness `tests/bench_neural_online.py` (task-7.3). **Measured on the 4090:** the file-handoff online swap runs end to end — 32 swaps, version++ each frame, publish ≈ 25 ms (the NFW1 CPU round-trip the interop path removes), render+swap ≈ 31 ms median. CUDA trainer cycle validated separately (2.2). **Still NVIDIA-box-pending** (needs the interop CUDA-write seam 5.2 + torch in the *renderer* venv): the file-vs-interop comparison, CUDA training concurrent with the Vulkan render in one process, and frames-to-recover on a moving object — documented in the harness
- [x] 7.4 `openspec validate neural-online-training --strict` passes
