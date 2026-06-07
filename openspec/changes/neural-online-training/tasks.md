## 1. Replay buffer (Mac-runnable)

- [ ] 1.1 `src/skinny/sampling/neural_replay.py`: recency-weighted ring buffer over the shipped `PathRecord` layout (`path_records.py`, 64 B/record, AABB header); exponential recency decay
- [ ] 1.2 Live drain: renderer reads the GPU record counter (bindings 36/37) each frame into the buffer instead of streaming to `.nrec` (reuse the dump path's reader)
- [ ] 1.3 Eviction/forgetting hook (stale-on-motion policy) — stubbed default, NVIDIA box tunes
- [ ] 1.4 Unit test: recency weighting (recent records sampled more often after a simulated scene change)

## 2. Async trainer (CPU/MPS skeleton + CUDA seam)

- [ ] 2.1 `src/skinny/sampling/neural_trainer.py`: reuse spline_flow `ConditionalSplineFlow2D(9,6,24,96)` + the `render_records.py` contribution-weighted MLE loss; warm-start from current weights; small step count per cycle
- [ ] 2.2 Device branch: real loop on CPU/MPS at reduced scale (Mac CI); CUDA + tensor-core fp16 path left as a guarded seam for the NVIDIA box
- [ ] 2.3 Bake updated weights via the shipped `export_weights` NFW1 format

## 3. Weight-handoff seam (two backends)

- [ ] 3.1 `neural_handoff.py`: `NeuralWeightPublisher` interface — `publish(weights)->version`, `current_version()`, `acquire_for_render()`
- [ ] 3.2 `neural_handoff_file.py`: trainer writes `.nfw1`; renderer hot-reloads via existing `neural_weights.py` loader; double-buffer swap at frame end. Mac-testable end to end
- [ ] 3.3 `neural_handoff_interop.py`: `VK_KHR_external_memory` exported buffer + `cudaImportExternalMemory` write; CUDA + extension capability guard; raises `NotImplementedError` off-CUDA with a clear message
- [ ] 3.4 `--neural-handoff file|interop` CLI + settings persistence (default `file`); renderer instantiates the chosen publisher

## 4. networkVersion lifecycle + swap point

- [ ] 4.1 Publisher increments `FrameConstants.neuralNetworkVersion` on each successful swap (baseline 0 already in code)
- [ ] 4.2 Frame-end double-buffer swap point in `renderer.py`: render weights frozen during a frame; swap + version++ at the boundary
- [ ] 4.3 Per-sample stamp set to the render buffer's version at draw; inverse-pdf reads the stamp, not the live value (confirm the shipped `WfNeuralSample` stamp is wired to the swap)

## 5. Interop GPU plumbing (seam only on Mac)

- [ ] 5.1 `vk_compute.py`: export the neural weight buffer (33/34/35) with external-memory create-info behind a capability check; no-op/guard where unsupported
- [ ] 5.2 Timeline-semaphore sync points for CUDA-writes-Vulkan-reads — stubbed interface, NVIDIA box implements

## 6. Docs

- [ ] 6.1 `docs/Architecture.md`: external-memory notes on bindings 33/34/35; the online-loop section
- [ ] 6.2 `docs/Wavefront.md`: frame-end swap point; `README.md`: `--neural-handoff` flag
- [ ] 6.3 SVG diagram of the online training loop under `docs/diagrams/`

## 7. Verification

- [ ] 7.1 Mac headless: file-handoff swap — train a few steps on drained records, publish, `networkVersion` increments, `{bsdf,neural}` stays valid + unbiased across a swap (extend `tests/test_neural_headless.py`)
- [ ] 7.2 `py_compile` + `ruff check src/` green; interop module imports but raises off-CUDA
- [ ] 7.3 NVIDIA box (off-Mac, later): CUDA trainer concurrent with Vulkan render; interop swap; file-vs-interop benchmark; frames-to-recover on a moving object
- [ ] 7.4 `openspec validate neural-online-training --strict` passes
