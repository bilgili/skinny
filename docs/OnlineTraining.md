# Skinny — Online Neural Training

This document covers the online-training loop: how the renderer records
radiance samples, hands them to a trainer, and takes new weights back at a
frame boundary.

For the guiding model and its equations see
[NeuralGuiding.md](NeuralGuiding.md). For the renderer overview see
[Architecture.md](Architecture.md).

---

## Online neural training

The neural directional proposal (the `{bsdf,neural}` wavefront proposal) can be
trained **continuously while the scene animates**, so the net adapts instead of
staying frozen on a per-scene offline bake (change `neural-online-training`,
Stage 2). The renderer runs a four-stage loop whose only render-thread cost is
the frame-end weight swap; training itself happens off the render path.

![Online neural training loop: the renderer emits path records to bindings 36/37, a recency-weighted ReplayBuffer feeds the NeuralTrainer, the NeuralWeightPublisher stages weights through a file or interop backend, and the frame-end swap promotes them into the binding-33/34/35 weight buffers while bumping networkVersion.](diagrams/neural/online_training_loop.svg)

1. **Drain** — `Renderer.drain_path_records_to_replay()` reads the per-vertex
   `PathRecord`s the producer appended to bindings 36 (append) / 37 (counter)
   into a recency-weighted `ReplayBuffer`
   (`sampling/neural_replay.py`), via the shared reader `records_from_buffer`
   (`sampling/path_records.py`). The default record producer is the wavefront
   path integrator itself (`wavefront-native-path-records` — no megakernel
   dispatch, on either GPU backend; change `metal-record-drain` for the Metal
   leg); the `mainImageRecord` **megakernel** stays an explicitly-selected
   source on Vulkan hardware without the 2 s-TDR watchdog limitation.
2. **Train** — `Renderer.online_train_and_publish()` runs one warm-started cycle
   of `NeuralTrainer.train_cycle` (`sampling/neural_trainer.py`):
   contribution-weighted MLE on the replay batch, reusing `spline_flow`'s
   `ConditionalSplineFlow2D` + `render_records` loss. Device branch: CPU/MPS for
   CI, CUDA + `autocast(fp16)` + `GradScaler` on the NVIDIA box (linear GEMMs in
   fp16 on tensor cores, the RQ-spline math in fp32); torch-free venvs fall back
   to a placeholder. It bakes the new weights and `publish()`es them.
3. **Publish** — a `NeuralWeightPublisher` (`sampling/neural_handoff.py`) stages
   the pending weights through one of **three handoff backends**, selected by
   `--neural-handoff` (env `SKINNY_NEURAL_HANDOFF`, persisted):
   - **`file`** (default, `neural_handoff_file.py`) — the trainer writes an NFW1
     file, the renderer hot-reloads it: a CPU round-trip through disk that works
     on **any** platform.
   - **`shared`** (`neural_handoff_shared.py`, `SharedWeightPublisher`, change
     `shared-neural-handoff`) — an in-process CPU double-buffer held in RAM. The
     trainer (a same-process daemon thread) `publish()`es a byte-faithful private
     copy via `serialize`/`deserialize_neural_weights` (the `file` path minus the
     filesystem, so the bytes match a `file` publish), and the frame-boundary
     `swap()` promotes it. No disk write, no CUDA / unified-memory device, no
     added dependency, **any** platform; the renderer uploads the swapped weights
     to the GPU through the same post-swap path as `file`. This backend never
     writes the GPU buffers directly (that is `interop`).
   - **`interop`** — the GPU handoff, resolved per backend by `make_publisher`
     (change `metal-neural-interop`). On **Vulkan**
     (`neural_handoff_interop.py`) CUDA writes weights (33) and biases (34)
     straight into the Vulkan-exported buffers via `cudaImportExternalMemory` →
     `cudaExternalMemoryGetMappedBuffer` → `cudaMemcpyAsync`, with no CPU
     round-trip, then signals the exported timeline semaphore at the staged
     version (`cudaSignalExternalSemaphoresAsync`). Needs `cuda-python`;
     implemented and verified on an RTX 4090 (`tests/test_neural_interop.py`);
     the interop `publish()` is **~54x faster** than the file backend's NFW1
     round-trip (~0.5 ms vs ~29 ms, `tests/bench_neural_online.py`). On the
     native **Metal** backend (`neural_handoff_interop_metal.py`,
     `MetalSharedWeightPublisher`) the bindings are UMA shared-storage buffers:
     `publish()` stages precision-cast bytes host-side and the frame-boundary
     `swap()` writes them in place (≤0.1 ms at the shipped fp32 size,
     `tests/test_neural_interop_metal.py`) — no file, no staging upload, no
     semaphore. Guarded — `make_publisher` raises a clear `NotImplementedError`
     naming `--neural-handoff file` on hosts with neither CUDA+external-memory
     nor Metal UMA.
4. **Swap** — `Renderer._online_frame_end_swap()` runs at the **frame boundary**
   (after the fence wait in `render_headless`, after present in `render`):
   `publisher.swap()` promotes pending→render and `networkVersion` is incremented
   in both `FrameConstants.neuralNetworkVersion` **and** the
   `WavefrontNeuralProposalPass` push-constant stamp. For the **file** backend the
   swap also `_apply_render_weights`-uploads the new weights to bindings 33/34/35;
   for **interop** there is no re-upload — `acquire_for_render()` returns no
   host-side weights because the GPU buffers already hold them. The CUDA
   publisher's swap **host-waits the timeline** to the staged version (the CUDA
   write is provably resident) and re-stamps the version; the Metal publisher's
   swap performs the staged in-place write right there on the render thread
   (the frame's device drain just completed, so nothing reads mid-write) and
   re-stamps the same way. On Metal both render paths (`render` windowed,
   `render_headless`) call the frame-end swap after their device drain.

Render weights stay **frozen during a frame**, so each sample's density is always
evaluated against the network version that drew it. An asynchronous swap
therefore raises **variance only, never bias** — mixture-MIS unbiasedness is
preserved exactly as it is for an untrained net. The wavefront-side commitment
discipline is detailed in
[Wavefront.md § Online neural training: frame-end weight swap](Wavefront.md#online-neural-training-frame-end-weight-swap).

---
