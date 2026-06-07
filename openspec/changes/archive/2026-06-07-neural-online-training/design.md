# Design — neural-online-training (Stage 2)

## Context

Stage 1 (frozen) shipped: Slang inference, weight buffers on bindings 33/34/35, the
wavefront neural pass, the `.nrec` record dump (`path_records.py`), and
`FrameConstants.neuralNetworkVersion` at baseline 0. Stage 2 makes the net **live**:
trained continuously from the rays it helped trace, swapped between frames.

The CUDA-concurrent-with-Vulkan part is the whole point and **cannot be exercised on
Mac**. So this change is **stubs with clean seams**: every module imports and
compiles on Mac, the file-handoff path runs end to end on MPS, and the interop +
CUDA-trainer internals are guarded `NotImplementedError` seams the NVIDIA box fills.

## The online loop

```
 renderer (Vulkan)  ──emits──►  (x, wi, contribution, networkVersion) records
        │                              │
        │                              ▼
        │                       neural_replay.py   recency-weighted replay buffer
        │                              │
        │                              ▼
        │                       neural_trainer.py  async PyTorch trainer (CUDA box /
        │                              │            MPS skeleton); warm-start; small
        │                              │            recency-weighted updates
        │                              ▼
        │                       neural_handoff.py  NeuralWeightPublisher
        │                         ├─ file: write .nfw1 + version++  (Mac-testable)
        │                         └─ interop: cudaImportExternalMemory write (CUDA-only)
        │                              │
        └──swap at frame end◄──────────┘   θ_render ← θ_train; networkVersion++
```

Unbiasedness contract (inherited from the seam, made explicit for async swap): a
sample's density is evaluated against **its stamped `networkVersion`**, never the
current one — so a swap mid-flight cannot bias the estimator. Staleness raises
variance only.

## Module seams (all import/compile-clean on Mac)

### `neural_replay.py`
- Consumes the shipped `PathRecord` layout (`path_records.py`, 64 B/record, scene AABB
  in header) **live** — same bytes the dump writes, drained from the GPU counter buffer
  (bindings 36/37) each frame instead of streamed to file.
- Recency weighting: ring buffer + exponential recency decay so recent paths dominate
  (tracks motion). Pure-Python/torch — runs on Mac.
- Eviction/forgetting hook (stale records on motion) — stubbed policy, NVIDIA box tunes.

### `neural_trainer.py`
- Reuses spline_flow's `ConditionalSplineFlow2D(cond=9, layers=6, bins=24, hidden=96)`
  and the contribution-weighted MLE loss (the shipped `render_records.py` math) — same
  arithmetic, fed live.
- Skeleton training step runs on CPU/MPS (real loop, small scale) for Mac CI; CUDA +
  tensor-core fp16 path is a device branch the NVIDIA box fills.
- Warm-start from current weights; small step count per cycle (smooth animation → cheap).

### `neural_handoff.py` — `NeuralWeightPublisher` interface + two backends
Common interface: `publish(weights) -> version`, `current_version()`, `acquire_for_render()`.

- **`neural_handoff_file.py`** (Mac-testable):
  - trainer writes a new `.nfw1` via the shipped `export_weights` format,
  - renderer hot-reloads through the **existing `neural_weights.py`** loader,
  - double-buffer: render buffer frozen during a frame, swap + `version++` at frame end.
- **`neural_handoff_interop.py`** (CUDA-only, guarded):
  - Vulkan exports the weight buffer with `VK_KHR_external_memory` (fd/handle),
  - CUDA imports via `cudaImportExternalMemory` → writes weights with no CPU round-trip,
  - guarded by a CUDA + extension capability check; raises `NotImplementedError` off-CUDA
    with a clear message. The NVIDIA box implements the import + sync (timeline semaphore)
    and benchmarks file vs interop.

Selection: `--neural-handoff file|interop` (default `file`), persisted in settings; the
renderer instantiates the chosen publisher and wires the frame-end swap.

## networkVersion lifecycle
- Baseline 0 already in `FrameConstants`. The publisher increments on each successful
  swap. The per-sample stamp (already on `WfNeuralSample`) is set to the render buffer's
  version at draw time; the inverse-pdf evaluation reads the stamp, not the live value.

## Why two seams
The file path is correct and Mac-runnable but pays a CPU round-trip + reload — fine for
correctness and slow animation, the wrong cost for real-time. The interop path is the
"best path" the scope doc names but is hardware-bound and untestable here. Landing both
lets the NVIDIA box A/B them and lets Mac validate the loop's correctness today.

## Non-goals
- The cost decision (spline_flow `cuda-cost-gate`).
- Real CUDA kernels / production interop sync (NVIDIA box).
- ReSTIR-hybrid (Stage 3), cross-scene generalisation, megakernel neural.

## Verification
- Mac: file-handoff double-buffer swap headless — train a few steps on drained records,
  publish, `networkVersion` increments, `{bsdf,neural}` render stays valid + unbiased
  across a swap; replay-buffer recency-weighting unit test; `py_compile` + `ruff` green;
  interop path imports but raises off-CUDA.
- NVIDIA box (off-Mac): CUDA trainer concurrent with Vulkan render; interop swap;
  file-vs-interop benchmark; frames-to-recover metric on a moving object.
