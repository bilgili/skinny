## Why

The shipped `neural-directional-proposal` is **frozen**: weights are trained offline
on one static scene and baked. skinny is **interactive and animates** — once
geometry or lights move, the incident radiance `Li` changes and the baked `q_θ`
goes stale ([PATH_GUIDING_SCOPE.md](../../../../spline_flow/PATH_GUIDING_SCOPE.md)
§8). Staleness is **safe** (the flow self-normalises and `q>0`, so
`contribution = f·Li·cos / q` stays unbiased no matter how stale `q` is — staleness
raises variance only, never bias), but it leaves the guiding win on the table where
the scene has changed. **Online / dynamic training is the real target; the frozen
net was only the foundation.**

This change scaffolds **Stage 2 — online training**: the renderer's own path
records feed a recency-weighted replay buffer; an async trainer (CUDA on the NVIDIA
box, while Vulkan renders) updates the net continuously; a double-buffer swap
publishes new weights at frame end and bumps `networkVersion`, so each sample uses
the pdf of the version that drew it.

The hard, hardware-bound part — CUDA training concurrent with Vulkan rendering — is
**untestable on Mac**. So this change lands **import/compile-clean stubs** with two
weight-handoff seams behind a flag: a **file double-buffer** path (works on Mac/MPS
now, reuses the shipped `neural_weights.py` loader) and a **GPU-shared-memory
interop** path (`VK_KHR_external_memory` + `cudaImportExternalMemory`, CUDA-guarded,
the real-time win) the NVIDIA box fills and benchmarks against the file path. Gated
by spline_flow's `cuda-cost-gate` result: if CUDA inference is too costly, fall back
to SD-tree instead.

## What Changes

- A **recency-weighted replay buffer** (`neural_replay.py`) that consumes the
  renderer's existing per-vertex `(x, wi, contribution)` records (the shipped Stage 1b
  `.nrec` format / `path_records.py`) **live** instead of to a file, weighting recent
  paths so the net tracks motion.
- An **async training loop** (`neural_trainer.py`): a PyTorch trainer that warm-starts
  from current weights and does small recency-weighted updates on the **exact** shipped
  flow arch (`ConditionalSplineFlow2D(cond=9, layers=6, bins=24, hidden=96)`), runnable
  as a CPU/MPS skeleton on Mac and on CUDA on the NVIDIA box.
- A **`NeuralWeightPublisher` seam** with **two flag-selectable backends**:
  - **file double-buffer** — trainer writes a new `.nfw1` + bumps `networkVersion`;
    renderer hot-reloads via the **existing `neural_weights.py`** loader; swap at frame
    end. Mac-testable end to end.
  - **GPU-shared-memory interop** — `VK_KHR_external_memory` exported buffer +
    `cudaImportExternalMemory` so CUDA writes weights into the GPU buffer Vulkan reads
    with no CPU round-trip; **CUDA-guarded**, raises off-CUDA. The real-time path.
- **`networkVersion` lifecycle**: the field already exists at baseline 0
  (`FrameConstants.neuralNetworkVersion`); this change makes the publisher **increment**
  it on swap and threads the per-sample stamp so async swaps stay unbiased.
- A **handoff-backend CLI/settings flag** (`--neural-handoff file|interop`) and a
  **double-buffer swap point** at frame end in the renderer.
- **Not in scope:** the cost decision itself (spline_flow `cuda-cost-gate`); ReSTIR
  hybrid (Stage 3); cross-scene generalisation; megakernel neural (stays inline
  `{bsdf,env}`); production CUDA kernels (the NVIDIA box fills interop + trainer
  internals).

## Capabilities

### New Capabilities
- `neural-online-training`: continuous recency-weighted training of the neural
  directional proposal from the renderer's live path records, with an async trainer, a
  double-buffered weight swap that increments `networkVersion`, and two weight-handoff
  backends (file double-buffer and GPU-shared-memory Vulkan↔CUDA interop) selectable at
  runtime; staleness degrades variance only, never bias.

### Modified Capabilities
- `neural-directional-proposal`: weights become **live-updated** rather than frozen —
  the proposal SHALL accept a weight swap between frames, each emitted sample SHALL carry
  the `networkVersion` that produced it, and the density used for a sample SHALL be that
  of its stamped version so an async swap mid-flight stays unbiased.
- `scene-sampling`: the mixture-MIS unbiasedness contract SHALL hold under a proposal
  whose density changes between frames, by keying each sample's density to its
  `networkVersion` stamp.

## Impact

- **Code:** new `src/skinny/sampling/{neural_replay.py, neural_trainer.py,
  neural_handoff.py}` (+ `neural_handoff_file.py` / `neural_handoff_interop.py`
  backends); `renderer.py` (live record drain → replay buffer, frame-end swap point,
  handoff selection); `sampling/neural_weights.py` (reload entry point — already loads,
  add hot-swap); `vk_compute.py` (exported external-memory buffer for the interop path).
- **Descriptor bindings:** the interop path makes the neural weight buffers (33/34/35)
  **externally-shared** memory — update the `Architecture.md` binding map's notes; no new
  binding numbers.
- **Backends:** wavefront-only (unchanged); megakernel stays inline `{bsdf,env}`.
- **Dependencies:** trainer reuses spline_flow's flow code (PyTorch); the interop path
  needs CUDA + `VK_KHR_external_memory` — present on the NVIDIA box only, guarded on Mac.
- **Docs:** `docs/Architecture.md` (external-memory binding notes, the online loop),
  `docs/Wavefront.md` (swap point), `README.md` (new flag); SVG diagram of the online
  loop.
- **Tests:** file-handoff double-buffer swap headless on Mac (weights change →
  `networkVersion` increments → render stays valid); replay-buffer recency weighting unit
  test; interop path guarded + skipped off-CUDA.
- **Cross-machine:** stubs authored + compiled on Mac; CUDA trainer internals + interop +
  the file-vs-interop benchmark done on the NVIDIA box. Coordinate via the GitLab remote.
- **Gate:** proceed with neural only if spline_flow `cuda-cost-gate` clears; else
  SD-tree fallback.
