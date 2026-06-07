## Why

The BRDF-only neural sampler was **rejected** (`spline_flow/FINDINGS.md`) and the
rejection was **sealed offline** by the size×precision study
(`spline_flow/BRDF_SIZE_PRECISION.md`, 27 cells). But that study ran on Apple MPS, and
its cost column is **PyTorch-MPS ms — explicitly labelled indicative, NOT renderer
ms/frame**. It left one question it could not answer, and said so in its own notes:

> *"fp16 buys no offline speed … the Apple-`half` ALU win is an in-renderer effect this
> offline study cannot measure."*

So the precision axis of the rejection rests on a cost number the hardware could not
produce. An RTX 4090 (sm_89, Ada) is now available with **real TensorCores** and native
low-precision **float** formats — fp16, bf16, and 8-bit fp8 (e4m3 / e5m2) — plus the
tf32 math mode. This lets us re-run the grid where the precision axis is *real*: measure
honest GPU inference time per precision, real fp32→low-precision speedup, and the pdf /
firefly drift that 8-bit formats actually cause on the near-delta specular lobe.

This is a **measurement, offline, in `spline_flow` only** — zero skinny shader code. It
is captured here as a skinny change for provenance: it is the same neural-sampler line
of work as `neural-precision-size-study` (which named "CUDA performance" an explicit
non-goal and "the eventual CUDA real-time target" as future work), and its result feeds
skinny's CUDA real-time target. The expected outcome is the rejection **re-sealed with
real-hardware cost**, not overturned; if a precision surprisingly survives equal-time at
low roughness, that flips to a follow-up (out of scope here).

## What Changes

- **Environment bootstrap (prerequisite).** The `spline_flow` env currently has
  `torch 2.11.0+cpu` and **no numpy** — CUDA is unavailable. Install numpy and a CUDA
  (cu12x) PyTorch build for the 4090; verify `torch.cuda.is_available()`, `sm_89`, and
  that `torch._scaled_mm` (fp8 GEMM) is present. No code runs until this passes.
- **Extended precision axis (NVIDIA float types).** `brdf_guiding_eval.py`'s
  `--precision` enum gains `bf16-compute`, `tf32`, `fp8-e4m3`, `fp8-e5m2` alongside the
  existing `fp32`, `fp16-storage`, `fp16-compute`. Semantics mirror the renderer's
  `NF_WT` (weight storage) / `NF_CT` (GEMM accumulate) split from
  `neural-precision-size-study`:
  - **tf32** — fp32 tensors, TensorCore tf32 math (`allow_tf32=True`).
  - **fp16 / bf16-compute** — the linear-layer GEMM runs in half/bf16, fp32 accumulate.
  - **fp8-e4m3 / fp8-e5m2** — GEMM inputs and weights quantised to fp8 via
    `torch._scaled_mm` with dynamic per-tensor amax scaling, fp32 accumulate.
  - **The RQ-spline math (softmax / cumsum / softplus / inverse-quadratic solve) and the
    returned solid-angle pdf stay fp32 in every mode** — identical boundary to the
    renderer. The insertion point is `train.py:458` (`raw = self.net(x_in)`); edit
    `train.py`'s `ConditionalSplineFlow2D` copy only (the `render_records.py` copy is
    left alone, per `BRDF_SIZE_PRECISION_SCOPE.md`).
- **Real GPU cost.** Replace the MPS wall-clock timing with `torch.cuda.Event`
  measurement (warmup + N iterations + `synchronize`) at a realistic inference batch
  (M = spp × cases, large; K = 64). Report **real ms** *and* the **fp32→precision speedup
  ratio**, with the GEMM padded to a multiple of 16 for TensorCore eligibility — and a
  reported caveat on whether this tiny net is TensorCore-bound or launch/bandwidth-bound
  (the speedup ratio exposes it; not assumed).
- **Full-grid CUDA driver.** `brdf_bake_grid.py` sweeps all 9 sizes × the 6 precisions
  (~54 cells) on the 4090; `log()`s exactly which cells ran (no silent caps); emits a
  manifest + CSV with the real-cost column.
- **Results doc.** New `spline_flow/BRDF_CUDA_PRECISION.md`: the real-4090 cost grid,
  per-precision pdf-parity + per-roughness firefly (low/mid/high), the equal-time verdict
  re-tested on real hardware, and a 1:1 cross-reference to the MPS `BRDF_SIZE_PRECISION.md`.

- **Not in scope:** integrating the BRDF sampler into skinny shaders; custom
  cuBLASLt/CUTLASS kernels (approach C uses torch ops); int / quantised-integer weights
  (float types only); *overturning* the rejection (the study reports whatever it finds);
  porting the path-guiding net.

## Capabilities

### New Capabilities
- `brdf-cuda-precision-study`: the rejected BRDF-only neural sampler's size×precision
  grid is re-run on real NVIDIA CUDA hardware with an extended floating-point precision
  axis (fp16, bf16, tf32, and 8-bit fp8 e4m3/e5m2 via TensorCores), measuring honest GPU
  inference cost (CUDA-event ms + fp32→precision speedup) against quality (pdf-parity
  drift, per-roughness firefly tail, held-out NLL, equal-time efficiency vs MIS),
  preserving an fp32 spline core and unbiasedness in every precision mode, and reporting
  exactly which grid cells ran.

## Impact

- **Code (`spline_flow`, not skinny):** `train.py` (`ConditionalSplineFlow2D` /
  `SplineCoupling` precision split at line 458 — add bf16 / tf32 / fp8 paths);
  `brdf_guiding_eval.py` (`--precision` enum + CUDA-event timing + speedup metric);
  `brdf_bake_grid.py` (6-precision × 9-size CUDA sweep + manifest/CSV cost column).
- **skinny:** **none** — no shader, no `neural_flow.slang`, no descriptor bindings. This
  change is captured here for provenance only; all edits are offline in `spline_flow`.
- **Hardware / deps:** requires the RTX 4090 (sm_89) and a CUDA cu12x PyTorch build +
  numpy in the `spline_flow` venv (the bootstrap task). `torch._scaled_mm` for fp8 is
  gated on sm_89+; fp8 cells **skip cleanly (logged)** on unsupporting devices.
- **Tests / gates:** every cell unbiased vs the independent hemisphere-quadrature
  reference (rel < 0.01); `norm_med ∈ [1.000, 1.001]`; per-roughness firefly p99.9 always
  reported (the median lies on this sampler); per-precision pdf-parity drift vs the fp32
  self reported, not hidden.
- **Docs:** `spline_flow/BRDF_CUDA_PRECISION.md` (new results doc, cross-referenced from
  `BRDF_SIZE_PRECISION.md` and `FINDINGS.md`).
