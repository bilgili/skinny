## Context

The BRDF-only neural sampler is rejected (`spline_flow/FINDINGS.md`): it loses equal-time
in every roughness regime and fireflies at low roughness (a near-delta specular lobe a
finite-bin spline cannot match the way analytic VNDF does). The offline size×precision
study (`spline_flow/BRDF_SIZE_PRECISION.md`, 27 cells = 9 sizes × 3 precisions) sealed
that across capacity and the three precision modes it had — but on **Apple MPS**, where
its own notes flag the cost as indicative and the fp16 ALU win as unmeasurable offline.

An RTX 4090 (sm_89, Ada) is now available. Ada TensorCores natively run fp16, bf16, tf32,
and **fp8** (e4m3 / e5m2). PyTorch exposes all of these: autocast / dtype casts route
fp16/bf16/tf32 GEMMs through cuBLAS TensorCore paths, and `torch._scaled_mm` (sm_89+)
runs fp8 GEMM via cuBLASLt. The harness is already precision-aware — `brdf_guiding_eval.py`
has `--precision fp32|fp16-storage|fp16-compute`, an independent hemisphere-quadrature
reference, equal-time efficiency, per-roughness firefly stats, and pdf-parity — and the
precision boundary (`NF_WT` storage vs `NF_CT` accumulate, fp32 spline core) is already
defined by `neural-precision-size-study`. Only the **CUDA real-cost path and the 8/16-bit
float types** are missing.

The `spline_flow` env is `torch 2.11.0+cpu` with no numpy, so CUDA is currently
unavailable — the bootstrap is the first task.

## Goals / Non-Goals

**Goals:**
- Re-run the full 9-size grid × an extended precision axis `{fp32, tf32, fp16-compute,
  bf16-compute, fp8-e4m3, fp8-e5m2}` (~54 cells) on the real 4090.
- Honest GPU cost: `cuda.Event` ms (warmup + N-iter + synchronize) at a realistic
  inference batch, plus the fp32→precision **speedup ratio** and a TensorCore-vs-launch-
  bound caveat (the tiny net may not be TensorCore-bound — reported, not assumed).
- Extend the renderer's precision split to the new float types with the **RQ-spline math
  and pdf kept fp32 in every mode**.
- Re-test the equal-time + firefly verdict on real hardware; report per-precision
  pdf-parity drift and per-roughness firefly, especially for the 8-bit risk probe.
- Unbiasedness preserved in every precision mode (the flow's pdf normalizes by
  construction; verified per cell).

**Non-Goals:**
- skinny shader integration; custom cuBLASLt/CUTLASS kernels (approach C is torch-ops);
  integer quantisation (float types only); overturning the rejection; path-guiding net.

## Decisions

- **Approach C (hybrid), torch ops — not custom kernels.** fp16/bf16/tf32 go through
  autocast / dtype casts (real cuBLAS TensorCore); fp8 goes through `torch._scaled_mm`.
  Custom cuBLASLt/CUTLASS (approach B) was rejected: the net is a Linear of K=64, far
  below MMA tile sizes, so a hand-written kernel is large effort for a GEMM whose honest
  win comes from batching M, not tiling — which torch already exposes. *Alt rejected:* B.
- **Precision boundary = the linear-layer GEMMs only; spline stays fp32.** Identical to
  `neural-precision-size-study`: `NF_WT` = weight storage dtype, `NF_CT` = GEMM accumulate
  dtype. Edit happens at `train.py:458` (`raw = self.net(x_in)`) in `ConditionalSplineFlow2D`.
  The softmax / cumsum / softplus / inverse-quadratic solve are catastrophic-cancellation
  prone and **stay fp32 in every mode** — they gate the near-delta peak and fp16/fp8 there
  would manufacture firefly, confounding the measurement. Edit `train.py`'s copy only; the
  `render_records.py` copy is left untouched (`BRDF_SIZE_PRECISION_SCOPE.md`).
- **fp8 via `_scaled_mm` with dynamic per-tensor amax scaling.** Per call: `scale =
  amax / fp8_max` (e4m3 max 448, e5m2 max 57344) for input and weight, quantise, GEMM in
  fp8 with fp32 accumulate, rescale. e4m3 (more mantissa) and e5m2 (more range) are
  separate cells — e5m2 is the wider-range / coarser-mantissa probe. K=64 satisfies the
  `_scaled_mm` multiple-of-16 constraint; other dims are padded to ×16 (padding logged).
- **tf32 is a math mode, not a stored type.** Cell `tf32` = fp32 tensors with
  `torch.backends.cuda.matmul.allow_tf32 = True`; it rides along as the cheapest
  TensorCore probe (one flag) and the natural fp32-fast baseline between fp32 and fp16.
- **Real cost = CUDA events at realistic batch, with the speedup ratio as the honest
  signal.** Wall-clock and MPS timing are replaced by `cuda.Event` start/stop around the
  inference GEMM with warmup + N iters + `synchronize`. Because a K=64 Linear is tiny, the
  cell may be launch/bandwidth-bound rather than TensorCore-bound; the reported
  **fp32→precision speedup ratio** exposes which, so the cost column is never dressed up.
  Batch M = spp × cases (large) so the GEMM shape (large M, K=N≈64) is the one where
  TensorCores *can* help.
- **Full grid, logged cuts.** All 9 sizes from the MPS study × 6 precisions, run on the
  4090; the driver `log()`s exactly which cells ran. fp8 cells skip cleanly (logged) if
  `_scaled_mm` / sm_89 is unavailable, so the table never reads as exhaustive when it isn't.
- **Headline stays the failure axis, not the median.** As in `BRDF_SIZE_PRECISION.md`,
  the headline is low-roughness firefly p99.9 + low-roughness equal-time efficiency vs MIS;
  the median is reported but is not the verdict (the median read "0.80× better" while low
  roughness fireflied 219×).

## Risks / Trade-offs

- **Tiny GEMM may not be TensorCore-bound.** A K=64 Linear can be dominated by kernel
  launch / memory bandwidth, so fp16/fp8 may show little real speedup → mitigated by
  reporting the measured speedup ratio and labelling the cell launch-bound vs compute-bound
  rather than asserting a TensorCore win. This is itself a finding for the CUDA target.
- **fp8 deepens the firefly.** Coarse 8-bit pdf valleys on the sharp lobe should worsen the
  low-roughness tail (fp16-compute already drifted pdf parity to ~1.5 on MPS) → measured
  per roughness and reported as a risk-probe result, not hidden; mixture interpretation and
  the fp32 spline still bound bias, and unbiasedness is checked per cell.
- **`_scaled_mm` shape / layout constraints.** Requires sm_89+, fp8 dtypes, K multiple of
  16, specific row/col-major operand layout → dims padded to ×16 (logged); cells that can't
  satisfy the constraint skip cleanly with a log.
- **Env bootstrap.** A wrong torch/CUDA build silently falls back to CPU → the bootstrap
  task asserts `cuda.is_available()`, device name, sm_89, and `_scaled_mm` presence before
  any cell runs.
- **Train noise per cell.** Bounded grid, fixed seed, fixed+seeded cases (so size/precision
  differences aren't confounded by case randomness); each size trains in minutes on the 4090.

## Migration Plan

Additive and offline. `--precision fp32` on CUDA reproduces the existing fp32 behaviour
(now timed with CUDA events instead of MPS wall-clock); the new precisions and the
real-cost column are opt-in via the harness flags. No skinny code, no descriptor bindings,
no weights-file change — the BRDF net is never baked into the renderer. Rollback = run the
`fp32` cell / read the existing `BRDF_SIZE_PRECISION.md`. The results land in a new
`BRDF_CUDA_PRECISION.md`; the MPS doc is preserved and cross-referenced.

## Open Questions

- Does fp16/bf16 actually beat fp32 on this K=64 net on the 4090, or is the inference
  launch/bandwidth-bound (in which case TensorCores don't move the equal-time verdict at
  all)? — the speedup ratio answers it.
- Does fp8-e4m3 ever stay within an acceptable pdf-parity bar at low roughness, or does
  every 8-bit cell deepen the firefly? — the per-roughness drift answers it; expected to
  deepen.
- If a precision surprisingly survives bounded-firefly AND equal-time ≥ MIS at low
  roughness on real hardware, that overturns the precision leg of the rejection and feeds a
  follow-up ship/CUDA-target change — out of scope here, but the study would flag it.
