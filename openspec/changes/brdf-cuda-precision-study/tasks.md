## 1. Environment bootstrap (prerequisite — no code runs until this passes)

- [ ] 1.1 Install `numpy` and a CUDA cu12x PyTorch build into the `spline_flow` venv (replacing `torch 2.11.0+cpu`). Match the wheel to the 4090 (sm_89) and the installed CUDA runtime.
- [ ] 1.2 Assert the device is live: `torch.cuda.is_available()` is `True`, device name is the RTX 4090, compute capability is `sm_89`, and `torch._scaled_mm` is importable/callable (fp8 GEMM available). Fail loudly with a clear message if any check fails.
- [ ] 1.3 Smoke-run the existing harness on CUDA: `python brdf_guiding_eval.py --gates-only --quick --device cuda` passes (unbiased gate + flow normalization) before touching precision code.

## 2. Extended precision axis (NVIDIA float types)

- [ ] 2.1 `train.py` (`ConditionalSplineFlow2D` / `SplineCoupling`): thread a `precision` parameter to the GEMM insertion point `raw = self.net(x_in)` (train.py:458). Implement the `NF_WT` (weight storage) vs `NF_CT` (GEMM accumulate) split; keep softmax / cumsum / softplus / inverse-quadratic solve and the returned solid-angle pdf in fp32 for **every** mode. Edit `train.py`'s copy only — leave `render_records.py`'s divergent copy untouched.
- [ ] 2.2 Add `bf16-compute` (GEMM in bf16, fp32 accumulate) and `tf32` (fp32 tensors, `torch.backends.cuda.matmul.allow_tf32 = True`) modes; confirm `fp32 @ baseline` is numerically identical to the pre-change fp32 path (allow_tf32 off).
- [ ] 2.3 Add `fp8-e4m3` and `fp8-e5m2` via `torch._scaled_mm`: dynamic per-tensor amax scaling (`scale = amax / fp8_max`, e4m3 max 448 / e5m2 max 57344) for input and weight, fp32 accumulate, rescale. Pad GEMM dims to a multiple of 16 (log the padding); skip the cell cleanly with a log when `_scaled_mm` / sm_89 is unavailable.
- [ ] 2.4 `brdf_guiding_eval.py`: extend the `--precision` enum to `{fp32, tf32, fp16-storage, fp16-compute, bf16-compute, fp8-e4m3, fp8-e5m2}`; route each to the 2.1–2.3 paths.

## 3. Real GPU cost measurement

- [ ] 3.1 Replace the MPS/wall-clock inference timing with `torch.cuda.Event` start/stop around the inference GEMM: warmup iterations, N timed iterations, `torch.cuda.synchronize()`. Time at a realistic batch (M = spp × cases) so the GEMM shape is the large-M / K≈64 case where TensorCores can help.
- [ ] 3.2 Report, per cell, real ms **and** the fp32→precision speedup ratio; label each cell launch/bandwidth-bound vs compute-bound from the ratio (do not assert a TensorCore win — measure it).

## 4. Full-grid CUDA driver

- [ ] 4.1 `brdf_bake_grid.py`: sweep the 9 sizes from `BRDF_SIZE_PRECISION.md` × the 6 precisions (~54 cells) on CUDA; per cell train (steps ∝ weight count, seeded) + eval; emit `manifest.json` + a `cuda_size_precision.csv` with the real-cost column.
- [ ] 4.2 `log()` exactly which cells ran (including skipped fp8 cells and any size cuts) so the table is never read as exhaustive when it isn't.

## 5. Gates + results doc

- [ ] 5.1 Per-cell gates (reuse the FINDINGS rigor): unbiased vs the independent hemisphere-quadrature reference (rel < 0.01); `norm_med ∈ [1.000, 1.001]`; per-roughness firefly p99.9 (low/mid/high) always reported; pdf-parity drift vs the fp32 self reported per precision, not hidden.
- [ ] 5.2 Write `spline_flow/BRDF_CUDA_PRECISION.md`: the real-4090 cost grid (ms + speedup), per-precision parity, per-roughness firefly, low-roughness equal-time vs MIS, and the verdict — does real fp16/bf16/fp8 cost change the equal-time outcome. Headline = the failure axis (low-roughness firefly + equal-time), not the median.
- [ ] 5.3 Cross-reference: link `BRDF_CUDA_PRECISION.md` from `BRDF_SIZE_PRECISION.md` and `FINDINGS.md`; state plainly whether the rejection is re-sealed on real hardware or a precision cell surprisingly survived (which would flag a follow-up).
