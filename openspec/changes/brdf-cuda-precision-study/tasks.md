## 1. Environment bootstrap (prerequisite — no code runs until this passes)

- [x] 1.1 Install `numpy` and a CUDA cu12x PyTorch build into the `spline_flow` venv (replacing `torch 2.11.0+cpu`). Match the wheel to the 4090 (sm_89) and the installed CUDA runtime. **Done:** `torch 2.11.0+cu128` + numpy/pandas/matplotlib in the root venv (`D:\Projects\spline_flow\Scripts\python.exe`).
- [x] 1.2 Assert the device is live: `torch.cuda.is_available()` is `True`, device name is the RTX 4090, compute capability is `sm_89`, and `torch._scaled_mm` is importable/callable (fp8 GEMM available). Fail loudly with a clear message if any check fails. **Done:** verified cuda True, RTX 4090, cap (8,9), `_scaled_mm` present; `fp8_supported()` guards the path.
- [x] 1.3 Smoke-run the existing harness on CUDA: `python brdf_guiding_eval.py --gates-only --quick --device cuda` passes (unbiased gate + flow normalization) before touching precision code. **Done:** neural unbiased + norm pass on CUDA. (One pre-existing `--quick` artifact: the VNDF *baseline* trips VNDF-bias at r=0.06 — a 512² quadrature reference under-resolving the near-delta lobe, not the neural sampler and not introduced here.)

## 2. Extended precision axis (NVIDIA float types)

- [x] 2.1 `train.py` (`ConditionalSplineFlow2D` / `SplineCoupling`): thread a `precision` parameter to the GEMM site (`_run_net`); `NF_WT`/`NF_CT` (storage vs accumulate) split; softmax / cumsum / softplus / inverse-quadratic solve and the pdf stay fp32 in **every** mode. `train.py`'s copy only — `render_records.py` untouched.
- [x] 2.2 Add `bf16-compute` (GEMM in bf16, fp32 accumulate) and `tf32` (`allow_tf32`, restored after) modes; `fp32` path is byte-identical (early return `self.net(x_in)`, allow_tf32 untouched). **Verified:** fp32 drift = 0.0 vs pre-change.
- [x] 2.3 Add `fp8-e4m3` and `fp8-e5m2` via `torch._scaled_mm`: dynamic per-tensor amax scaling, fp32 accumulate, mult-of-16 K/N padding. **Note:** Ada `_scaled_mm` forbids e5m2×e5m2 — weights are always e4m3, e5m2 applies to activations only. `fp8_supported()` skips the cell (logged) off sm_89.
- [x] 2.4 Precision axis lives in the grid driver, **not** `brdf_guiding_eval.py` (which has no `--precision` arg). `brdf_bake_grid.py` `PRECISIONS_CUDA` + `--precisions` route `{fp32, tf32, fp16-storage, fp16-compute, bf16-compute, fp8-e4m3, fp8-e5m2}` to the 2.1–2.3 paths. **Verified:** all 7 run on CUDA, drift ordered fp32 < tf32 ≈ fp16-storage < fp16-compute < bf16 < fp8-e4m3 < fp8-e5m2.

## 3. Real GPU cost measurement

- [ ] 3.1 Replace the MPS/wall-clock inference timing with `torch.cuda.Event` start/stop around the inference GEMM: warmup iterations, N timed iterations, `torch.cuda.synchronize()`. Time at a realistic batch (M = spp × cases) so the GEMM shape is the large-M / K≈64 case where TensorCores can help.
- [ ] 3.2 Report, per cell, real ms **and** the fp32→precision speedup ratio; label each cell launch/bandwidth-bound vs compute-bound from the ratio (do not assert a TensorCore win — measure it).

## 4. Full-grid CUDA driver

- [x] 4.1 `brdf_bake_grid.py --study cuda`: sweeps the 9 sizes × the precision axis on CUDA; per cell train (steps ∝ weight count, seeded) + eval; emits `manifest.json` (with study/device/infer_batch) + `size_precision.csv` with the real-cost columns (`infer_ms`, `speedup`).
- [x] 4.2 `log()` exactly which cells ran + the resolved precision list; fp8 cells dropped+logged off sm_89; `--study cuda` off-CUDA warns and emits NaN cost (no silent fake numbers).

## 5. Gates + results doc

- [x] 5.1 Per-cell gates (reuse the FINDINGS rigor): unbiased (`norm_med`, the firefly-robust structural gate) + large-N probe; per-roughness firefly p99.9 (low/mid/high); pdf-parity drift vs the fp32 self reported per precision. All flow through the cuda study unchanged.
- [x] 5.2 `spline_flow/BRDF_CUDA_PRECISION.md` written from the **full 54-cell** real-4090 run (committed `f73f007`): real cost grid (`infer_ms` + `speedup`), per-precision parity, per-roughness firefly, low-roughness equal-time vs MIS, failure-axis verdict + Pareto + speedup summary.
- [x] 5.3 Cross-references both ways: the CUDA doc links back to `BRDF_SIZE_PRECISION.md` + `FINDINGS.md`; forward pointers added in `FINDINGS.md`, `README.md`, and `BRDF_SIZE_PRECISION.md` (+ its generator, durable). **Result: rejection RE-SEALED on real hardware** — min ff_low=2.1 (target ~1), best eq_low=0.0045 (loses equal-time ~1000×), best non-fp32 speedup 1.1× (fp8 0.72–0.78× *slower*, launch-bound K=64 net), fp8 parity drift up to 11; all modes unbiased (norm 0.998–1.000).
