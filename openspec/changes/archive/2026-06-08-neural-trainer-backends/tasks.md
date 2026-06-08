## 1. Backend interface + selection

- [x] 1.1 Add `TrainingBackend` ABC (`is_available`, `supports_precision`, `warm_start`, `update`, `export`) in `src/skinny/sampling/training_backends.py`, mirroring the `NeuralWeightPublisher` ABC style
- [x] 1.2 Add `TRAINING_BACKENDS` name-keyed registry + `make_training_backend(kind, device="auto")` factory (`cpu`→numpy, `cuda`→torch+CUDA, `mlx`→reserved), with `auto` selection (CUDA if torch+CUDA else numpy) and a clear error on unavailable explicit token
- [x] 1.3 Unit-test selection + gating: `auto` picks numpy on a torch-free host; explicit `cuda` without torch raises a clear error; `supports_precision` reports unsupported precisions

## 2. Shared numpy dataset contract

- [x] 2.1 Implement `build_dataset_np(batch, bounds) → (cond, z, w)` returning contiguous float32 arrays (geometry mirrors spline_flow's `build_dataset`)
- [x] 2.2 Parity test: `build_dataset_np` vs spline_flow's torch `build_dataset` on a fixed batch + bounds, within tolerance
- [x] 2.3 Assert float32/contiguous so `torch.from_numpy` stays zero-copy (regression guard)

## 3. Torch backend (adapt current CUDA loop)

- [x] 3.1 Port `_build_model` / `_run_steps` / `_bake` into `TorchTrainingBackend(device=cpu|mps|cuda)`; move device + autocast/GradScaler inside it (autocast fires when device==cuda; MPS falls back to fp32 reporting the fallback)
- [x] 3.2 Consume `build_dataset_np` output via `torch.from_numpy(...).to(device)` instead of spline_flow's device-side build
- [x] 3.3 Honor `train_precision` (fp32 | fp16) in `update`
- [x] 3.4 In-memory `export()`: bake live params directly into `NeuralWeights` (remove the tempfile write+read)

## 4. Numpy reference backend (oracle)

- [x] 4.1 Implement `NumpyTrainingBackend`: forward + backward of the contribution-weighted MLE (`-Σ w·log q / Σ w`) on the shipped flow arch, torch-free (optionally via a pure-numpy autodiff lib, e.g. HIPS `autograd`)
- [x] 4.2 Persist warm state (params + Adam moments) across cycles; warm_start from `NeuralWeights`
- [x] 4.3 In-memory `export()` to `NeuralWeights` (fp32 NFW1 layout)
- [x] 4.4 Make it the guaranteed-available fallback (remove the placeholder path)

## 5. Trainer integration

- [x] 5.1 Replace `NeuralTrainer`'s two-tier branch with the selected `TrainingBackend`; keep `train_cycle` contract unchanged (sample → `build_dataset_np` → backend → publish)
- [x] 5.2 Add `backend` + `train_precision` to `TrainerConfig`; keep `device` as the torch sub-device; treat `arch.precision` as `infer_precision` (default: match `train_precision`)
- [x] 5.3 Remove `_placeholder_update`, `_probe` two-tier logic, and the tempfile `_bake`

## 6. fp8 inference-storage precision

- [x] 6.1 Add `NeuralPrecision.FP8_STORAGE` (e4m3) + capability/quarter-footprint helpers in `neural_weights.py`
- [x] 6.2 e4m3 encode in `weight_bytes_for` / `bias_bytes_for`; NFW1 stays fp32 on disk
- [x] 6.3 `neural_flow.slang`: `NF_WT` fp8 decode path (read fp8 bytes → float in the scalar GEMM), no device feature required
- [x] 6.4 Recompile the neural `.spv` fp8-storage variant via `slangc`
- [x] 6.5 Tests: e4m3 round-trip (encode→decode) within tolerance; fp8-storage produces a valid sample + finite positive pdf; weights occupy a quarter of fp32 bytes

## 7. Parity + correctness tests

- [x] 7.1 Numpy backend ≈ torch-cpu backend: same fixed seed/batch, one cycle, weights agree within a documented tolerance (drift guard)
- [x] 7.2 fp16 training still bakes fp32 weights (format identical to fp32 training)
- [x] 7.3 Torch-free host performs a real update (weights change when the batch carries signal), not a no-op

## 8. CLI + wiring + persistence

- [x] 8.1 Add `--neural-trainer cpu|cuda|mlx` and `--train-precision fp32|fp16` to `cli_common.py`
- [x] 8.2 Wire into `renderer.enable_online_training` (construct backend from config)
- [x] 8.3 Persist `neural_trainer` + `train_precision` in `app.py` settings (mirror `neural_handoff` save/restore)

## 9. Cross-change check + docs

- [x] 9.1 Confirm `FP8_STORAGE` enum extension is compatible with `neural-precision-size-study` and the in-flight `brdf-cuda-precision-study` (12/14) — archive/rebase as needed before merging the enum change
- [x] 9.2 `docs/PythonAPI.md`: new symbols (`TrainingBackend`, backends, `make_training_backend`, `build_dataset_np`, `train_precision`, `FP8_STORAGE`)
- [x] 9.3 `docs/NeuralGuiding.md`: training-backend seam + the train/infer precision matrix
- [x] 9.4 `README.md`: `--neural-trainer` / `--train-precision` flags
- [x] 9.5 `docs/Architecture.md`: fp8 `-D NF_WT` define + neural module notes

## 10. Validation

- [x] 10.1 `ruff check src/` clean; `pytest` green (incl. new parity/precision tests)
- [x] 10.2 `openspec validate neural-trainer-backends --strict` passes
