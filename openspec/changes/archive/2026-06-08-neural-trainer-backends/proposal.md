## Why

Online neural-proposal training currently hard-codes a single compute path:
`NeuralTrainer.train_cycle()` either runs a torch + spline_flow loop (with
`device=cpu|mps|cuda` and autocast-fp16 hard-wired to CUDA) or, when torch is
absent, a placeholder that fakes weights so the loop stays exercised but learns
nothing. There is no way to swap the training compute framework, and a torch-free
box (notably macOS CI) cannot actually train. To bring training to Apple Silicon
(MLX, later) and to validate any new framework against a trusted oracle, the
compute step must sit behind an interface. The same change formalizes precision
as explicit dials — training precision and inference precision — so quality-vs-cost
studies can vary them independently, including a new fp8 inference-storage mode.

## What Changes

- Introduce a `TrainingBackend` interface (`is_available`, `supports_precision`,
  `warm_start`, `update`, `export`) behind which the per-cycle gradient step runs.
  `NeuralTrainer` stays the orchestrator; its `train_cycle` contract is unchanged.
- Adapt the existing torch/CUDA loop onto the interface as `TorchTrainingBackend`
  (device + autocast move inside it).
- Ship a torch-free `NumpyTrainingBackend` as the **reference oracle**: a
  hand-written forward + backward of the contribution-weighted MLE on the shipped
  flow. It becomes the guaranteed-available fallback and **replaces the placeholder**
  — torch-free boxes now train for real instead of faking weights. **BREAKING** for
  the trainer's torch-absent behavior (placeholder removed).
- Add a skinny-owned numpy dataset contract `build_dataset_np(batch, bounds) →
  (cond, z, w)` (float32, contiguous) shared by all backends, with a parity test
  against spline_flow's torch `build_dataset`.
- Select the backend by source token: `--neural-trainer cpu|cuda|mlx` (`cpu`→numpy,
  `cuda`→torch+CUDA, `mlx`→later). `auto` picks CUDA when available, else numpy.
- Add a `train_precision` dial (`fp32` | `fp16`) independent of inference precision
  (post-training-quantization model; infer defaults to match train). Reduced
  precision stays correctness-safe because the pdf is full-precision — it costs
  variance, never bias.
- Add `fp8-storage` (e4m3) to the inference precision set: host casts fp32→e4m3,
  the shader decodes to float in the scalar GEMM. Manual decode needs no device
  feature, so it is portable across Vulkan/Metal/MoltenVK. fp8 *training/compute*
  is out of scope (cuda-only, marginal at hidden=96).
- `export()` bakes **in-memory**, removing the current per-cycle tempfile
  write+read round-trip in `_bake()`.
- MLX is explicitly a later change; the registry stays extensible for it.

## Capabilities

### New Capabilities
- `neural-training-backends`: a pluggable training-compute backend behind a common
  interface — the numpy reference oracle, the torch adapter, the shared numpy
  dataset contract, backend selection + capability gating, the `train_precision`
  dial, and the in-memory bake. (MLX deferred.)

### Modified Capabilities
- `neural-online-training`: the "async trainer" requirement now delegates the
  per-cycle compute to a selectable `TrainingBackend` rather than a fixed
  torch/placeholder branch, and the torch-free (Mac) path **trains via the numpy
  reference** instead of running the placeholder.
- `neural-precision-size-study`: the selectable inference-precision set gains
  `fp8-storage` (e4m3, manual in-shader decode, no device feature required) while
  the spline core and reported pdf stay full precision.

## Impact

- **Code**: `src/skinny/sampling/` — new `training_backends.py` (or a package),
  `build_dataset_np`; refactor `neural_trainer.py` (remove two-tier branch, drop
  placeholder, in-memory bake); `neural_weights.py` (`NeuralPrecision.FP8_STORAGE`,
  e4m3 encode in `weight_bytes_for`/`bias_bytes_for`); `renderer.py`
  (`enable_online_training` wiring); `cli_common.py` + `app.py`
  (`--neural-trainer`, `--train-precision`, settings persistence).
- **Shaders**: `shaders/sampling/neural_flow.slang` — `NF_WT` fp8 decode path;
  recompile the neural `.spv` fp8-storage variant via `slangc`.
- **Config**: `TrainerConfig` gains `backend` + `train_precision`; existing `device`
  stays the torch sub-device; `arch.precision` is the inference-precision dial.
- **Dependencies**: optional pure-numpy autodiff lib (e.g. HIPS `autograd`) for the
  reference backward, kept torch-free; no new mandatory runtime dep.
- **Interaction to confirm**: `FP8_STORAGE` extends `NeuralPrecision`, owned by
  `neural-precision-size-study`; an in-progress `brdf-cuda-precision-study` (12/14)
  also touches precision. Confirm those are compatible/archived so the enum
  extension does not collide.
- **Docs**: `docs/PythonAPI.md` (new symbols), `docs/NeuralGuiding.md` (backend
  seam + precision matrix), `README.md` (flags), `docs/Architecture.md` (defines).
