# MLX Neural Trainer Backend

## Why

On Apple-Silicon Mac hosts the only working `--neural-trainer` today is the numpy
CPU oracle — correct but slow (pure-Python autodiff, fp32 only), which caps how
many Adam steps fit in a frame budget during online training. The `mlx` token is
already reserved in the CLI and factory but raises `NotImplementedError`. Apple
MLX provides a GPU-accelerated, UMA-native array framework that runs the spline-flow
training loop on the same Metal device the renderer uses — closing the "fast trainer
on Mac without CUDA" gap and completing the trainer-backend matrix.

## What Changes

- Implement `MlxTrainingBackend` in `src/skinny/sampling/training_backends.py`
  conforming to the existing `TrainingBackend` ABC (`is_available`,
  `supports_precision`, `warm_start`, `update`, `export`): the 6-layer
  `ConditionalSplineFlow2D` forward pass, contribution-weighted MLE loss, and Adam
  optimizer expressed in MLX (`mlx.core` / `mlx.nn`), exporting fp32 NFW1 weights
  identical in format to the numpy/torch backends.
- Replace the `mlx` factory branch raise with real construction, gated on
  `import mlx` success and Metal availability; clear error when MLX is not
  installed or the host has no Metal device.
- Extend `auto` selection: on a Metal-capable Apple-Silicon host prefer `mlx`
  when importable, else fall back to the numpy oracle (CUDA hosts keep
  `auto → cuda`).
- `--train-precision fp16` support on MLX (mlx native float16 compute, fp32
  master weights / fp32 NFW1 export), mirroring the torch autocast contract.
- Numeric parity tests: MLX backend vs numpy oracle (loss trajectory + exported
  weights within tolerance), following the existing torch-vs-numpy parity suite.
- Optional dependency: `mlx` added as a `[mlx]` extra in `pyproject.toml`; never
  a hard import.
- Docs: compatibility matrix rows for `--neural-trainer mlx` in `README.md` and
  `CLAUDE.md`, plus `docs/NeuralGuiding.md` trainer-backend section.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `neural-training-backends`: the `mlx` trainer token changes from
  "reserved — raises" to a working GPU backend on Apple-Silicon Metal hosts;
  `auto` selection on Metal hosts changes from "numpy oracle" to "mlx if
  importable, else numpy oracle"; fp16 train precision becomes supported on MLX
  (was CUDA-autocast-only).

## Impact

- **Code**: `src/skinny/sampling/training_backends.py` (new backend class +
  factory/auto changes), `src/skinny/cli_common.py` (help text),
  `src/skinny/sampling/neural_trainer.py` (precision fallback note only, if any),
  `pyproject.toml` (`[mlx]` extra).
- **Unchanged seams**: `TrainingBackend` ABC, dataset contract
  (`build_dataset_np` → cond[N,9]/z[N,2]/w[N]), NFW1 export format,
  `--neural-handoff file|interop` (handoff consumes fp32 NFW1 regardless of
  trainer), inference precision dials (fp32/fp16/fp8 unchanged).
- **Tests**: extend `tests/test_training_backends.py` (selection, capability
  gating) and add MLX↔numpy parity coverage alongside
  `tests/test_neural_parity.py`; all MLX tests skip cleanly when `mlx` absent.
- **Dependencies**: new optional `mlx` extra (Apple-Silicon-only wheel);
  no change for Vulkan/CUDA users.
- **Docs**: README + CLAUDE.md compatibility matrices, docs/NeuralGuiding.md.
