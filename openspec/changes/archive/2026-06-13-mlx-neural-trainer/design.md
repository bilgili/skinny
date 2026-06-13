# Design — MLX Neural Trainer Backend

## Context

Online training of the neural directional proposal runs behind the
`TrainingBackend` ABC (`training_backends.py:167`): `is_available`,
`supports_precision`, `warm_start`, `update`, `export`. Two backends exist —
`TorchTrainingBackend` (CUDA, fp16 autocast) and `NumpyTrainingBackend`
(torch-free fp32 oracle). The `mlx` token is reserved: `TRAINING_BACKENDS`
lists it, `make_training_backend` raises `NotImplementedError` at
`training_backends.py:276`, and the CLI help marks it "reserved".

The model is the shipped `ConditionalSplineFlow2D` (6 coupling layers, 24 bins,
hidden 96, cond 9 — `neural_weights.py:38`), trained by contribution-weighted
MLE (`loss = -Σ w·log q / Σ w`) with Adam. Dataset prep is backend-agnostic:
`build_dataset_np(batch, bounds) → (cond[N,9], z[N,2], w[N])` float32
contiguous. Export bakes fp32 NFW1 `NeuralWeights` in memory; the handoff
(`file` or `interop`) is downstream and format-blind to the trainer.

On a Mac host (`--backend metal`), `auto` resolves to the numpy oracle today —
correct but pure-Python slow, limiting Adam steps per cycle. Apple MLX runs the
same training math on the Metal GPU with unified memory and a torch-like API.

## Goals / Non-Goals

**Goals:**
- Working `--neural-trainer mlx`: GPU training on Apple-Silicon Metal via MLX,
  numerically matching the numpy oracle within the existing parity tolerance.
- `auto` on a Metal-capable Apple-Silicon host prefers `mlx` when importable,
  falling back to the numpy oracle; CUDA hosts unchanged (`auto → cuda`).
- `--train-precision fp16` on MLX: reduced-precision compute, fp32 master
  weights, fp32 NFW1 export (same contract as torch autocast).
- Optional dependency (`pip install -e ".[mlx]"`); zero impact when absent.

**Non-Goals:**
- No change to the `TrainingBackend` ABC, `train_cycle` orchestration,
  dataset contract, NFW1 format, handoff paths, or inference precision dials.
- No MLX inference — inference stays in-shader (Slang); MLX is trainer-only.
- No MLX-on-CUDA / MLX-on-Linux support: the `mlx` token is gated to
  Apple-Silicon Metal hosts even though upstream MLX has other backends.
- No GUI changes — trainer choice is CLI-only today and stays so.

## Decisions

### D1: Mirror the torch backend structure with `mlx.nn` + `mlx.optimizers`
Implement `MlxTrainingBackend` as a sibling of `TorchTrainingBackend`: an
`mlx.nn.Module` spline flow mirroring the torch module layout (3 Linears per
coupling, `_LINEAR_IDX=(0,2,4)`), `mlx.optimizers.Adam` with the same
hyperparameters, and `mlx.nn.value_and_grad` for the loss step.

*Alternative considered:* functional `mlx.core`-only implementation mirroring
the numpy oracle. Rejected — the torch backend is the closer API match (MLX
deliberately apes torch), and module/optimizer state gives warm-start and Adam
moments for free, satisfying the "stateful across cycles" requirement.

### D2: Force evaluation each update step
MLX is lazy: graphs build until `mx.eval()`. Each `update` call ends with
`mx.eval(model.parameters(), optimizer.state)` so the cycle's wall-time cost is
paid inside `update` (where the trainer budget measures it), not deferred into
`export`. `export` then reads back via `np.array(...)` per tensor (UMA — cheap).

### D3: Availability gate = import + platform, checked in `is_available`
`mlx` is constructible only when `import mlx.core` succeeds **and**
`platform.machine() == "arm64"` / `platform.system() == "Darwin"` (equivalently
`mx.metal.is_available()`). Explicit `--neural-trainer mlx` on a failing host
raises a clear error naming the missing piece (no silent fallback), matching
the existing capability-gating requirement. The import lives inside the
backend module function, never at module top level.

### D4: `auto` precedence — cuda > mlx > numpy
`auto` keeps its CUDA-first rule (a CUDA box stays on torch), then tries MLX
(import + Metal check), then lands on the numpy oracle. This makes the
documented "supported Mac combo" (`--neural-trainer cpu`) strictly opt-in
slower than the new default, and `auto` never errors.

*Alternative considered:* keep `auto → numpy` on Mac and require explicit
`mlx`. Rejected — the point of `auto` is "best available"; the parity tests
make MLX trustworthy enough to be the Mac default.

### D5: fp16 = compute-dtype cast with fp32 master weights
MLX has no autocast. For `train_precision=fp16`: keep fp32 master parameters
in the optimizer, cast parameters + inputs to `float16` for the
forward/backward, accumulate the loss in fp32, apply fp32 gradients. If the
fp16 loss goes non-finite the cycle falls back to fp32 with a one-time warning
(same observable contract as the torch path's GradScaler skip). `export`
always bakes fp32 — unchanged NFW1.

*Alternative considered:* declare fp16 unsupported on MLX (fall back like
numpy). Kept as the fallback posture if instability shows up during
implementation, but the spec scenario targets working fp16.

### D6: Parity is asserted against the numpy oracle, not torch
The CI/dev box for this change is the Mac; torch-CUDA isn't present. The
existing oracle is already pinned to torch by `test_neural_parity.py`, so
MLX↔numpy parity transitively covers MLX↔torch. Same fixture pattern: fixed
batch + seed, one cycle, exported weights within documented tolerance.

### D7: Packaging as `[mlx]` extra
`pyproject.toml` gains `mlx = ["mlx>=0.30"]` under optional-dependencies.
Nothing imports MLX unconditionally; all tests `pytest.importorskip("mlx")`.

## Risks / Trade-offs

- [MLX rsplines/gather kernels diverge numerically from numpy at fp32] →
  parity test with explicit tolerance (same budget the torch parity uses);
  if a single op is the offender, compute that op in fp32 explicitly.
- [Lazy-eval graph leak across cycles (state growing per `update`)] →
  `mx.eval` barrier each step (D2) + a multi-cycle test asserting stable
  memory/time per cycle.
- [Adam semantics differ (bias correction, eps placement) between
  `mlx.optimizers.Adam` and the oracle's Adam] → pin hyperparameters
  explicitly; if MLX's Adam variant can't match, implement the oracle's Adam
  update manually in MLX (it is ~10 lines) rather than loosening tolerance.
- [fp16 spline-flow training unstable (logit/softplus ranges)] → D5 fallback:
  non-finite loss ⇒ fp32 for the rest of the session, warn once; spec keeps
  fp32 the default.
- [`auto` behavior change surprises an existing Mac user mid-session] →
  startup log line names the selected backend; docs matrices updated; explicit
  `cpu` token unchanged.
- [MLX wheel availability / version churn] → optional extra with a floor pin,
  import-guarded; absence degrades to today's behavior exactly.

## Open Questions

- None blocking. fp16 stability (D5) resolves during implementation with the
  fallback posture pre-agreed.
