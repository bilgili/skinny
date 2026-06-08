## Context

Online neural-proposal training (`neural-online-training`) runs an async trainer
that warm-starts the shipped `ConditionalSplineFlow2D` (cond=9, layers=6, bins=24,
hidden=96) and does small recency-weighted contribution-weighted MLE updates, then
bakes a `NeuralWeights` the renderer double-buffers in. Today
`NeuralTrainer.train_cycle()` hard-codes a two-tier branch: a torch + spline_flow
loop (`device=cpu|mps|cuda`, autocast-fp16 wired to CUDA) and, when torch is absent,
a placeholder that returns unchanged weights so the publish→swap→version loop stays
exercised but learns nothing.

Two facts shape this change:

- **The fp32 boundary already decouples training from inference precision.** NFW1 on
  disk and the `NeuralWeights` container are always fp32; inference precision is an
  upload-time cast (`weight_bytes_for`) plus a shader compile variant
  (`-D NF_WT/NF_CT`). The RQ-spline math and the reported solid-angle pdf stay
  full-precision float in every inference mode.
- **The weight-handoff seam is already abstracted and orthogonal.**
  `NeuralWeightPublisher` (`file` | `interop`) owns how weights reach the renderer;
  this change does not touch it except that the upload-time precision cast flows
  through it unchanged.

This change abstracts the remaining hard-coded axis — the *training compute* — and
formalizes precision as explicit dials, including a new fp8 inference-storage mode.

## Goals / Non-Goals

**Goals:**
- A `TrainingBackend` interface the per-cycle gradient step plugs into, with the
  current torch/CUDA loop adapted onto it and a torch-free numpy reference oracle.
- A single skinny-owned numpy dataset contract shared by all backends.
- Independent `train_precision` and `infer_precision` dials under a post-training
  quantization (PTQ) model; a new `fp8-storage` (e4m3) inference mode.
- In-memory weight bake (remove the per-cycle tempfile round-trip).
- Real training on torch-free hosts (macOS CI) via the numpy backend.

**Non-Goals:**
- The MLX backend (later change; the registry stays extensible).
- fp8 *training* or fp8 *compute* (cuda-Hopper/Ada + torchao/Transformer-Engine, or
  a cooperative-matrix shader rewrite; marginal at hidden=96).
- Quantization-aware training (QAT). Deferred; only warranted if PTQ fp8-inference
  variance proves too high.
- Any change to the weight-handoff seam (`file` | `interop`) or to the
  double-buffered swap / version-stamping behavior.

## Decisions

### Seam placement: orchestrator keeps the loop, backend owns the math
`NeuralTrainer` stays the orchestrator (replay sampling, dataset build, version and
loss bookkeeping, `publish`). The backend owns only the framework-specific work
behind a small stateful interface:

```
class TrainingBackend(abc.ABC):
    name: str
    def is_available(self) -> bool
    def supports_precision(self, p, device) -> bool
    def warm_start(self, weights: NeuralWeights, cfg) -> None   # build live model once
    def update(self, cond, z, w) -> float | None               # steps_per_cycle steps
    def export(self) -> NeuralWeights                           # bake fp32 NFW1, in-memory
```

The backend is **stateful** — it retains the warm model + optimizer (Adam moments)
across cycles, exactly as the current torch loop keeps `_model/_opt/_scaler`.

*Alternative considered — wrap the whole cycle* (`train_cycle(batch, weights, cfg)`
per backend): rejected because each backend would re-implement warm-start, the
minibatch loop, and the bake, duplicating the NFW1-layout contract and inviting
drift. The narrow `warm_start`/`update`/`export` split keeps orchestration shared.

### Taxonomy: classes by framework, CLI tokens by source
Internal classes are `TorchTrainingBackend(device=cpu|mps|cuda)` and
`NumpyTrainingBackend` (MLX later). CLI exposes the user's mental model:
`--neural-trainer cpu|cuda|mlx` (`cpu`→numpy, `cuda`→torch+CUDA, `mlx`→later). A
`make_training_backend(kind, device="auto")` factory + `TRAINING_BACKENDS`
name-keyed dict mirror `sampling/registry.py` and the `make_publisher` /
`NeuralWeightPublisher` house style. `auto` picks CUDA if torch+CUDA, else numpy;
an unavailable explicit token raises clearly (as `make_publisher` does). "source"
is avoided as a class noun because it already means *record source*
(megakernel vs wavefront) elsewhere in the renderer.

### CPU backend = torch-free numpy oracle
The `cpu` backend is a hand-written forward + backward of the contribution-weighted
MLE (`loss = -Σ w·log q / Σ w`, `w = luminance(contribution)`) on the shipped flow,
in numpy with no torch dependency. It MAY use a pure-numpy reverse-mode autodiff
library (e.g. HIPS `autograd`) to avoid hand-deriving the RQ-spline gradients while
staying torch-free. It serves three purposes: (1) an **independent numeric oracle**
to catch drift in the torch and (future) MLX backends — MLX cannot execute a
`torch.nn.Module`, so it reimplements from scratch and needs something independent
to check against; (2) **real training on torch-free hosts**; (3) the **canonical
spec** of one training cycle. It replaces the placeholder as the guaranteed-available
fallback.

*Alternative considered — torch device=cpu as the "reference":* trivial, but then
`cpu` and `cuda` are one backend with a device knob, there is no torch-free training,
and parity is torch-vs-torch with no independent oracle. Rejected; the numpy
contract makes the oracle cheap enough to be worth its weight.

### Dataset contract: one numpy `build_dataset_np`, float32, contiguous
A skinny-owned `build_dataset_np(batch, bounds) → (cond, z, w)` returns contiguous
float32 arrays consumed by every backend; the torch backend wraps them with
`torch.from_numpy(...).to(device)`. Because `torch.from_numpy` shares the host
buffer, this adds **no GPU copy** over today: the replay buffer is already host
numpy, so the batch→device H2D was always paid exactly once; the numpy build only
moves the feature-engineering compute to the CPU (µs–ms at batch=4096) and keeps a
single dataset codepath. float32 is mandatory so `from_numpy` stays zero-copy (no
cast) and so the numpy build does not drift from the reference. The math duplicates
spline_flow's torch `build_dataset`; a parity test on a fixed batch guards the
duplication (spline_flow keeps its own for offline use).

### Precision: PTQ with two independent dials
`train_precision` ∈ {fp32, fp16} controls only the optimizer compute (fp16 = torch
autocast+GradScaler on CUDA, partial on MPS; cpu/numpy = fp32). `infer_precision`
is the existing inference dial extended with `fp8-storage`. They are independent;
inference defaults to match training. This is sound because reduced precision here
is **variance, not bias**: the pdf is full-precision in every inference mode, so a
lower-precision GEMM perturbs the *proposal* but never invalidates the *density* —
the estimator stays unbiased (consistent with `neural-precision-size-study`'s
"Unbiased composition" requirement). PTQ (train fp32/fp16 → cast at upload) is
therefore correctness-safe and simplest.

`supports_precision(p, device)` gates "if hardware supports", mirroring the existing
inference gates (`needs_device_fp16_compute` → `shaderFloat16`,
`needs_device_fp16_storage` → 16-bit storage). fp8-storage inference needs no gate.

### fp8 = inference storage only (e4m3), manual decode
`NeuralPrecision` gains `FP8_STORAGE`: the host casts fp32→e4m3 in
`weight_bytes_for`/`bias_bytes_for`; `neural_flow.slang` reads the fp8 bytes and
decodes to float in the scalar GEMM. The scalar GEMM (no tensor-core MMA) means fp8
*compute* is not reachable without a cooperative-matrix rewrite, and the manual
decode needs no device feature — so fp8-storage is the most *portable* precision
added (works on Vulkan, Metal, MoltenVK alike), even though fp8 is usually a
big-GEMM win and is marginal at hidden=96. e4m3 (not e5m2) for weights: more
mantissa, the standard choice for weight storage.

### export() bakes in-memory
The current `_bake()` writes a temp NFW1 file and reloads it every cycle. `export()`
bakes the flow's live parameters directly into a `NeuralWeights` in memory, removing
the per-cycle disk write+read — a latency and IO win on the online loop. The NFW1
layout stays the single source of truth, just applied in memory.

## Risks / Trade-offs

- **Numpy reference backward is real work and must track the flow arch** → lean on a
  pure-numpy autodiff lib (HIPS `autograd`) rather than hand-deriving; pin
  correctness with a parity test (numpy ≈ torch-cpu on a fixed seed/batch within a
  documented tolerance). If the arch changes, the parity test fails loudly.
- **`build_dataset_np` duplicates spline_flow's torch build** → a fixed-batch parity
  test against spline_flow's `build_dataset` guards drift; spline_flow remains the
  offline source of truth.
- **fp8 PTQ variance** → fp8-storage is correctness-safe (no bias) but may raise
  variance; the precision-size study already reports drift rather than hiding it.
  QAT remains the escape hatch if it proves too high.
- **`FP8_STORAGE` extends an enum owned by `neural-precision-size-study`, and
  `brdf-cuda-precision-study` (12/14) is in flight** → confirm those are
  compatible/archived before extending the enum, to avoid a collision.
- **fp16 training on MPS is partial in torch** → `supports_precision` falls back to
  fp32 on MPS where autocast is unsupported, reporting the fallback.

## Migration Plan

- The placeholder backend is removed; torch-absent hosts now run the numpy backend.
  This changes torch-free behavior from "weights unchanged" to "real update" — a
  behavior change for that path, but the public `train_cycle` contract is unchanged.
- Existing `--neural-handoff` flag and persisted settings are untouched. New
  `--neural-trainer` / `--train-precision` flags default to `auto`/match-infer, so
  existing invocations behave as before (CUDA box → torch+CUDA; Mac → numpy instead
  of placeholder).
- Recompile the neural `.spv` fp8-storage variant via `slangc`; existing precision
  variants are unchanged.

## Open Questions

- Whether to take the optional HIPS `autograd` dependency for the numpy backward, or
  hand-derive the RQ-spline + MLP gradients to keep the dependency set minimal.
- Final CLI surface for `train_precision` vs the existing inference-precision flag
  (one flag with a `train:`/`infer:` syntax, or two flags) — resolved during the CLI
  task.
