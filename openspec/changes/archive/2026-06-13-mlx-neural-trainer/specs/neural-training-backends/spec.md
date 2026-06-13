# neural-training-backends Delta — mlx-neural-trainer

## ADDED Requirements

### Requirement: MLX GPU training backend on Apple-Silicon Metal
The system SHALL provide an `MlxTrainingBackend` that implements the
`TrainingBackend` interface using Apple MLX, running the contribution-weighted
MLE update of the shipped flow architecture on the Metal GPU. The backend SHALL
be available only when the `mlx` package is importable and the host is an
Apple-Silicon Metal device; MLX SHALL remain an optional dependency (installable
via a packaging extra) that is never imported unconditionally. The backend SHALL
consume the shared `build_dataset_np` arrays, retain warm model and Adam
optimizer state across cycles, and `export` fp32 `NeuralWeights` identical in
format to the numpy and torch backends.

#### Scenario: MLX backend trains on the Metal GPU
- **WHEN** a training cycle runs with `--neural-trainer mlx` on an Apple-Silicon
  Metal host with `mlx` installed
- **THEN** the MLX backend computes a real contribution-weighted MLE update on
  the GPU and `export` returns fp32 NFW1-format weights that differ from the
  input when the batch carries signal

#### Scenario: MLX matches the numpy reference numerically
- **WHEN** the MLX backend and the numpy backend run the same fixed batch and
  seed for one cycle
- **THEN** their resulting weights agree within a documented numerical
  tolerance, guarding against drift in either implementation

#### Scenario: Absent MLX package leaves the system unaffected
- **WHEN** skinny is installed without the `mlx` extra
- **THEN** importing and running the renderer, trainer, and tests behaves
  exactly as before this change, with MLX-specific tests skipping cleanly

## MODIFIED Requirements

### Requirement: Backend selection by source token with capability gating
The system SHALL select the training backend by a source token
(`--neural-trainer cpu|cuda|mlx`, mapped onto the framework backends: `cpu`→numpy,
`cuda`→torch on CUDA, `mlx`→MLX on Apple-Silicon Metal), defaulting to `auto`
which picks CUDA when torch + a CUDA device are available, otherwise MLX when
the `mlx` package is importable on an Apple-Silicon Metal host, and otherwise
the numpy backend. An explicitly requested backend or precision that the host
cannot provide SHALL fail with a clear message rather than silently degrading.

#### Scenario: Auto selects CUDA when available, numpy otherwise
- **WHEN** the trainer starts with the default `auto` selection
- **THEN** it uses the torch CUDA backend on a CUDA box and the numpy backend on
  a box without torch/CUDA, MLX, or a Metal device

#### Scenario: Auto prefers MLX on an Apple-Silicon Metal host
- **WHEN** the trainer starts with `auto` on an Apple-Silicon Metal host where
  `mlx` is importable and no CUDA device is present
- **THEN** it selects the MLX backend, and logs the selected backend at startup

#### Scenario: Unavailable explicit backend fails clearly
- **WHEN** `--neural-trainer cuda` is requested on a machine without torch or a
  CUDA device, or `--neural-trainer mlx` is requested where `mlx` is not
  importable or the host has no Apple-Silicon Metal device
- **THEN** selection raises a clear error naming the missing requirement rather
  than silently falling back

#### Scenario: Unsupported precision is reported
- **WHEN** a `train_precision` is requested that the selected backend/device does
  not support
- **THEN** `supports_precision` reports it as unsupported and the trainer
  surfaces a clear message rather than running an unsupported path

### Requirement: Selectable training precision independent of inference precision
The system SHALL expose a `train_precision` dial (`fp32` | `fp16`) that controls
the optimizer's compute precision and SHALL be independent of the inference
precision, defaulting inference to match training (post-training quantization).
Training SHALL always bake full-precision (fp32) weights, so the choice of
training precision does not change the on-disk/handoff weight format. `fp16`
training SHALL use the framework's mixed-precision facilities where the device
supports them — torch autocast on CUDA, and reduced-precision compute with fp32
master weights on MLX — and SHALL fall back to fp32 where it does not (including
a runtime fall-back to fp32 with a one-time warning if the MLX fp16 loss becomes
non-finite).

#### Scenario: fp16 training still bakes fp32 weights
- **WHEN** training runs at `train_precision=fp16` on a capable device
- **THEN** the linear-layer GEMMs run in reduced precision but the exported
  weights are full-precision fp32, identical in format to fp32 training

#### Scenario: Training precision and inference precision are set independently
- **WHEN** a `train_precision` is selected and no inference precision override is
  given
- **THEN** inference precision defaults to match the training precision, and
  either dial can be overridden without affecting the other

#### Scenario: MLX fp16 falls back to fp32 on non-finite loss
- **WHEN** an MLX fp16 training step produces a non-finite loss
- **THEN** the backend completes the session in fp32, warns once, and the
  exported weights remain valid fp32 NFW1
