# neural-training-backends Specification

## Purpose
TBD - created by archiving change neural-trainer-backends. Update Purpose after archive.
## Requirements
### Requirement: Pluggable training-compute backend interface
The system SHALL run the per-cycle training compute behind a `TrainingBackend`
interface — `is_available`, `supports_precision`, `warm_start`, `update`, and
`export` — so the gradient step can be provided by different frameworks without
changing the trainer. `NeuralTrainer` SHALL remain the orchestrator and its
`train_cycle` contract (sample replay → build dataset → backend → return weights)
SHALL be unchanged. A backend SHALL be stateful across cycles, retaining its warm
model and optimizer state (e.g. Adam moments) between calls.

#### Scenario: Trainer delegates the gradient step to the backend
- **WHEN** a training cycle runs
- **THEN** `NeuralTrainer` builds the dataset and calls the active backend's
  `update`, and the backend returns control with a new set of weights via `export`,
  without the trainer knowing which framework ran the step

#### Scenario: Warm state persists across cycles
- **WHEN** a second training cycle runs on the same backend instance
- **THEN** the backend continues from the model and optimizer state of the previous
  cycle rather than rebuilding them from scratch

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

### Requirement: Torch-free numpy reference backend
The system SHALL provide a torch-free `NumpyTrainingBackend` that implements the
forward and backward of the contribution-weighted MLE on the shipped flow
architecture and SHALL be available on any platform with no torch dependency. It
SHALL be the guaranteed-available fallback and SHALL replace the previous
torch-absent placeholder, so a torch-free host performs a real training update
rather than emitting unchanged weights.

#### Scenario: Numpy backend trains without torch
- **WHEN** a training cycle runs on a host without torch
- **THEN** the numpy backend computes a real contribution-weighted MLE update and
  returns weights that differ from the input when the batch carries signal

#### Scenario: Reference matches the torch backend numerically
- **WHEN** the numpy backend and the torch CPU backend run the same fixed batch and
  seed for one cycle
- **THEN** their resulting weights agree within a documented numerical tolerance,
  guarding against drift in either implementation

### Requirement: Shared numpy dataset contract
The system SHALL build training datasets through a single skinny-owned
`build_dataset_np(batch, bounds) → (cond, z, w)` that returns contiguous float32
arrays, consumed by every backend (the torch backend wrapping them with a
zero-copy host view before its device upload). The numpy dataset SHALL match
spline_flow's torch `build_dataset` on a fixed batch.

#### Scenario: Dataset arrays are float32 and contiguous
- **WHEN** `build_dataset_np` is called on a batch
- **THEN** `cond`, `z`, and `w` are returned as contiguous float32 arrays so the
  torch backend can wrap them without an extra host copy or dtype cast

#### Scenario: Numpy dataset matches the spline_flow reference
- **WHEN** the same fixed batch and bounds are run through `build_dataset_np` and
  through spline_flow's torch `build_dataset`
- **THEN** the produced `(cond, z, w)` values agree within tolerance

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

### Requirement: In-memory weight bake
The system SHALL bake the live model back to `NeuralWeights` in memory, without a
per-cycle temporary-file write-and-read round-trip.

#### Scenario: Bake performs no filesystem round-trip
- **WHEN** `export` runs at the end of a training cycle
- **THEN** the resulting `NeuralWeights` is produced in memory with no temporary
  file written or read

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

MLX arrays and GPU streams are thread-affine. The backend SHALL therefore
**confine all of its MLX work to a single owned worker thread**: `warm_start`,
`update`, and `export` SHALL execute their MLX operations on one dedicated
thread the backend owns, so the leaf arrays are always created and evaluated on
the same thread regardless of which caller thread invokes the backend. Driving
the backend from more than one thread SHALL be serialized onto that worker and
SHALL NOT raise `There is no Stream(gpu, N) in current thread.`. Single-thread
(production) use SHALL be functionally unchanged.

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

#### Scenario: Driven from multiple threads without a stream crash
- **WHEN** the backend is invoked from a thread other than the one that
  warm-started it (e.g. a direct call concurrent with the background daemon
  trainer thread)
- **THEN** the MLX work is marshaled onto the backend's single owned worker
  thread and completes, rather than raising
  `There is no Stream(gpu, N) in current thread.`

#### Scenario: Absent MLX package leaves the system unaffected
- **WHEN** skinny is installed without the `mlx` extra
- **THEN** importing and running the renderer, trainer, and tests behaves
  exactly as before this change, with MLX-specific tests skipping cleanly

