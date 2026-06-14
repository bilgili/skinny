## MODIFIED Requirements

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
