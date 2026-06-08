## MODIFIED Requirements

### Requirement: Async trainer on the shipped flow architecture
The system SHALL train the neural proposal with an asynchronous trainer that
warm-starts from the current weights and performs small recency-weighted updates on
the exact flow architecture and loss the renderer ships. The trainer SHALL delegate
the per-cycle compute to a selectable `TrainingBackend` (see the
`neural-training-backends` capability) rather than a fixed torch/placeholder branch,
running on CUDA on the target hardware and on a torch-free host via the numpy
reference backend.

#### Scenario: Warm-started incremental update
- **WHEN** a training cycle runs
- **THEN** it begins from the current weights and applies a small number of update steps
  rather than retraining from scratch

#### Scenario: Mac trains via the numpy reference
- **WHEN** the trainer runs on a Mac without CUDA
- **THEN** it performs a real training update through the torch-free numpy backend at
  reduced scale, rather than running a placeholder that leaves the weights unchanged
