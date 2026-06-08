# neural-online-training Specification

## Purpose
TBD - created by archiving change neural-online-training. Update Purpose after archive.
## Requirements
### Requirement: Recency-weighted replay buffer from live records
The renderer SHALL feed its per-vertex `(x, wi, contribution)` path records into a
recency-weighted replay buffer consumed live by the trainer, using the same record
layout the offline dump produces, so that recent paths dominate and the learned
distribution tracks scene motion.

#### Scenario: Live records accumulate
- **WHEN** the renderer traces frames with the neural proposal and online training active
- **THEN** its emitted path records are drained into the replay buffer each frame rather
  than written to a file

#### Scenario: Recent paths dominate
- **WHEN** the replay buffer is sampled for a training step after the scene has changed
- **THEN** recent records are weighted above older ones so the trainer adapts toward the
  current scene state

### Requirement: Async trainer on the shipped flow architecture
The system SHALL train the neural proposal with an asynchronous trainer that
warm-starts from the current weights and performs small recency-weighted updates on
the exact flow architecture and loss the renderer ships. The trainer SHALL delegate
the per-cycle compute to a selectable `TrainingBackend` (see the
`neural-training-backends` capability) rather than a fixed torch/placeholder branch,
running on CUDA on the target hardware and on a torch-free host via the numpy
reference backend. The async trainer SHALL be reachable from the interactive
front-ends: a user-facing trigger (see the `online-training-control` capability)
enables it after the scene is ready and a per-frame driver runs the
drain → train → publish → frame-end-swap loop, with the per-cycle training
executing off the render thread so it does not stall rendering.

#### Scenario: Warm-started incremental update
- **WHEN** a training cycle runs
- **THEN** it begins from the current weights and applies a small number of update steps
  rather than retraining from scratch

#### Scenario: Mac trains via the numpy reference
- **WHEN** the trainer runs on a Mac without CUDA
- **THEN** it performs a real training update through the torch-free numpy backend at
  reduced scale, rather than running a placeholder that leaves the weights unchanged

#### Scenario: Front-end trigger drives the loop
- **WHEN** an interactive front-end is started with online training enabled and its
  prerequisites met
- **THEN** the trainer is constructed, the per-frame driver drains records and the
  background trainer publishes new weights that the frame-end swap promotes,
  without a slow cycle stalling the render

### Requirement: Two selectable weight-handoff backends
The system SHALL provide two weight-handoff backends behind a common publisher
interface selectable at runtime: a file double-buffer backend and a GPU-shared
-memory Vulkan↔CUDA interop backend; the file backend SHALL work without CUDA and
the interop backend SHALL be guarded so it fails clearly where CUDA or the required
Vulkan external-memory extension is unavailable.

#### Scenario: File backend publishes without CUDA
- **WHEN** `--neural-handoff file` is selected on any platform
- **THEN** the trainer writes new weights to a file the renderer hot-reloads through the
  existing loader, with no CUDA dependency

#### Scenario: Interop backend guarded off-CUDA
- **WHEN** `--neural-handoff interop` is selected on a machine without CUDA or the required
  external-memory extension
- **THEN** the backend reports it as unavailable with a clear message rather than silently
  degrading or crashing

#### Scenario: Interop backend shares GPU memory on CUDA
- **WHEN** `--neural-handoff interop` is selected on a CUDA machine with external memory
  support
- **THEN** CUDA writes updated weights into the Vulkan-exported weight buffer with no CPU
  round-trip

### Requirement: Double-buffered swap increments the network version
The renderer SHALL hold the render-side weights frozen for the duration of a frame
and SHALL swap to newly published weights only at a frame boundary, incrementing the
network version on each successful swap.

#### Scenario: Swap at frame boundary
- **WHEN** the trainer publishes new weights mid-frame
- **THEN** the render-side weights are unchanged until the frame completes, and the swap
  occurs at the frame boundary

#### Scenario: Version increments on swap
- **WHEN** a weight swap completes
- **THEN** the network version is greater than before the swap

### Requirement: Staleness raises variance only, never bias
The estimator SHALL remain unbiased under asynchronous weight swaps by evaluating
each sample's density against the network version that produced it, so that a swap
in flight changes variance but not the expected value.

#### Scenario: Sample uses its stamped version's density
- **WHEN** a sample drawn under one network version is evaluated after a swap to a newer
  version
- **THEN** its density is computed from the version stamped on the sample, not the current
  version

#### Scenario: Converges to the reference across swaps
- **WHEN** the renderer runs with online training active and weights swapping between frames
- **THEN** the converged image matches the unbiased reference for that scene state

