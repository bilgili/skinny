# metal-backend — Delta (metal-record-drain)

## MODIFIED Requirements

### Requirement: Wavefront, ReSTIR, and neural parity on Metal

The Metal compute surface SHALL support the multi-pass, GPU-driven dispatch the
wavefront axis requires: indirect compute dispatch (dispatch sizes written by a
GPU kernel into an arguments buffer) and a single-frame command encoding that runs
the staged bounce loop with memory barriers between stages rather than a host idle
between every dispatch. Where the installed slang-rhi/SlangPy binding does not
expose indirect dispatch, the system SHALL fall back to a CPU slot-count readback
plus direct dispatch and SHALL select the path through a one-time, logged
capability probe. All pipeline/uniform parameters SHALL continue to be set via
`set_data` byte blobs, never per-field cursor writes.

Rendering the wavefront path/BDPT integrators, ReSTIR DI reuse, and the neural
directional proposal on the Metal backend SHALL reach image equivalence with the
Vulkan backend: exact agreement on the deterministic structural outputs and
agreement within the established perceptual tolerance on the converged shaded
color, for the same scene, sample budget, and configuration.

The online neural-training loop SHALL run fully on the Metal backend: the
wavefront path integrator emits training records the live drain reads each frame
(capability `wavefront-native-path-records`), the trainer consumes them, and the
trained weights publish back through the unified-memory interop handoff
(capability `neural-online-training`) with the frame-end swap incrementing the
network version — no Vulkan device, no megakernel record dispatch, and no file
round-trip required anywhere in the loop.

#### Scenario: Indirect dispatch drives the per-material shade

- **WHEN** the wavefront bounce loop runs on the Metal backend with a constructible
  indirect-dispatch binding
- **THEN** the per-material shade and tiled stages dispatch from the GPU-written
  arguments buffer with no host round-trip, and the frame's fences signal without an
  indefinite wait

#### Scenario: Indirect-dispatch fallback stays correct

- **WHEN** the installed binding cannot dispatch indirect on Metal
- **THEN** the loop reads the slot counts back to the host and issues equivalent
  direct dispatches, the probe logs the slow path, and the rendered image is
  unchanged from the indirect path

#### Scenario: Metal wavefront matches Vulkan wavefront

- **WHEN** the same scene is rendered with the wavefront path integrator (and again
  with BDPT, ReSTIR DI, and the neural proposal) on the Metal backend and on the
  Vulkan backend at an equal sample budget
- **THEN** the structural outputs are exact across backends and the converged color
  agrees within the established perceptual tolerance

#### Scenario: Fully-on-Metal online training loop

- **WHEN** online training is enabled on the native Metal backend with the
  wavefront path integrator, a neural proposal active, and
  `--neural-handoff interop`
- **THEN** path records drain into the replay buffer each frame, the trainer
  publishes updated weights through the shared-storage handoff, the network
  version increments at frame-end swaps, and the converged mixture estimate
  remains unbiased — with no Vulkan device constructed and no NFW1 file written
