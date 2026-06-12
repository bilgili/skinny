# neural-online-training — Delta (metal-neural-interop)

## MODIFIED Requirements

### Requirement: Two selectable weight-handoff backends
The system SHALL provide two weight-handoff backends behind a common publisher
interface selectable at runtime: a file double-buffer backend and a GPU-shared
-memory interop backend. The file backend SHALL work without CUDA on any platform.
The interop backend SHALL resolve per GPU backend: on Vulkan it SHALL use the
Vulkan↔CUDA external-memory path, and on the native Metal backend it SHALL use a
unified-memory shared-storage path that writes published weights into the
renderer's weight and bias buffers with no disk round-trip and no NFW1
serialize/parse. The Metal interop publish SHALL produce byte-identical buffer
contents to a file-backend publish of the same weights at the same
`NeuralPrecision` (fp32, fp16 storage, fp8 e4m3). The interop backend SHALL be
guarded so it fails clearly, with a message naming the file fallback, on hosts
that have neither CUDA with the required Vulkan external-memory extension nor a
Metal unified-memory device.

#### Scenario: File backend publishes without CUDA
- **WHEN** `--neural-handoff file` is selected on any platform
- **THEN** the trainer writes new weights to a file the renderer hot-reloads through the
  existing loader, with no CUDA dependency

#### Scenario: Interop backend guarded where no GPU path exists
- **WHEN** `--neural-handoff interop` is selected on a machine with neither CUDA (with the
  required external-memory extension) nor a Metal unified-memory device
- **THEN** the backend reports it as unavailable with a clear message naming
  `--neural-handoff file` rather than silently degrading or crashing

#### Scenario: Interop backend shares GPU memory on CUDA
- **WHEN** `--neural-handoff interop` is selected on a CUDA machine with external memory
  support
- **THEN** CUDA writes updated weights into the Vulkan-exported weight buffer with no CPU
  round-trip

#### Scenario: Interop backend publishes via shared storage on Metal
- **WHEN** `--neural-handoff interop` is selected on the native Metal backend on a
  unified-memory device and the trainer publishes new weights
- **THEN** the staged weights and biases land in the renderer's shared-storage weight
  buffers at the frame-boundary swap, with no file written and no NFW1 round-trip,
  and the network version increments as for any publisher

#### Scenario: Metal interop publish is precision-faithful
- **WHEN** the same weights are published once through the file backend and once through
  the Metal interop backend at the same `NeuralPrecision`
- **THEN** the resulting weight- and bias-buffer contents are byte-identical
