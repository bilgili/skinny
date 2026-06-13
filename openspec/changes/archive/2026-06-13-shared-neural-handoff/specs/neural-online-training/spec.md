## RENAMED Requirements

- FROM: `### Requirement: Two selectable weight-handoff backends`
- TO: `### Requirement: Selectable weight-handoff backends`

## MODIFIED Requirements

### Requirement: Selectable weight-handoff backends
The system SHALL provide three weight-handoff backends behind a common publisher
interface selectable at runtime via `--neural-handoff` (`file` | `interop` |
`shared`): a file double-buffer backend, a GPU-shared-memory interop backend, and
an in-process shared CPU double-buffer backend. The file backend SHALL work
without CUDA on any platform. The interop backend SHALL resolve per GPU backend:
on Vulkan it SHALL use the Vulkan↔CUDA external-memory path, and on the native
Metal backend it SHALL use a unified-memory shared-storage path that writes
published weights into the renderer's weight and bias buffers with no disk
round-trip and no NFW1 serialize/parse. The Metal interop publish SHALL produce
byte-identical buffer contents to a file-backend publish of the same weights at
the same `NeuralPrecision` (fp32, fp16 storage, fp8 e4m3). The interop backend
SHALL be guarded so it fails clearly, with a message naming the file fallback, on
hosts that have neither CUDA with the required Vulkan external-memory extension
nor a Metal unified-memory device.

The `shared` backend SHALL hold the staged weights in process memory and hand
them across the trainer→render boundary with no disk write and no GPU-interop
requirement, and SHALL be available on every platform without CUDA, a Metal
unified-memory device, or any added dependency. Its `publish` SHALL store a copy
of the supplied weights private to the publisher so the trainer MAY keep mutating
its own working weights without affecting the frozen render-side buffer; the bytes
the renderer consumes from a `shared` publish SHALL be identical to those from a
`file` publish of the same weights. The `shared` backend SHALL NOT write the GPU
weight/bias buffers directly; the renderer SHALL upload swapped weights to the GPU
through the same post-swap path it uses for the `file` backend. All three backends
SHALL honor the same frame-boundary swap and network-version-increment contract.

#### Scenario: File backend publishes without CUDA
- **WHEN** `--neural-handoff file` is selected on any platform
- **THEN** the trainer writes new weights to a file the renderer hot-reloads through the
  existing loader, with no CUDA dependency

#### Scenario: Shared backend publishes in process without disk or GPU interop
- **WHEN** `--neural-handoff shared` is selected on any platform and the trainer publishes
  new weights
- **THEN** the weights are staged in process memory with no file written, no NFW1
  serialize/parse to disk, and no CUDA or unified-memory device required, and the swap at
  the frame boundary increments the network version as for any publisher

#### Scenario: Shared backend freezes the render buffer against trainer mutation
- **WHEN** the trainer publishes weights through the `shared` backend and then mutates its
  own working weights in place before the next swap
- **THEN** the weights returned by `acquire_for_render` are unaffected by the post-publish
  mutation until a swap promotes a subsequently published version

#### Scenario: Shared publish is byte-faithful to a file publish
- **WHEN** the same weights are published once through the file backend and once through
  the shared backend
- **THEN** the weight, bias, and header contents the renderer would consume are identical

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
