# Metal Neural Interop

## Why

On Vulkan+CUDA hosts the neural directional proposal gets trained weights via the
zero-copy `InteropWeightPublisher` (CUDA writes straight into the Vulkan-exported
weight buffer). On the native Metal backend there is no interop path at all:
`--neural-handoff interop` is unavailable, so online training falls back to the
`file` publisher — every weight publish serializes NFW1 to disk, reloads on the
render side, and re-uploads through a staging copy. On Apple Silicon this is pure
waste: the GPU is a unified-memory device, the torch trainer already runs on MPS,
and the renderer's weight buffer can be host-visible shared storage. Closing this
gap makes online neural guiding on Metal publish weights with no disk round-trip
and no staging upload, matching the Vulkan+CUDA experience.

## What Changes

- New `MetalSharedWeightPublisher` (UMA interop): the trainer publishes weights
  directly into the renderer's Metal weight buffer through shared-storage memory
  mapping, with frame-boundary synchronization so the renderer never samples a
  half-written network — the Metal sibling of `InteropWeightPublisher`.
- `metal_compute.StorageBuffer` gains a shared-storage allocation mode and a
  host-pointer/native-handle accessor so the publisher can write in place
  (today `external=False` and `export_handle()` returns `None` unconditionally).
- `MetalContext` reports a new `supports_shared_memory` capability flag (true on
  Apple-Silicon UMA devices); `supports_external_memory` stays `false` — Metal
  interop is UMA-shared, not exported-handle.
- `--neural-handoff interop` resolves to the Metal publisher on the Metal backend
  instead of erroring; the existing clear-failure guard remains for hosts with
  neither CUDA nor Metal UMA. `--neural-handoff auto` (if/where defaulted) picks
  interop on Metal UMA.
- Weight precision parity: shared publishes honor `NeuralPrecision`
  (fp32/fp16-storage/fp8-e4m3) identically to the file path — same bytes land in
  the buffer either way.
- Tests: publisher unit tests (precision round-trip, torn-write guard) plus a
  GPU-marked parity test that trains a few steps and asserts interop-published
  and file-published renders match.
- Docs: `docs/NeuralGuiding.md` handoff section, `docs/Architecture.md`
  capability-flag table, `README.md` flag docs.

Out of scope (possible follow-up): zero-copy path-record drain on Metal (trainer
reading the record buffer via shared storage instead of `download_sync`); true
cross-device MTLBuffer sharing with torch's internal MTLDevice.

## Capabilities

### New Capabilities

(none — this extends existing capabilities)

### Modified Capabilities

- `neural-online-training`: the weight-handoff requirement gains a Metal UMA
  interop backend — `--neural-handoff interop` SHALL work on Metal Apple-Silicon
  hosts via shared-storage publish with no CPU/disk round-trip, with the
  unavailability guard narrowed to hosts that have neither CUDA nor Metal UMA.
- `metal-backend`: capability-flag requirement changes — Metal SHALL report a new
  `supports_shared_memory` flag (true on UMA devices) while keeping
  `supports_external_memory=false`; the Metal buffer wrapper SHALL support
  shared-storage allocation and in-place host writes for the neural weight
  buffer.

## Impact

- `src/skinny/metal_compute.py` — shared-storage `StorageBuffer` mode, native
  handle / host pointer accessor.
- `src/skinny/metal_context.py` — `supports_shared_memory` capability probe.
- `src/skinny/sampling/neural_handoff_interop.py` (or a new
  `neural_handoff_interop_metal.py`) — `MetalSharedWeightPublisher`
  (publish/swap/acquire against the shared buffer).
- `src/skinny/renderer.py` — publisher selection (`_neural_handoff_kind`) gains
  the Metal branch; weight-buffer allocation requests shared storage when the
  Metal interop publisher is active.
- `src/skinny/cli_common.py` — `--neural-handoff` help text / validation.
- `tests/` — new `test_neural_interop_metal.py`; existing
  `test_neural_interop.py` (CUDA) untouched.
- Docs: `docs/NeuralGuiding.md`, `docs/Architecture.md`, `README.md`,
  `CHANGELOG.md`.
- No shader changes; no descriptor-binding changes (bindings 33–35 unchanged).
