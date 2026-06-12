# metal-backend — Delta (metal-neural-interop)

## MODIFIED Requirements

### Requirement: Native Metal device backend foundation

The system SHALL provide a native Metal rendering backend built on SlangPy's
`DeviceType.metal`, exposed through a `MetalContext` that mirrors the surface the
renderer consumes from the Vulkan context — dimensions, compute and present
queues, a swapchain-equivalent, command-buffer allocation, capability flags,
`recreate_swapchain`, and `destroy` — so the renderer drives it duck-typed
without Metal-specific knowledge at the context layer. The Metal backend SHALL be
import- and platform-guarded so non-Apple-Silicon hosts never construct it. The
device foundation supports device bring-up, compute dispatch, and present; the
megakernel head render on this device is delivered by the megakernel render-parity
requirement. Wavefront execution, ReSTIR DI reuse, and neural directional-proposal
inference on Metal are delivered by the wavefront-parity requirement below and the
corresponding `wavefront-execution`, `restir-di`, and `neural-directional-proposal`
deltas; they are no longer deferred.

On the Metal backend the capability flags `supports_external_memory` and
`supports_external_semaphore` SHALL report `false` (Metal exposes no exported
memory or semaphore handles). The context SHALL additionally report a
`supports_shared_memory` capability flag that is `true` on unified-memory Metal
devices and `false` otherwise; consumers on other backends read the flag as
`false` by default. The Metal storage-buffer wrapper SHALL support a
shared-storage allocation mode whose contents the host can update in place
(without a staging round-trip) for host→GPU weight handoff — this is the
unified-memory interop path used by the `neural-online-training` Metal interop
publisher. The flags `supports_fp16_storage` and `supports_fp16_compute` SHALL
report the real Metal device's fp16 support, so the renderer uses fp16 neural
storage/inference where available and falls back to fp32 on a device that lacks
it.

#### Scenario: Metal context constructs windowed and headless

- **WHEN** a `MetalContext` is constructed on an Apple-Silicon macOS host, both
  with a GLFW window and headless (`window=None`)
- **THEN** it builds a native Metal device via SlangPy `DeviceType.metal` and
  exposes the Vulkan-compatible context surface, without falling back to MoltenVK

#### Scenario: Trivial Metal dispatch matches the Vulkan backend

- **WHEN** the same trivial Slang compute kernel (e.g. write the global invocation
  id into a storage buffer) is dispatched on the Metal backend and on the Vulkan
  backend
- **THEN** the two output buffers are identical

#### Scenario: Metal present clears and presents without hanging

- **WHEN** a windowed `MetalContext` clears and presents several frames
- **THEN** every frame's GPU fence is signalled and the frames present, with no
  indefinite fence wait (pipeline parameters are set via `set_data` byte blobs,
  never per-field cursor writes)

#### Scenario: fp16 capability flags reflect the device

- **WHEN** a `MetalContext` is constructed on a device that supports half-precision
  storage and compute
- **THEN** `supports_fp16_storage` and `supports_fp16_compute` report `true`, and on
  a device without fp16 support they report `false` and the renderer falls back to
  fp32 neural state

#### Scenario: Shared-memory capability flag reflects unified memory

- **WHEN** a `MetalContext` is constructed on an Apple-Silicon unified-memory device
- **THEN** `supports_shared_memory` reports `true` while `supports_external_memory`
  and `supports_external_semaphore` still report `false`

#### Scenario: Shared-storage buffer accepts in-place host writes

- **WHEN** a Metal storage buffer is allocated in shared-storage mode, the host
  writes new contents in place, and a compute dispatch then reads the buffer
- **THEN** the dispatch observes the written bytes without any explicit staging
  upload call, and a buffer allocated without shared mode behaves as before
