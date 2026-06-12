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
`supports_external_semaphore` SHALL report `false` (frozen neural weights load by
buffer upload, not GPU↔GPU interop). The flags `supports_fp16_storage` and
`supports_fp16_compute` SHALL report the real Metal device's fp16 support, so the
renderer uses fp16 neural storage/inference where available and falls back to fp32
on a device that lacks it.

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

## ADDED Requirements

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
