# metal-backend Specification

## Purpose

Provide a native Metal rendering backend, built on SlangPy's `DeviceType.metal`,
that mirrors the context surface the renderer consumes from Vulkan so the
renderer drives it duck-typed, and resolve the active backend through a single
shared resolver used by every front-end (`--backend {auto,metal,vulkan}` with a
`SKINNY_BACKEND` fallback and capability gating). This foundation phase covers
device bring-up, a trivial compute dispatch, present, and backend selection;
full render parity arrives in later changes, so `auto` resolves to Vulkan and an
explicit `--backend metal` refuses the full renderer with a clear message.

## Requirements

### Requirement: Native Metal device backend foundation

The system SHALL provide a native Metal rendering backend built on SlangPy's
`DeviceType.metal`, exposed through a `MetalContext` that mirrors the surface the
renderer consumes from the Vulkan context — dimensions, compute and present
queues, a swapchain-equivalent, command-buffer allocation, capability flags,
`recreate_swapchain`, and `destroy` — so the renderer drives it duck-typed
without Metal-specific knowledge at the context layer. The Metal backend SHALL be
import- and platform-guarded so non-Apple-Silicon hosts never construct it. This
requirement covers device bring-up, a trivial compute dispatch, and present only;
full render parity (the megakernel head render, materials, ReSTIR, neural,
wavefront) is delivered by later changes.

On the Metal backend in this foundation phase, the capability flags
`supports_external_memory`, `supports_external_semaphore`,
`supports_fp16_storage`, and `supports_fp16_compute` SHALL report `false`, so the
renderer stays on its fp32 and file-handoff paths until the corresponding Metal
equivalents are added.

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

### Requirement: Backend selection with capability gating and fallback

The system SHALL resolve the active rendering backend from a single shared
resolver used by every front-end, selected by `--backend {auto,metal,vulkan}`
with a `SKINNY_BACKEND` environment fallback and the interactive front-ends'
persisted setting. Precedence SHALL be explicit flag, then environment, then
persisted value, then `auto`. An explicitly requested backend that cannot be
constructed SHALL fail with a clear message naming the missing requirement rather
than silently degrading. The `vulkan` selection SHALL remain byte-identical to
the current behavior on every platform (MoltenVK under Vulkan on macOS
unchanged).

In this foundation phase (P1) the renderer is not yet ported to Metal, so `auto`
SHALL resolve to Vulkan on every host, and a front-end that resolves to `metal`
(an explicit `--backend metal` on an Apple-Silicon host where the Metal device
constructs) SHALL refuse to launch the full renderer with a clear message stating
that the Metal device foundation is built but full rendering lands in a later
phase, directing the user to `--backend vulkan`. The native-Metal device,
trivial compute dispatch, and present foundation are exercised through the shared
resolver's `make_context` (tests and the present smoke), not the full renderer.
The `auto`→Metal selection for the full renderer arrives with megakernel render
parity in a later change.

#### Scenario: Auto resolves to Vulkan in the foundation phase

- **WHEN** the application is launched with `--backend auto` (or no `--backend`)
  on any host, including an Apple-Silicon macOS host where the Metal device
  constructs
- **THEN** the active backend resolves to Vulkan (the renderer is not yet ported
  to Metal in this phase), and the renderer behaves byte-identically to before
  this change

#### Scenario: Explicit Metal on a real front-end reports the foundation phase

- **WHEN** `--backend metal` is requested on an Apple-Silicon host where the
  Metal device constructs, on one of the real front-ends
- **THEN** the front-end refuses to launch the full renderer with a clear message
  that the Metal device foundation is built but full rendering lands in a later
  phase, directing the user to `--backend vulkan`

#### Scenario: Explicit unavailable backend fails clearly

- **WHEN** `--backend metal` is requested on a host without a constructible Metal
  device (non-Apple-Silicon, or the Metal toolchain/device unavailable)
- **THEN** selection raises a clear error naming the missing requirement rather
  than silently falling back to Vulkan

#### Scenario: Vulkan selection is unchanged

- **WHEN** `--backend vulkan` (or `SKINNY_BACKEND=vulkan`) is selected on any
  platform
- **THEN** the renderer constructs a `VulkanContext` and behaves byte-identically
  to the renderer before this change

#### Scenario: Same backend flag on every front-end

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-web`, `skinny-render` is run
  with `--help`
- **THEN** the `--backend {auto,metal,vulkan}` flag is present with identical
  choices and default, defined from the single shared source
