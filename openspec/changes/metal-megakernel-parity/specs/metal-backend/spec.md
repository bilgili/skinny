# metal-backend Specification (delta)

## ADDED Requirements

### Requirement: Megakernel render parity on Metal

The system SHALL render the head through the megakernel path (`main_pass.slang`,
entry `mainImage`) on the Metal backend, driving the same `Renderer` and the same
descriptor-binding map (bindings 0–24 and 30–32) used on Vulkan. The renderer
SHALL build its megakernel GPU resources (storage buffers, storage images,
sampled images, the uniform block, compute pipelines) on a `MetalContext` through
a resource layer (`metal_compute`) that mirrors the public API of the Vulkan
resource layer (`vk_compute`) — identical class names, constructor signatures, and
upload-helper names — so the renderer constructs resources without backend-specific
branches at the construction sites. The active resource layer SHALL be resolved
once from the context (keyed on `is_metal`).

The Metal megakernel pipeline SHALL compile `main_pass.slang` to Metal in-process
via SlangPy and bind buffers, textures, and samplers through resource binds; it
SHALL NOT perform per-field scalar `ShaderCursor` writes into uniform structs
(uniform data is uploaded via `set_data` byte blobs), preserving the foundation
phase's fence-hang discipline.

The rendered output on Metal SHALL match the Vulkan megakernel at **structural
parity**: exact equality on the deterministic structural outputs (geometry/instance
identifiers, the hit/miss mask, and depth), and equality within a stated perceptual
tolerance on the converged shaded color of a fixed scene. Byte-identical shaded
color across backends is NOT required.

The Metal capability flags (`supports_external_memory`,
`supports_external_semaphore`, `supports_fp16_storage`, `supports_fp16_compute`)
SHALL remain `false`; the megakernel render SHALL NOT depend on them. ReSTIR,
neural inference, and wavefront execution remain out of scope on Metal and SHALL
continue to fold back to the megakernel via the existing capability gates.

#### Scenario: Megakernel renders the head on Metal

- **WHEN** the renderer is launched on an Apple-Silicon host with a constructible
  Metal device and a scene loaded, in megakernel execution mode
- **THEN** the megakernel path compiles `main_pass.slang` (`mainImage`) on Metal,
  binds the full descriptor set, and produces a head render frame (no foundation
  refusal, no fence hang)

#### Scenario: Structural outputs match Vulkan exactly

- **WHEN** the same fixed scene is rendered headless through the megakernel on the
  Metal backend and on the Vulkan backend
- **THEN** the deterministic structural outputs (geometry/instance IDs, hit/miss
  mask, depth) are byte-identical between the two backends

#### Scenario: Shaded color matches Vulkan within tolerance

- **WHEN** the same fixed scene is rendered to convergence through the megakernel
  on the Metal backend and on the Vulkan backend
- **THEN** the converged shaded color (linear-HDR accumulation image) matches
  within the stated perceptual tolerance, with no requirement of byte-identical
  color

#### Scenario: Resource layer resolves by backend without per-site branches

- **WHEN** the renderer constructs its megakernel GPU resources on either a
  `MetalContext` or a `VulkanContext`
- **THEN** the resource classes resolve from a single backend-keyed resolution
  point, and no megakernel construction site contains a backend-specific
  conditional

### Requirement: MSL-correct uniform layout on Metal

The system SHALL upload the megakernel uniform block (`FrameConstants`,
`SkinParams`, `Light`) to the Metal pipeline using a layout that matches the
Metal Shading Language struct layout produced by Slang's Metal target, which
differs from the Vulkan `std140` layout (notably Slang pads `float3` to 16 bytes,
so the Metal struct is larger than the scalar/`std140` blob). The MSL field
offsets SHALL be derived from the compiled module's reflection rather than a
hand-maintained table, and the packed blob SHALL be uploaded via `set_data` only.
The Vulkan `std140` packing (`SkinParameters.pack()` and the Vulkan uniform path)
SHALL remain unchanged.

#### Scenario: Uniform blob matches the reflected MSL struct size

- **WHEN** the megakernel uniform block is packed for the Metal backend
- **THEN** the packed length equals the size of the MSL uniform struct reported by
  the compiled module's reflection, and field offsets are taken from that
  reflection

#### Scenario: Vulkan uniform packing is unchanged

- **WHEN** the megakernel uniform block is packed for the Vulkan backend
- **THEN** the `std140` layout from `SkinParameters.pack()` and the Vulkan uniform
  path are byte-identical to before this change

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
requirement. ReSTIR, neural inference, and wavefront execution on Metal remain
deferred to later changes.

On the Metal backend the capability flags `supports_external_memory`,
`supports_external_semaphore`, `supports_fp16_storage`, and
`supports_fp16_compute` SHALL report `false`, so the renderer stays on its fp32
and file-handoff paths until the corresponding Metal equivalents are added.

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

Now that the megakernel renders on Metal, `auto` SHALL resolve to Metal on an
Apple-Silicon macOS host where the Metal device constructs, and to Vulkan
otherwise. A front-end that resolves to `metal` SHALL launch the full renderer on
the Metal backend (the foundation-phase refusal notice is removed). On every
non-Apple-Silicon host and whenever the Metal device does not construct, `auto`
SHALL resolve to Vulkan.

#### Scenario: Auto resolves to Metal on Apple Silicon

- **WHEN** the application is launched with `--backend auto` (or no `--backend`)
  on an Apple-Silicon macOS host where the Metal device constructs
- **THEN** the active backend resolves to Metal and the full renderer runs on the
  Metal backend

#### Scenario: Auto resolves to Vulkan elsewhere

- **WHEN** the application is launched with `--backend auto` (or no `--backend`)
  on a host that is not Apple-Silicon macOS, or where the Metal device does not
  construct
- **THEN** the active backend resolves to Vulkan and the renderer behaves
  byte-identically to the Vulkan path

#### Scenario: Explicit Metal launches the renderer

- **WHEN** `--backend metal` is requested on an Apple-Silicon host where the
  Metal device constructs, on one of the real front-ends
- **THEN** the front-end launches the full renderer on the Metal backend with no
  foundation-phase refusal

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
