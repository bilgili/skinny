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

Parity SHALL hold for **flat / standard_surface materials and procedural
MaterialX-graph materials**, not only skin and simple textured materials. Each
generated per-graph parameter buffer (`StructuredBuffer<GraphParams_*>`) SHALL be
bound by name on the Metal megakernel dispatch, and its scalar/std430-packed
record SHALL be relocated into the reflected MSL element layout (Slang pads
`float3` to 16 B on Metal, shifting every field after a leading `float3`) at the
MSL element stride, using offsets taken from live pipeline reflection. A
purely-procedural graph material whose colour comes entirely from its parameter
SSBO (e.g. a `fractal3d` marble) SHALL render the same colour on Metal as on
Vulkan and SHALL NOT render black.

The rendered output on Metal SHALL match the Vulkan megakernel at **structural
parity**: with rays held identical across backends (matched `frameIndex` AA
jitter), the deterministic structural outputs (geometry/instance + material
identifiers, the hit/miss mask, and depth) SHALL agree up to pure floating-point
cross-compiler divergence — hit-count parity and hit-mask agreement bounded to a
thin silhouette edge band, identifiers exact on commonly-hit pixels, and depth
within a robust tolerance. Bit-identical structural outputs across the two shader
compilers (Slang→Metal vs Slang→SPIR-V) are NOT required — sub-ULP ray
differences flip near-tangent hit/miss and perturb grazing-ray depth. The
converged shaded color SHALL match within a stated perceptual tolerance; byte-
identical shaded color across backends is NOT required.

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

#### Scenario: Structural outputs match Vulkan

- **WHEN** the same fixed scene is rendered headless through the megakernel on the
  Metal backend and on the Vulkan backend, with rays held identical (matched
  `frameIndex` AA jitter)
- **THEN** geometry is traced on both backends; the hit counts agree (hit-count
  parity) and the hit/miss masks agree on the bulk of hits (residual differences
  confined to a thin silhouette edge band); the geometry/instance + material IDs
  are exact on commonly-hit pixels; and depth agrees within a robust tolerance
  (median and 95th-percentile relative bounds) — the residual divergence being
  pure FP cross-compiler codegen, not a geometry mismatch

#### Scenario: Shaded color matches Vulkan within tolerance

- **WHEN** the same fixed scene is rendered to convergence through the megakernel
  on the Metal backend and on the Vulkan backend
- **THEN** the converged shaded color (linear-HDR accumulation image) matches
  within the stated perceptual tolerance, with no requirement of byte-identical
  color

#### Scenario: Procedural MaterialX-graph material matches Vulkan

- **WHEN** a multi-material USD scene containing a purely-procedural graph
  material (colour driven entirely by `GraphParams_*` parameters, e.g. a
  `fractal3d` marble) is rendered to convergence through the megakernel on both
  the Metal and Vulkan backends
- **THEN** the per-graph parameter buffer is bound and read with MSL-correct
  field offsets on Metal, the material's converged colour matches Vulkan within
  the perceptual tolerance, and no material that is lit on Vulkan is near-black
  on Metal

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

### Requirement: Vulkan-only host paths degrade safely on Metal

Every renderer or front-end path implemented against the Vulkan context SHALL either run a Metal-equivalent path or short-circuit on `is_metal`, and SHALL NOT crash by dereferencing a Vulkan-only attribute on the compute-only `MetalContext`.

The compute-only `MetalContext` does not expose the Vulkan-specific context
surface (`command_pool`, `instance`, `physical_device`, `queue_family_indices`,
external-handle types) and has no Vulkan descriptor sets (`descriptor_sets is
None`); it binds every megakernel resource by name fresh at each dispatch.

Specifically: the Vulkan descriptor-set rewrite helpers (texture-pool descriptor
update; scene, auxiliary-material, and mesh descriptor rebinds invoked when a
buffer reallocates as a scene streams) SHALL be no-ops on Metal, since the next
dispatch rebinds the live buffers by name. The renderer's duck-typed context
surface SHALL expose `gpu_info` (carrying at least `name`, `is_discrete`, and
`preferred_h264_encoder`) on both backends so the front-ends and the video
encoder stay backend-agnostic. Backend-neutral image-format tokens (including the
`rgba8_srgb` token) SHALL resolve to a valid `slangpy.Format` on Metal.

Vulkan-only GUI tools that have no Metal implementation yet SHALL degrade
visibly rather than crash: the debug viewport (a Vulkan graphics rasteriser)
SHALL refuse to open on Metal with a clear message and stay closed, and the
BXDF/BSSRDF visualiser SHALL return an empty result grid on Metal.

#### Scenario: Streaming a many-instance scene does not crash on Metal

- **WHEN** a USD scene whose instance/material/mesh buffers grow and reallocate
  as it streams (e.g. a 20+-instance scene) is loaded on the Metal megakernel
  backend
- **THEN** the descriptor-rebind helpers short-circuit on Metal and the scene
  renders a full frame, with no `TypeError`/`AttributeError` from a Vulkan
  descriptor call on a slangpy buffer

#### Scenario: Front-end reads gpu_info on Metal

- **WHEN** a front-end or the video encoder reads `ctx.gpu_info.name` /
  `ctx.gpu_info.preferred_h264_encoder` on a `MetalContext`
- **THEN** the attributes resolve (e.g. the adapter name and a VideoToolbox
  encoder) with no missing-attribute error

#### Scenario: Vulkan-only GUI tool degrades on Metal

- **WHEN** the debug viewport or the BXDF/BSSRDF visualiser is invoked on the
  Metal backend
- **THEN** the debug viewport raises a clear "requires the Vulkan backend"
  message and stays closed, and the visualiser returns an empty grid — neither
  crashes the application or the render thread

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

