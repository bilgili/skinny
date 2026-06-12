## MODIFIED Requirements

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

## ADDED Requirements

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
