## Why

The native Metal megakernel reached parity with Vulkan only for skin and simple
textured materials — the parity tests render `head.obj`, which has no MaterialX
graph materials. Real USD scenes broke on Metal: a purely-procedural graph
material (marble `fractal3d`) rendered **black**, and several Vulkan-only host
paths (descriptor rebinds on buffer growth, GUI visualiser tools) crashed the
render thread because they dereference attributes the compute-only Metal context
does not expose. This change closes the megakernel material-parity gap and makes
every Vulkan-only host path degrade safely on Metal.

## What Changes

- **Per-graph MaterialX parameter SSBOs work on Metal.** Each generated
  `GraphParams_*` buffer is now (a) bound by name on the megakernel dispatch
  (`_build_metal_binds` previously omitted them, so the shader read a zeroed
  buffer) and (b) relocated from scalar/std430 layout into the reflected MSL
  element layout (Slang pads `float3` to 16 B on Metal, shifting every field
  after a leading `float3`). Procedural graph materials (marble) and
  graph-driven std_surface materials now match Vulkan instead of rendering black.
- **Vulkan descriptor helpers no-op on Metal.** `_update_texture_pool_descriptors`
  and the scene/aux/mesh descriptor rebinds short-circuit on Metal (it rebinds
  live buffers by name at dispatch), instead of crashing on `descriptor_sets is
  None` / `vk.VkDescriptorBufferInfo` over a slangpy buffer.
- **Vulkan-only GUI tools degrade gracefully on Metal.** The debug viewport (a
  Vulkan graphics rasteriser) refuses to open with a clear message; the
  BXDF/BSSRDF visualiser returns an empty grid rather than dereferencing
  `ctx.command_pool`.
- **Duck-typed surface completeness.** `MetalContext` exposes `gpu_info`
  (`name` / `is_discrete` / `preferred_h264_encoder`) read by the front-ends and
  the video encoder; the `rgba8_srgb` neutral format token resolves to the
  slang-rhi `rgba8_unorm_srgb` enum.
- A gated Metal↔Vulkan **procedural-graph parity test** drives a multi-material
  USD scene (the head-only parity tests could never catch this).

No user-facing CLI/flag changes. No **BREAKING** changes — Metal-only behavior
fixes; the Vulkan path is byte-unchanged.

## Capabilities

### New Capabilities
<!-- none — this extends the existing metal-backend capability -->

### Modified Capabilities
- `metal-backend`: the **Megakernel render parity on Metal** requirement is
  extended to cover flat/std_surface and procedural MaterialX-graph materials
  (per-graph param SSBO bind + MSL-correct layout), and a new requirement is
  added that **every Vulkan-only host path degrades safely on the compute-only
  Metal context** (descriptor rebinds no-op; GUI visualiser tools and the debug
  viewport degrade without crashing).

## Impact

- Code: `src/skinny/renderer.py` (graph-param bind + MSL relocation, descriptor-
  rebind guards, BXDF degrade), `src/skinny/metal_compute.py`
  (`graph_param_layouts` reflection), `src/skinny/metal_context.py` (`gpu_info`,
  `MetalGpuInfo`), `src/skinny/debug_viewport.py` (Metal refusal),
  `docs/Architecture.md` (duck-typed surface + binding notes).
- Tests: `tests/test_metal_foundation.py` (gpu_info, format tokens, descriptor
  no-ops, debug-viewport refusal, BXDF degrade) and a new
  `tests/test_metal_procedural_graph_parity.py` (gated converged A/B).
- No dependency or API changes; the Vulkan backend and descriptor binding map
  are unaffected.
