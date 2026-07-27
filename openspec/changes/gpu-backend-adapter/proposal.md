# Change: gpu-backend-adapter

## Why

`backend_select.resource_module` documents that `vk_compute` and
`metal_compute` "expose the same public API". They do not, and the renderer
pays for the gap with 40 live backend branches.

Measured asymmetries between the two modules:

- **5 classes exist on one side only.** `PreviewPipeline` (vk) and
  `PreviewPipelineMetal` (metal) are the same concept under two names, so
  `self._gpu.PreviewPipeline` is impossible — `renderer.py:7115` branches and
  imports each by name. `ExternalTimelineSemaphore`, `DebugRasterMetal` and
  `MetalFrameEncoder` have no counterpart.
- **Zero module-level function overlap.** Two of the Metal-only ones are
  nevertheless consumed through the "backend-agnostic" handle:
  `renderer.py:1482` calls `self._gpu._make_sampler(...)` (private,
  Metal-only, guarded by an `is_metal` on the next line) and
  `renderer.py:9970` imports `_rgba_f32_to_rgba8` from `metal_compute`
  directly.
- **`BINDLESS_TEXTURE_CAPACITY` is 128 in `vk_compute` and 119 in
  `metal_compute`** — same name, different number, with a hand-maintained
  "MUST equal metal_compute.BINDLESS_TEXTURE_CAPACITY" comment in
  `shaders/bindings.slang`.
- **`SampledImage.address_mode_u` takes a `VkEnum` int on Vulkan and a string
  on Metal**; `metal_compute` carries `_VKFORMAT_INTS` / `_VK_ADDRESS_INTS`
  translation tables to absorb callers that pass the wrong domain.
  `upload_sync`'s parameter is named `rgba_f32` on one and `data` on the
  other, so any keyword call breaks on one backend.
- **`ComputePipeline.dispatch()` exists only on Metal.** On Vulkan the
  renderer records `vk.vkCmdDispatch` inline — 106 raw `vk.vk*` calls in
  `renderer.py`. "Dispatch a frame" is not on the shared surface at all.

The renderer therefore carries **40 live `is_metal` branches** and **15
Metal-only methods** with no Vulkan twin, plus two broken backend probes used
as if they were the seam:

- `descriptor_sets is None` is used as an "is Metal" sentinel at 12 gates
  (5 of them compound with `is_metal` in the same expression — the same fact
  stated twice).
- `hasattr(ctx, "compute_queue")` is used as "is Vulkan" at 7 sites, but
  `MetalContext.compute_queue = None` (`metal_context.py:300`), so the
  attribute exists and the test is **always true**. `renderer.py:1775`
  (`hasattr(...) or self.is_metal`) is unconditionally `True`; three of the
  four `vk_wavefront` factories rely on the caller never routing Metal to
  them.

Two adapters make the seam real. The interface at that seam is nominal, so
divergence is checked only by two statistical parity tests that need a Metal
device and a Vulkan device on the same machine.

## What Changes

- Declare the interface the two adapters actually have to satisfy: resource
  construction, binding, dispatch, readback, and the capability facts the
  renderer branches on today (does this target have descriptor sets, external
  memory, indirect dispatch, in-place shared writes). Name each concept once —
  `PreviewPipeline`, not `PreviewPipeline`/`PreviewPipelineMetal`.
- Replace the two broken probes: `hasattr(ctx, "compute_queue")` and
  `descriptor_sets is None` become named capability reads on the interface.
  No consumer outside the adapters tests for a backend by attribute presence.
- Fold the per-backend constants that share a name but not a value
  (`BINDLESS_TEXTURE_CAPACITY`) into a capability the shader define and the
  host both read, so `bindings.slang`'s hand-maintained comment becomes a
  test.
- Unify argument domains: one address-mode / format vocabulary, one parameter
  name per method. Delete the `_VKFORMAT_INTS` / `_VK_ADDRESS_INTS`
  translation tables once no caller passes Vulkan ints.
- Add a **third adapter** that records calls instead of executing them, so
  dispatch ordering, binding coverage and pass sequencing become hostlessly
  assertable on any machine.
- Add a hostless conformance test run against every adapter: same interface,
  same method names, same argument domains, with each intentional
  one-sided member declared in an explicit table rather than discovered.

## Capabilities

### New Capabilities

- `gpu-backend-adapter`: the interface at the Vulkan/Metal seam — resource,
  dispatch and capability contract; one name per concept; a recording adapter
  for hostless tests; and a conformance test that fails when the adapters
  drift.

### Modified Capabilities

- `metal-backend`: "Vulkan-only host paths degrade safely on Metal" is
  restated in terms of declared capabilities rather than attribute-presence
  probes and `descriptor_sets is None` sentinels.

## Impact

- New: `src/skinny/gpu_backend.py` (interface + capability record + recording
  adapter), conformance test module.
- Modified: `src/skinny/vk_compute.py`, `src/skinny/metal_compute.py`,
  `src/skinny/vk_context.py`, `src/skinny/metal_context.py`,
  `src/skinny/backend_select.py`, `src/skinny/renderer.py` (40 branches),
  `src/skinny/vk_wavefront.py` (4 probes), `src/skinny/debug_viewport.py`
  (5 sites), `src/skinny/mlt_chain.py` (the `is_metal` parameter).
- Unchanged: shaders (except the `bindings.slang` capacity comment becoming a
  tested fact), descriptor binding numbers, dispatch behaviour, CLI surface.
- **Sequencing**: land `renderer-gpu-resource-set` first — it removes ~10 of
  the 40 branches and gives this interface its first consumer. This change is
  the largest in the set and should be split into landable stages (see design).
