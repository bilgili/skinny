# Tasks

All tasks are **complete** — shipped to `main` (commits `1568d01`, `586d104`,
`4f77e1e`, `d9bf0b9`, `74d5ad6`, `b8f6dfe`). This change formalises the spec
delta retroactively.

## 1. Duck-typed context surface

- [x] 1.1 Add `MetalGpuInfo` + `MetalContext.gpu_info` (`name` from `device.info.adapter_name`, `is_discrete=False`, `preferred_h264_encoder="h264_videotoolbox"`); keep it local so the Metal path never imports `skinny.hardware`/`vulkan`
- [x] 1.2 Map the `rgba8_srgb` neutral format token (and VkFormat 43) to the slang-rhi `rgba8_unorm_srgb` enum in `metal_compute._FORMAT_TOKENS`/`_VKFORMAT_INTS`
- [x] 1.3 Update `MetalContext` docstring + `docs/Architecture.md` duck-typed surface list to include `gpu_info`

## 2. Per-graph MaterialX param SSBOs on Metal

- [x] 2.1 Reflect each `graphParams_<sanitized>` element's MSL `{field: (offset, size)}` + stride into `ComputePipeline.graph_param_layouts` (`metal_compute._reflect_graph_param_layouts`, called from `_build`)
- [x] 2.2 Bind every active graph's param buffer by name in `renderer._build_metal_binds` (`graphParams_<sanitized>` → buffer)
- [x] 2.3 In `_upload_graph_param_buffers`, on Metal relocate each scalar-packed field into its reflected MSL offset and size the buffer at the MSL stride (fallback to scalar when reflection is unavailable)

## 3. Vulkan descriptor helpers no-op on Metal

- [x] 3.1 `is_metal` early return in `_update_texture_pool_descriptors`
- [x] 3.2 `is_metal` early return in `_rebind_scene_descriptors`, `_rebind_aux_material_descriptors`, `_rebind_mesh_descriptors`

## 4. Vulkan-only GUI tools degrade on Metal

- [x] 4.1 `DebugViewport.open()` raises a clear "requires the Vulkan backend" `RuntimeError` on a Metal context and stays closed
- [x] 4.2 `request_bxdf_eval` / `request_bssrdf_eval` return an empty grid on Metal (route through the existing `pipeline is None` zeroed-grid fallback)

## 5. Tests & verification

- [x] 5.1 Foundation guards in `tests/test_metal_foundation.py`: gpu_info shape, format-token→`slangpy.Format` resolution, texture-pool + rebind no-ops, debug-viewport refusal, BXDF degrade
- [x] 5.2 New gated `tests/test_metal_procedural_graph_parity.py` — converged Metal↔Vulkan A/B on a multi-material USD scene; whole-frame rel-MSE + per-material "not near-black on one backend" guard
- [x] 5.3 Headless verification: marble exact match; overall path/BDPT parity rel-MSE ≈0.0004; 20+-instance scene renders without crashing

## 6. Follow-ups (out of scope, tracked separately)

- [ ] 6.1 Graph-internal `SamplerTexture2D` parity (wood ~10% bright on Metal — texture sampled through a sampler field inside the graph struct)
- [ ] 6.2 Native Metal tool-dispatch siblings for the debug viewport and BXDF/BSSRDF visualiser
