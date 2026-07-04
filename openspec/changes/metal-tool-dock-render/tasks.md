# Tasks — Metal render paths for the tool docks

## 1. P1 — Metal material preview (small)
- [ ] 1.1 `PreviewPipelineMetal` in `metal_compute.py`: compile `preview_pass.slang`
      to MSL via the megakernel SlangPy session (reuse its `generated_materials`
      link); `dispatch(push_bytes, set0_binds, output_image)` = root object +
      bind-by-name (set-0 material resources + output image) + `set_data(push)` +
      one compute pass over `size×size` + wait idle. No descriptor sets, no barriers.
- [ ] 1.2 Metal branch in `Renderer.render_material_preview`: `if self.ctx.is_metal`
      → build the set-0 bind dict (reuse `_build_metal_binds`, filtered), pack the
      same 32-byte push (`material_id, graph_id, prim_kind, size, yaw, pitch,
      distance, fov_tan`), dispatch `PreviewPipelineMetal`, read back the RGBA32F.
      Else the existing Vulkan path, byte-unchanged.
- [ ] 1.3 Allocate the Metal preview output image + readback (Metal `StorageImage`
      + `ReadbackBuffer`) lazily on first Metal preview; add a cleanup hook.
- [ ] 1.4 Verify: headless Metal render test — scene with a std_surface/MaterialX
      material, `render_material_preview(mid, 0, 128)` returns `(bytes, 128)` with
      non-degenerate lit pixels (not all-zero/NaN); Vulkan preview numbers
      unchanged. Then `skinny-gui --backend metal` → Material Graph preview panel
      shows the material.

## 2. Interim — Camera Debug Metal notice — DONE (883ef3e)
- [x] 2.1 On Metal (`proxy._backend_name == "metal"`), both gap docks show a
      "renders on the Vulkan backend only" notice and skip the render attempt
      (Camera Debug: `_DebugCanvas.set_notice` + showEvent skips create; Material
      Graph: `_render_preview` early notice). Stops the per-frame `[render error]`.
      Verified ruff + 12 guards + offscreen (metal→notice+no-post, vulkan→normal).
      Material Graph notice removed by P1 (task 1.2); Camera Debug notice removed
      by P2 (task 3.9).

## 3. P2 — Metal Camera Debug compute rasteriser (XL, phased)
- [ ] 3.1 `debug_raster.slang`: line rasteriser (DDA/Bresenham) — transform
      `float3 pos + float4 color` vertices by the view·proj UBO, scan-convert to the
      RGBA8 output; no depth yet. `DebugRasterMetal` host in `metal_compute.py`
      (upload vertex buffer, dispatch, readback).
- [ ] 3.2 Triangle rasteriser (edge-function + barycentric) into the same output.
- [ ] 3.3 Depth buffer: `uint` UAV `w×h`, `atomic_min` packed depth; opaque
      lines/tris depth-test-and-write.
- [ ] 3.4 Alpha blend pass for transparent planes (`color.a < 1`), depth-test-no-
      write, ordered after opaque (mirrors the Vulkan two-pipeline split).
- [ ] 3.5 Screen-space HUD text (`color.a > 1.5` sentinel bypasses view·proj).
- [ ] 3.6 Numpy/host parity harness for the rasteriser (transform + a few
      line/tri/blend cases) so the kernel is checkable without a GPU.
- [ ] 3.7 `DebugViewport` Metal branch: where the guard raises, construct the Metal
      rasteriser instead; `render_embedded(renderer)` fills the (unchanged) CPU
      vertex streams, uploads, dispatches, returns RGBA8 — same signature the worker
      `_maybe_render_debug` already calls. Vulkan graphics path untouched.
- [ ] 3.8 Verify: headless Metal — `DebugViewport` embedded render returns a
      non-empty RGBA8 frame with the grid/frustum visible; then `skinny-gui
      --backend metal` → Camera Debug shows the live debug scene, orbit/pan/zoom +
      overlays work.
- [ ] 3.9 Remove the interim Metal notice (task 2.1); Camera Debug renders on both
      backends.

## 4. Docs + close-out
- [ ] 4.1 Update the CLAUDE.md / README Compatibility matrix: material preview +
      Camera Debug now ✅ on Metal (was Vulkan-only); update `docs/Architecture.md`
      Metal sections.
- [ ] 4.2 `ruff check src/` clean; `openspec validate metal-tool-dock-render
      --strict` passes; archive on completion.
