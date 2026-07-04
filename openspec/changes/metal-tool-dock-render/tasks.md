# Tasks — Metal render paths for the tool docks

## 1. P1 — Metal material preview (small)
- [x] 1.1 `PreviewPipelineMetal` in `metal_compute.py`: compile `preview_pass.slang`
      to MSL via a SlangPy session configured like the megakernel `ComputePipeline`
      (same include paths / `SKINNY_COMPUTE_PIPELINE`+`SKINNY_METAL` defines /
      `column_major`), linking the same emit-time `generated_materials` modules;
      `dispatch(size, push_bytes, binds, output_image, bindless)` = root object +
      bind-by-name (set-0 material resources + output image) + `set_data(fc/pc)` +
      one compute pass over `size×size` + wait idle. No descriptor sets, no barriers.
- [x] 1.2 Metal branch in `Renderer.render_material_preview`: `if self.is_metal`
      → `_render_material_preview_metal` builds the set-0 bind dict (reuse
      `_build_metal_binds`) + bindless pool, packs the same 32-byte push
      (`material_id, graph_id, prim_kind, size, yaw, pitch, distance, fov_tan`),
      dispatches `PreviewPipelineMetal`, reads back the RGBA32F float image
      directly (Metal `ReadbackBuffer` would down-convert to RGBA8). Else the
      existing Vulkan path, byte-unchanged. Removed the interim Material Graph
      Metal notice (`material_graph.py`). `preview_pass.slang`: `pc` bound as a
      plain `uniform` under `SKINNY_METAL` (slangpy rejects `set_data` on a
      push-constant `ConstantBuffer`); Vulkan SPIR-V byte-unchanged (`#else`).
- [x] 1.3 Allocate the Metal preview output image + readback (Metal `StorageImage`
      + `ReadbackBuffer`) lazily on first Metal preview; cleanup covered by the
      existing size-keyed teardown + the renderer cleanup hook.
- [x] 1.4 Verify: headless Metal render test (`tests/test_metal_material_preview.py`,
      gated `RUN_METAL_PREVIEW_COMPILE=1` under `scripts/guarded_metal.sh`) — demo
      scene material, `render_material_preview(mid, 0, 128)` returns `(bytes, 128)`
      with non-degenerate lit pixels (finite, not all-zero); size-rebuild path
      exercised. PASSED on Metal (rc=0, clean GPU exit); marble/wood/brass previews
      captured. Vulkan path logically unchanged (only guard order moved). Live
      `skinny-gui --backend metal` panel check remains a manual step.

## 2. Interim — Camera Debug Metal notice — DONE (883ef3e)
- [x] 2.1 On Metal (`proxy._backend_name == "metal"`), both gap docks show a
      "renders on the Vulkan backend only" notice and skip the render attempt
      (Camera Debug: `_DebugCanvas.set_notice` + showEvent skips create; Material
      Graph: `_render_preview` early notice). Stops the per-frame `[render error]`.
      Verified ruff + 12 guards + offscreen (metal→notice+no-post, vulkan→normal).
      Material Graph notice removed by P1 (task 1.2); Camera Debug notice removed
      by P2 (task 3.9).

## 3. P2 — Metal Camera Debug compute rasteriser (XL, phased)
- [x] 3.1 `debug_raster.slang`: line rasteriser (DDA) — transforms
      `float3 pos + float4 color` vertices (7 floats) via explicit view·proj rows,
      scan-converts to a packed-RGBA8 `colorOut` buffer; **no depth** (opaque
      last-writer-wins). `clearImage` + `rasterLines` kernels. `DebugRasterMetal`
      host in `metal_compute.py` (reused/grown vertex + color buffers, two bounded
      dispatches, readback → RGBA8 bytes). GPU cross-check vs the numpy reference
      PASSED (`tests/test_metal_debug_raster.py`, gated `RUN_METAL_DEBUG_RASTER`).
- [x] 3.2 Triangle rasteriser (`blendTris`): edge-function + barycentric,
      one thread per (triangle, screen-row) so each thread walks ≤ width pixels
      (watchdog-safe); flat vertex-A colour. Mirrored in the reference.
- [x] 3.3 Depth buffer: `uint` UAV `w×h` (`depthOut`), `InterlockedMin` packed
      depth (near→0). Lines are opaque and depth-ordered via a two-pass
      `depthLines` (atomic_min) → `colorLines` (write where the pixel owns the
      winning depth). `clearDepth` kernel resets it each frame.
- [x] 3.4 Alpha blend: `blendTris` composes transparent planes src-alpha over,
      depth-tested against the opaque line depth, NO depth write, ordered after
      the opaque passes — mirrors the Vulkan two-pipeline split. Verified: nearer
      line occludes farther, translucent panel blends (GPU cross-check +
      hostless: `test_opaque_triangle_fills_interior`, `test_alpha_blend_over_
      background`, `test_nearer_line_wins_depth`, `test_transparent_triangle_
      occluded_by_nearer_line`).
- [x] 3.5 Screen-space HUD text (`color.a > 1.5` sentinel bypasses view·proj) —
      implemented in `projectVertex` (kernel treats sentinel `xy` as NDC),
      mirrored in the reference; HUD glyph lines render (demo grid+AABB+HUD frame).
- [x] 3.6 Numpy/host parity harness (`src/skinny/debug_raster_ref.py` +
      `tests/test_debug_raster_ref.py`, 7 hostless tests): transform (projected +
      HUD sentinel + behind-eye drop), DDA line raster, NDC→pixel, RGBA8 pack —
      the authoritative CPU mirror the MSL kernel matches. Tri/blend cases extend
      this as those phases land.
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
