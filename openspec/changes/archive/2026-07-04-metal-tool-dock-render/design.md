# Design — Metal render paths for the tool docks

## Context

Native Metal (`metal_context.py`/`metal_compute.py`) is **compute-only** via
SlangPy/slang-rhi: `ComputePipeline` (compiles `.slang` → MSL, dispatches by
binding resources on a root shader object through a `ShaderCursor`, no descriptor
sets, no command-buffer recording, no memory barriers), plus `StorageBuffer`,
`StorageImage`, `SampledImage(3D)`, `UniformBuffer`, `ReadbackBuffer`,
`HostStorageBuffer`. There is **no graphics pipeline / render pass / render
encoder**. The megakernel Metal dispatch (`ComputePipeline.dispatch(w, h,
uniform_blob=…, binds={name: res}, bindless=…, bands=…)`) is the reference
pattern for any new Metal compute work.

## P1 — Metal material preview

`render_material_preview(material_id, prim, size)` renders a preview sphere/cube/
plane shaded with the material via `preview_pass.slang` (backend-agnostic: imports
`common`/`bindings`/`mtlx_*`/`generated_materials`, reads a 32-byte push block,
generates an orbit camera ray, intersects the primitive, evaluates the material,
writes RGBA32F to `previewOutput`). The Vulkan path uses `vk_compute.PreviewPipeline`
(descriptor set 0 = shared scene material bindings, set 1 = the output image) +
command-buffer recording + barriers + image→buffer readback.

### D1 — mirror the megakernel Metal dispatch (no descriptor sets)

Add `PreviewPipelineMetal` in `metal_compute.py`:
- Compile `preview_pass.slang` to MSL via the same SlangPy session path the
  megakernel `ComputePipeline` uses, **linking the same emit-time
  `generated_materials` module** so the preview shades identically (reuse the
  megakernel's program/session rather than re-linking from scratch — avoids a
  material-module mismatch).
- `dispatch(push_bytes, set0_binds, output_image)`: create the root shader object,
  bind the set-0 material resources by name (the same dict the megakernel builds
  via `_build_metal_binds`, filtered to material/scene bindings), bind the output
  image by name, `cur[pushName].set_data(push_bytes)`, run one compute pass over
  `size×size`, submit, wait idle. No barriers (Metal auto-syncs dispatch→readback).

### D2 — backend branch in `render_material_preview`

Top of the method: `if self.ctx.is_metal:` → the Metal path (build set-0 binds,
pack the same 32-byte push, dispatch `PreviewPipelineMetal`, read back the existing
`_preview_readback` → RGBA32F bytes). Else the existing Vulkan path, byte-unchanged.
The preview output image + readback are allocated backend-appropriately (Metal
`StorageImage` + `ReadbackBuffer`, already Metal-capable).

### D3 — verification

Headless Metal render test: build a Metal `Renderer` with a scene that has a
MaterialX/std_surface material, call `render_material_preview(mid, 0, 128)`, assert
it returns `(bytes, 128)` with non-degenerate pixels (a lit primitive, not all
zero / NaN). Vulkan preview numbers unchanged. Then rerun `skinny-gui --backend
metal` and confirm the Material Graph preview panel shows the material.

## P2 — Metal Camera Debug compute rasteriser (XL)

The debug viewport draws lines + transparent triangles (grid, frustum, camera
glyph, lens rings, focus/DOF planes, mesh wireframes, screen-space HUD text). CPU
geometry generation (`_gen_*`, `_emit_line/_emit_tri`, ~60% of the module) is
backend-agnostic and reused verbatim; only the Vulkan graphics rendering (render
pass, two graphics pipelines, swapchain, command-buffer draw) is Metal-incapable.

### D4 — software rasteriser in a compute kernel

New `debug_raster.slang` compute kernel + `DebugRasterMetal` host in
`metal_compute.py`:
- **Inputs** (StorageBuffers): a line-vertex buffer and a triangle-vertex buffer
  (each vertex = `float3 pos + float4 color`, the exact stream the CPU generators
  already fill), their counts, the 64-byte view·proj UBO, and output width/height.
- **Depth buffer**: a `uint` UAV (packed depth) sized `w×h`, cleared each frame.
- **Kernel(s)**:
  - Transform + scan-convert. Two viable structures — chosen at build time,
    measured: (a) **per-primitive** threads (one thread per line/triangle, each
    rasterises its own pixels with `atomic_min` depth into the UAV then a colour
    write guarded by a depth re-read), or (b) **per-pixel** tiled (each thread
    owns a pixel, walks primitives). Start with (a) for lines (Bresenham/DDA) +
    triangles (edge-function + barycentric); depth via `atomic_min` on packed
    `depth<<8 | tag`. Opaque lines/tris write depth; transparent planes
    (`color.a < 1`) blend in a second, depth-test-no-write pass ordered after
    opaque — mirrors the Vulkan two-pipeline split.
  - Screen-space HUD text (`color.a > 1.5` sentinel) bypasses view·proj, same as
    `debug_line.slang`.
- **Output**: an RGBA8 `StorageImage`; the dock blits it via the existing worker
  `DebugFrame` path — no new plumbing (the render-thread port already routes
  `renderer.debug_viewport` frames through `debug_frame_ready`).

### D5 — Metal DebugViewport plugs into the existing seam

A `DebugViewport(embedded=True)` on Metal constructs the Metal rasteriser instead
of the Vulkan graphics pipelines (branch on `ctx.is_metal` where the guard now
raises). `render_embedded(renderer)` fills the CPU vertex streams (unchanged
generators), uploads them to the Metal buffers, dispatches the raster kernel, and
returns the RGBA8 bytes — same signature the worker `_maybe_render_debug` already
calls. Camera/display state (orbit/free cam, toggles) is unchanged (CPU-side).
The Vulkan graphics path stays intact for the Vulkan backend.

### D6 — interim + sequencing

- **Interim** (ships with P1): the Camera Debug dock detects `proxy._backend_name
  == "metal"` and, until P2 lands, shows a "Camera Debug view renders on the
  Vulkan backend only" notice instead of attempting the create/render (stops the
  per-frame `[render error]`).
- **Sequencing**: P1 (small, ship first) → interim notice → P2 in its own phases
  (kernel: lines → triangles → depth → blend → HUD → wire into DebugViewport →
  drop the interim notice). P2 is landable incrementally and never touches the
  production render path.

### D7 — risks

- P1: `generated_materials` module identity across the megakernel and preview
  Metal sessions — reuse the megakernel program to avoid a re-link mismatch; the
  32-byte push scalar layout is consistent (SlangPy scalar rules).
- P2: `atomic_min` packed-depth contention on dense wireframes; blend ordering
  correctness for overlapping transparent planes; MSL rasteriser numerical edge
  cases (degenerate tris, near-plane clipping). All confined to a dev tool, not
  the production path. Metal watchdog: the raster dispatch is bounded (fixed
  vertex budget); no per-pixel unbounded loops.
