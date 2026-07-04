# Metal render paths for the tool docks

## Why

`restore-render-thread-tool-docks` restored all five View-menu tool docks under
render-thread ownership. Interactive testing on the native **Metal** backend
(the default on this Apple-Silicon host) then exposed that **two of the docks
render nothing on Metal** because their GPU render paths are Vulkan-only —
latent gaps that were hidden while the docks were stubbed:

- **Material Graph preview** — `Renderer.render_material_preview` drives
  `vk_compute.PreviewPipeline` with Vulkan descriptor sets and command-buffer
  recording. On Metal it raises `AttributeError: 'ComputePipeline' object has no
  attribute 'descriptor_set_layout'`, so the preview panel stays blank.
- **Camera Debug viewport** — `DebugViewport` is a full Vulkan **graphics**
  rasteriser (render pass + line/triangle pipelines + vertex/fragment shaders +
  swapchain). It explicitly raises "requires the Vulkan backend (it is a Vulkan
  rasteriser)" on Metal. The native Metal backend is **compute-only** — no
  graphics pipeline, render encoder, or render pass — so there is nothing to bind
  it to.

Both render on Vulkan; neither renders on Metal. This change closes the two gaps
so the tool docks are usable on the machine's default backend.

## What changes

Two independent deliverables (the second is far larger):

- **P1 — Metal material preview** (small). Add a Metal-native preview dispatch
  that mirrors the megakernel Metal dispatch (`ComputePipeline.dispatch` — root
  shader object + bind-by-name via `ShaderCursor`, no descriptor sets). The
  preview shader (`preview_pass.slang`) is already backend-agnostic and compiles
  to MSL via SlangPy. `render_material_preview` branches on `ctx.is_metal`:
  Vulkan path unchanged; Metal path builds the set-0 material bind dict (reusing
  the megakernel `_build_metal_binds` machinery), sets the 32-byte push block,
  dispatches `size×size`, and reads back the RGBA32F output. Vulkan output is
  byte-unchanged.

- **P2 — Metal Camera Debug compute rasteriser** (extra-large). The native Metal
  backend has no graphics pipeline, so the debug viewport is re-implemented as a
  **software line/triangle rasteriser running in a Metal compute kernel**. The
  existing CPU geometry generation (grid, frustum, camera glyph, lens rings,
  focus/DOF planes, mesh wireframes, HUD text — ~60% of `debug_viewport.py`, all
  backend-agnostic) is reused unchanged; a new MSL compute kernel scan-converts
  the emitted line/triangle vertex streams through the debug camera's view·proj,
  with a UAV depth buffer (depth test + line/tri ordering) and alpha blending for
  the transparent planes, writing an RGBA8 output image the dock blits via the
  existing worker `DebugFrame` path. The Metal debug viewport plugs into the same
  `renderer.debug_viewport` seam the render-thread port already established; the
  Vulkan graphics rasteriser is untouched and stays the Vulkan-backend path.

Until P2 lands, the Camera Debug dock on Metal shows a clear "Vulkan-only on this
backend" notice instead of a blank canvas + per-frame `[render error]`.

## Impact

- **Affected specs**: `metal-backend` (ADD the two Metal render-path
  requirements).
- **Affected code**: `src/skinny/metal_compute.py` (preview adapter; P2 raster
  kernel host), `src/skinny/renderer.py` (`render_material_preview` Metal
  branch), `src/skinny/debug_viewport.py` (P2 Metal rasteriser path + the interim
  Metal notice), new `src/skinny/shaders/debug_raster.slang` (P2), and the two
  docks' Metal-aware messaging.
- **Risk**: P1 is bounded and mirrors a proven dispatch. P2 is a new
  self-contained compute subsystem for a development tool; it does not touch the
  production render path and is landable in its own phases. Both are gated so the
  Vulkan paths are byte-unchanged.
- **Backends**: no change to Vulkan behaviour; adds Metal parity for the two
  dock render paths.
