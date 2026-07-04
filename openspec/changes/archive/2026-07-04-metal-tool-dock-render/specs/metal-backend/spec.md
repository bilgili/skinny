## ADDED Requirements

### Requirement: Metal material preview render
The material preview render SHALL run on the native Metal backend, not only Vulkan.
`Renderer.render_material_preview` SHALL branch on the active backend: on Metal it
dispatches the backend-agnostic `preview_pass.slang` through a compute pipeline that
binds the scene material resources and the preview output image by name (no Vulkan
descriptor sets), reads back the RGBA32F pixels, and returns the same
`(pixels, size)` shape the Vulkan path returns; the Vulkan path is byte-unchanged.

#### Scenario: Material Graph preview renders on Metal
- **WHEN** `skinny-gui --backend metal` opens the Material Graph dock for a scene
  material and requests a preview
- **THEN** `render_material_preview` returns a rendered RGBA32F image (a lit
  primitive, not all-zero/NaN and not an error) and the dock shows the material,
  without the Vulkan-only `descriptor_set_layout` failure

#### Scenario: Vulkan preview unchanged
- **WHEN** the preview renders on the Vulkan backend
- **THEN** its output is identical to before this change

### Requirement: Metal Camera Debug viewport via compute rasteriser
The Camera Debug viewport SHALL render on the native Metal backend via a software line/triangle rasteriser running in a Metal compute kernel, since the backend has no graphics pipeline.
The Metal path SHALL reuse the existing backend-agnostic CPU geometry generation,
scan-convert the emitted line/triangle vertex streams through the debug camera's
view·proj with a depth buffer and alpha blending for transparent planes, and emit
an RGBA8 frame through the same worker `DebugFrame` path the dock already consumes.
Until the Metal rasteriser lands, the dock SHALL show a clear "Vulkan-only on this
backend" notice on Metal rather than a blank canvas with per-frame render errors.
The Vulkan graphics rasteriser SHALL remain the Vulkan-backend path, unchanged.

#### Scenario: Camera Debug renders on Metal
- **WHEN** `skinny-gui --backend metal` opens the Camera Debug view for a loaded
  scene
- **THEN** the embedded debug render returns a non-empty RGBA8 frame showing the
  grid/frustum/geometry, and orbit/pan/zoom + overlay toggles update it

#### Scenario: Interim Metal notice before the rasteriser lands
- **WHEN** the Camera Debug dock is opened on Metal and the Metal rasteriser is not
  yet available
- **THEN** the dock shows a "Vulkan-only on this backend" notice and does not emit
  per-frame render errors
