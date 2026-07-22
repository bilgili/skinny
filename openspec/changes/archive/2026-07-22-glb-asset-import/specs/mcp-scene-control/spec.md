# mcp-scene-control (delta)

## ADDED Requirements

### Requirement: GLB import tool
The MCP server SHALL expose a `scene_import_glb` structural tool that converts a GLB file to USD via a built-in, platform-independent converter (pure Python: GLB parsing plus pxr USD authoring; available identically on macOS, Linux, and Windows) and references the converted result into the scene through the same code path, validation, and job-degradation behavior as `scene_add_model`. The converter SHALL author `UsdUVTexture` `file` values directly, extract embedded images (PNG/JPEG/WebP) beside the `.usdc`, wire packed metallicRoughness as `metallic ← .b` and `roughness ← .g`, and emit UVs in USD's V convention. A GLB using a glTF feature outside the converter's scope (Draco compression, sparse accessors, skinning, animation, vendor extensions) SHALL be refused with an error naming the unsupported feature. The tool SHALL accept `glb_path`, optional `name`, `parent`, transform arguments identical to `scene_add_model` (`translate` / `rotate_euler_deg` / `scale` / `matrix`), an optional `out_dir` (default: a `<glb_stem>_usd` directory beside the GLB, created if absent), and an `overwrite` flag (default false). Both `glb_path` and the resolved `out_dir` SHALL pass the allowed-roots check before any conversion runs. When `out_dir` already contains a prior conversion (any `.usd*` file) and `overwrite` is false, the tool SHALL refuse without running the converter or touching the renderer. Conversion SHALL run on the tool thread (never the render thread) and SHALL complete (or fail) before any renderer mutation is attempted.

#### Scenario: One-call GLB import
- **WHEN** `scene_import_glb` is called with a GLB inside the allowed roots
- **THEN** the converter produces a `.usdc` plus extracted textures in the output directory, the result is referenced under the requested parent with the requested transform, and the reply carries the new prim path and version counters exactly as a `scene_add_model` reply does

#### Scenario: GLB path outside allowed roots is refused
- **WHEN** `scene_import_glb` is called with a `glb_path` (or an `out_dir`) that resolves outside the allowed roots
- **THEN** the tool returns a path-rejection error naming the offending path before any subprocess or renderer interaction occurs

#### Scenario: Unsupported glTF feature is refused by name
- **WHEN** `scene_import_glb` is called with a GLB that uses an out-of-scope glTF feature (e.g. Draco compression)
- **THEN** the tool returns an error naming the unsupported feature and the renderer state is untouched

#### Scenario: Conversion failure leaves the scene untouched
- **WHEN** the converter raises on a malformed GLB
- **THEN** the tool returns the failure as an error, no prim is added, and no renderer mutation occurs

#### Scenario: Existing conversion is not silently overwritten
- **WHEN** `scene_import_glb` is called with an `out_dir` that already contains a `.usd*` file and `overwrite` is not set
- **THEN** the tool returns an error naming the existing conversion and neither the converter nor the renderer is invoked
