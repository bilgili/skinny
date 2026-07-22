# Proposal: glb-asset-import

## Why

Locally generated 3D assets (TRELLIS.2 image→3D on this machine, and any other GLB source) cannot currently be used in skinny with their materials: converting a GLB to USD with Apple's system `usdextract` produces a textbook UsdPreviewSurface network, but the USD loader drops every texture in it, so the asset renders flat white. Conversion is well-understood (Apple's `/usr/bin/usdextract` proved the target UsdPreviewSurface shape on macOS), but the pipeline must run on Windows and Linux hosts too, so the converter ships in-repo as pure Python; the gaps are two loader defects, the converter, and a one-call import path for MCP agents.

Verified end-to-end with a TRELLIS.2-generated crown (2.59 M faces): geometry, BVH, and framing render correctly on Metal; all three textures (diffuse, packed metallic `.b`, packed roughness `.g`) are silently discarded.

## What Changes

- **Fix interface-connected texture `file` resolution.** `_resolve_texture_binding` in `usd_loader.py` reads `file_input.Get()` directly; when `file` is a *connection* to a Material interface input (how Apple's glTF converter authors it: `file <- Material0.baseColorTexture`), `Get()` returns None and the whole binding is dropped. The loader must follow the connection to the interface input and read the asset there — mirroring what `_resolve_connected_value` already does for constants.
- **Honor `UsdTransform2d` in the st chain.** The loader ignores `UsdTransform2d` nodes between the primvar reader and `UsdUVTexture` (glTF-derived USD always carries `scale (1,-1)`, `translation (0,1)` — the V-flip). Sampling without it flips every texture vertically.
- **New MCP tool `scene_import_glb`.** One call: GLB path (in allowed roots) → built-in pure-Python GLB→USD conversion (pygltflib + pxr, platform-independent: macOS, Linux, Windows) into an allowed root → `scene_add_model` reference with the same transform arguments. Makes generated-asset import a single MCP action for agents driving the renderer on every platform skinny runs on. Apple's `usdextract` remains a manual alternative on macOS, not a dependency.
- **Regression asset + gate.** A GLB-derived confirming scene (decimated crown or equivalent packed-texture UsdPreviewSurface asset) with non-white patch assertions, in the style of `TestMaterialXGraphDemoRender`.

## Capabilities

### New Capabilities
- `usd-texture-intake`: resolution of UsdPreviewSurface texture bindings from composed USD scenes — interface-connected `file` inputs, `UsdTransform2d` st-chain transforms, packed-channel selectors, colorspace/wrap/scale/bias — independent of which front-end loaded the stage.

### Modified Capabilities
- `mcp-scene-control`: adds the `scene_import_glb` structural tool (conversion + reference in one call, allowed-roots checked on both the GLB input and the USD output, degrades to a pollable job like other structural adds).

## Impact

- `src/skinny/usd_loader.py` — `_resolve_texture_binding` (interface connection follow), st-chain walk (`UsdTransform2d`), `TextureBinding` carries the UV transform (or the loader bakes it into mesh UVs — design decision).
- `src/skinny/mcp_server.py` — new `scene_import_glb` tool; new `src/skinny/glb_import.py` converter module (pygltflib + pxr); `pygltflib` added to project dependencies.
- Renderer texture sampling path only if the per-binding UV-transform design is chosen (shader change + recompile); the UV-bake design leaves shaders untouched.
- Tests: hostless loader tests for both fixes; GPU-marked render gate with the regression asset; MCP tool tests mirroring `scene_add_model` path-validation coverage.
- Docs: `docs/PythonAPI.md` (MCP tool contract), `docs/Architecture.md` (loader texture resolution), `README.md` (asset-import workflow incl. TRELLIS.2 pipeline).
