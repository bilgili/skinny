# Skinny — Assets

This document covers the asset directories the renderer discovers on disk:
`hdrs/` environments, `heads/` models with their texture maps, and USD scene
assets.

Tattoo images (`tattoos/`) are documented with the model that consumes them, in
[SkinRendering.md § Tattoos (asset)](SkinRendering.md#tattoos-asset), along with
the rest of the skin model these assets feed.

---

## Assets

### HDR Environments

Radiance `.hdr` (and discovered sibling `.exr` / `.pfm`) files in `hdrs/`. The
helper script `src/skinny/fetch_hdrs.py` documents the curated Poly Haven
HDRIs used for portrait/skin lighting. The Qt and web sidebars expose a
"Load HDR" picker that scans the chosen file's directory for additional
formats.

### Head Models

Head geometry (analytic SDF head + discovered `heads/*.obj` mesh heads with
detail maps) is documented in [SkinRendering.md](SkinRendering.md).

### USD Scenes

Example scenes ship in `assets/`:

Lighting is all-or-nothing: a USD scene containing any active supported light
or emissive material uses only its authored sources. A light-less USD scene,
OBJ, or default head receives Skinny's default DistantLight and built-in IBL
together. Zero-intensity and runtime-disabled authored lights still express
author intent and therefore suppress the fallback pair.

| File | Description |
|------|-------------|
| `demo_head.usda` | Head mesh with layered skin material |
| `cornell_box_emissive.usda` | Cornell box with emissive geometry |
| `cornell_box_rectlight.usda` | Cornell box with rect light |
| `cornell_box_sphere.usda` | Cornell box with sphere light |
| `dual_skin_demo.usda` | Two prims with different skin materials |
| `glass_caustics_test.usda` | Glass material refraction / caustics test |
| `mtlx_skin_demo.usda` | MaterialX skin material demo |
| `skin_sphere_light_demo.usda` | Skin under sphere lighting |
| `test_scene.usda` | Multi-material test scene |
| `three_materials_demo.usda` | Marble + wood + brass MaterialX nodegraphs |

#### Importing generated GLB assets (image-to-3D)

Local image-to-3D models (e.g. **TRELLIS.2**) emit textured `.glb` meshes with
PBR materials (base color + packed metallic-roughness). Bring one into a scene
in one step with the `scene_import_glb` MCP tool — it runs a built-in
pure-Python GLB→USD converter (`skinny.glb_import`, pygltflib + pxr; the same
on macOS, Linux, and Windows, no external tools) and references the result:

```python
from skinny.glb_import import convert_glb_to_usd
usd = convert_glb_to_usd("crown.glb", "crown_usd/")   # → crown_usd/crown.usdc + textures
```

The converter authors a UsdPreviewSurface network the renderer reads directly:
base color and packed metallic (`.b`) / roughness (`.g`) as `UsdUVTexture`
nodes, UVs pre-flipped to USD's V convention. Out-of-scope glTF features (Draco
compression, sparse accessors, skinning, animation) are refused by name. On
macOS, Apple's system `usdextract` is an alternative that produces
interface-connected texture inputs and a `UsdTransform2d` V-flip; the loader
resolves both shapes too, so externally-converted USD renders correctly as
well.
