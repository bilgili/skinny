# Changelog

All notable changes to Skinny are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### UI and interaction

- Camera debug viewport (`F2`) with frustum, lens rings, focus plane, DOF
  planes, render-area outline, ground grid, mesh wireframes, AABBs, and
  camera-body glyph
- Screen-space HUD inside the debug viewport listing its keyboard
  shortcuts; toggleable with `Space`
- Lens focus overlay (`L`), lens-vignette debug visualisation (`V`),
  zoom-rectangle drag (`Z` to arm, `X` to reset) hotkeys on the main
  window
- Updated on-screen HUD and `H` help text to list the full key set

## [0.1.0] - 2026-05-02

First release. Skinny is a physically based renderer built on a Vulkan compute
shader pipeline. It started as a human skin rendering testbed; this release
ships the full skin feature set alongside generic MaterialX material support
and OpenUSD scene loading.

### Rendering

- Three-layer biological skin model: epidermis (melanin absorption), dermis
  (hemoglobin + blood oxygenation + optional tattoo ink), subcutaneous fat
- GGX microfacet specular with Fresnel
- Point-BSSRDF subsurface scattering (quantized diffusion / normalized
  diffusion profiles)
- Delta-tracked (Woodcock) volume transport through the layered skin medium
- Scattering mode selector: BSSRDF + Volume, BSSRDF only, Volume only, Off
- Four sampling strategies: path tracing, MIS, bidirectional, stored BDPT
- Image-based lighting from Radiance HDR environments
- Analytic light (distant) and area lights (sphere, rect, emissive triangles)
- Statistical pore and vellus hair detail layer
- Furnace mode for energy conservation verification
- ACES filmic tone mapping

### MaterialX

- Custom nodedefs for skin layers: `ND_skinny_skin_epidermis`,
  `ND_skinny_skin_dermis`, `ND_skinny_skin_subcut`, plus generic
  `ND_skinny_scattering_layer` escape hatch
- `ND_skinny_layered_skin_stack` combiner producing `surfaceshader` +
  `volumeshader` outputs
- Function-form Slang implementations (`mtlx/genslang/`) referenced by
  `<implementation target="genslang">` tags
- MaterialX runtime (`materialx_runtime.py`) for document loading, Slang code
  generation, uniform block reflection, and scalar-layout buffer packing
- All skin biological parameters exposed as MaterialX inputs, driveable by
  constants or UV-sampled images

### OpenUSD

- USD scene loader (`usd_loader.py`): `UsdGeom.Mesh` triangulation, transform
  baking, `UsdShade.Material` binding resolution
- Light import: `DomeLight`, `DistantLight`, `SphereLight`, `RectLight`
- Per-prim material assignment with `materialId` dispatch
- Flat material path for `UsdPreviewSurface` and MaterialX `standard_surface`
  prims alongside skin materials
- Example scenes in `assets/`: Cornell box variants, demo head, dual-skin demo,
  sphere light demo, multi-material test scene

### Scene and geometry

- Scene graph abstraction (`scene.py`): `MeshInstance`, `Material`, `Light*`,
  `Scene` data classes
- TLAS + per-instance BLAS pool with transform and material ID
- OBJ mesh loading with optional midpoint subdivision and displacement baking
- BVH construction (median-split) for ray/triangle intersection
- Analytic SDF head fallback (Loomis-style proportions)
- Head texture maps: normal, roughness, displacement (auto-discovered by
  filename keyword)
- Tattoo support: alpha-driven ink density in dermis layer

### UI and interaction

- GLFW window with orbit and free camera modes
- Tk control panel with collapsible per-material sections
- Colour picker for light and material diffuse colour
- Light direction picker (hemisphere widget)
- Keyboard parameter navigation (Tab/Shift+Tab, arrow keys, number jump)
- Fitzpatrick I--VI presets (male/female variants, 12 total)
- User preset save/load to `~/.skinny/presets/`
- Persistent settings across sessions (`skinny.settings`)

### Infrastructure

- Vulkan 1.2 compute pipeline with scalar block layout
  (`VK_EXT_scalar_block_layout`)
- Descriptor indexing for bindless texture arrays
  (`VK_EXT_descriptor_indexing`)
- Slang shader compilation via `slangpy`
- Python 3.11+, packaged via `pyproject.toml` with `skinny` entry point
- Optional `usd-core` dependency (`pip install -e ".[usd]"`)
