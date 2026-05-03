# Skinny

> **Note:** This project is developed with [Claude Code](https://claude.ai/claude-code)
> and serves as a testbed for experimenting with new rendering algorithms.
> The codebase evolves rapidly and stability is not guaranteed.

Skinny is a physically based renderer built on a Vulkan compute shader
pipeline. It started as a human skin rendering testbed -- and retains
first-class skin support -- but the core pipeline handles arbitrary MaterialX
materials, OpenUSD scenes, ray-traced geometry, image-based lighting,
microfacet specular, and energy-conservation checks.

## Features

- **Three-layer skin optics** -- epidermis (melanin), dermis (hemoglobin + ink),
  subcutaneous fat, each with independent absorption, scattering, thickness, and
  anisotropy
- **MaterialX material pipeline** -- custom `ND_skinny_skin_*` layer nodedefs
  plus a `ND_skinny_layered_skin_stack` combiner, code-generated to Slang via
  `MaterialXGenSlang`
- **OpenUSD scene loading** -- meshes, transforms, `UsdShade.Material` bindings,
  lights (`DomeLight`, `DistantLight`, `SphereLight`, `RectLight`), and
  per-prim material assignment
- **Flat material support** -- USD prims bound to `UsdPreviewSurface` or
  MaterialX `standard_surface` render alongside skin materials in the same scene
- **Four sampling strategies** -- path tracing, MIS, bidirectional, and stored
  BDPT
- **Scattering modes** -- BSSRDF + Volume, BSSRDF only, Volume only, or Off,
  selectable per scene
- **Furnace mode** -- unit-sphere + white-environment energy conservation test;
  violations tinted pink
- **Fitzpatrick I--VI presets** -- male/female variants covering the clinical
  skin-colour axis
- **Detail layer** -- statistical pores and vellus hair sheen
- **Tattoo support** -- alpha-driven ink density in the dermis layer
- **Tk control panel** -- collapsible per-material sliders, colour pickers,
  light direction picker, preset save/load
- **Web mode** -- Panel (HoloViz) browser UI with per-user server-side
  rendering, H264 video streaming over WebSocket, hardware-accelerated encoding
  (NVENC / QSV / AMF), and WebCodecs decoding in the browser
- **Multi-user sessions** -- up to 4 concurrent browser sessions, each with
  independent renderer, camera, and parameters
- **GPU selection** -- `--gpu {intel,nvidia,amd,discrete,auto}` flag on both
  desktop and web entry points
- **Persistent settings** -- parameter snapshots saved and restored between
  sessions

## Requirements

- Python 3.11 or newer
- Vulkan 1.2 capable GPU and current graphics driver
- GLFW-compatible desktop environment
- Slang compiler (`slangc`) on `PATH`

Python dependencies (`pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `numpy` | Linear algebra, mesh processing |
| `slangpy` | Slang shader compilation and reflection |
| `vulkan` | Vulkan API bindings |
| `glfw` | Window creation and input |
| `Pillow` | Image I/O (HDR, textures, tattoos) |
| `MaterialX` | Material definitions and Slang code generation |

Optional:

| Package | Purpose |
|---------|---------|
| `usd-core` | OpenUSD scene loading (`pip install -e ".[usd]"`) |
| `panel` | Web UI framework (`pip install -e ".[web]"`) |
| `bokeh` | Panel dependency (Tornado server) |
| `av` (PyAV) | H264 video encoding via FFmpeg bindings |

## Setup

```powershell
python -m venv .
.\Scripts\python -m pip install --upgrade pip
.\Scripts\python -m pip install -e .
```

For USD scene support:

```powershell
.\Scripts\python -m pip install -e ".[usd]"
```

For web mode (Panel + H264 streaming):

```powershell
.\Scripts\python -m pip install -e ".[web]"
```

For development tools:

```powershell
.\Scripts\python -m pip install -e ".[dev]"
```

Verify the Slang compiler:

```powershell
slangc -version
```

## Running

### Default (SDF head + HDR environments)

```powershell
.\Scripts\skinny.exe
```

Or as a module:

```powershell
.\Scripts\python -m skinny.app
```

### USD scene

```powershell
.\Scripts\skinny.exe assets/demo_head.usda
```

Any `.usda` / `.usdc` / `.usdz` file with MaterialX-bound or
`UsdPreviewSurface`-bound materials will load.

The renderer has been tested with the
[Usd-Mtlx-Example](https://github.com/pablode/Usd-Mtlx-Example) repository, a
USD scene using MaterialX resources designed to assess rendering consistency
across different renderers.

### Web mode

```powershell
.\Scripts\skinny-web.exe --port 8080
```

With a USD scene:

```powershell
.\Scripts\skinny-web.exe --port 8080 --usd assets/Usd-Mtlx-Example/scene.usda
```

Or as a module:

```powershell
.\Scripts\python -m skinny.web_app --port 8080 --gpu auto --usd assets/demo_head.usda
```

Open `http://localhost:8080/skinny` in a browser. Each tab gets an independent
renderer session with its own camera and parameters. Video is H264-encoded
server-side and decoded via WebCodecs in the browser.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Server port |
| `--gpu` | auto | GPU selection: `intel`, `nvidia`, `amd`, `discrete`, `auto` |
| `--max-sessions` | 4 | Max concurrent browser sessions |
| `--usd` | — | Path to USD scene (alternative to positional arg) |
| `--usdMtlx` | off | Use USD's built-in usdMtlx plugin instead of MaterialX API fallback |

### Mesh heads (legacy)

Place `.obj` files (with optional normal/roughness/displacement maps) in
`heads/<name>/` directories. They are discovered automatically at startup.

## Controls

Keyboard and mouse controls are shown in the on-screen HUD.

| Input | Action |
|-------|--------|
| Left drag | Orbit camera (orbit mode) / look around (free mode) |
| Right drag | Pan orbit target |
| Scroll | Zoom (orbit) / adjust speed (free) |
| `C` | Toggle orbit / free camera |
| `W A S D` | Move in free-camera mode |
| `Q / E` | Move down / up in free-camera mode |
| `Tab / Shift+Tab` | Next / previous parameter |
| Arrow keys | Adjust selected parameter |
| `1`--`9` | Jump to parameter |
| `F` | Recenter camera |
| `R` | Reset parameters |
| `P` | Print all parameters |
| `H` | Print help |
| `Space / F1` | Toggle HUD |
| `Esc` | Quit |

## Assets

### HDR Environments

Radiance `.hdr` files in `hdrs/`. The helper script `src/skinny/fetch_hdrs.py`
documents the curated Poly Haven HDRIs used for portrait/skin lighting.

### Head Models

The analytic fallback is an SDF head based on Loomis-style proportions.
Additional mesh heads are discovered from `heads/`:

- Each subdirectory containing an `.obj` becomes one model
- Loose top-level `.obj` files are also loaded
- Texture maps are matched by filename keyword:
  `normal`/`nrm`/`nor`, `rough`/`roughness`, `displacement`/`disp`/`height`/`bump`

Displacement can be baked into the mesh after midpoint subdivision.

### USD Scenes

Example scenes ship in `assets/`:

| File | Description |
|------|-------------|
| `demo_head.usda` | Head mesh with layered skin material |
| `cornell_box.usda` | Classic Cornell box |
| `cornell_box_emissive.usda` | Cornell box with emissive geometry |
| `cornell_box_rectlight.usda` | Cornell box with rect light |
| `cornell_box_sphere.usda` | Cornell box with sphere light |
| `dual_skin_demo.usda` | Two prims with different skin materials |
| `mtlx_skin_demo.usda` | MaterialX skin material demo |
| `skin_sphere_light_demo.usda` | Skin under sphere lighting |
| `test_scene.usda` | Multi-material test scene |

### Tattoos

Tattoo images in `tattoos/`. Alpha drives ink density; RGB drives pigment
colour contribution in the dermis.

## Rendering Modes

### Scattering

| Mode | Description |
|------|-------------|
| BSSRDF + Volume | Both subsurface estimators active |
| BSSRDF only | Smooth layered diffuse response |
| Volume only | Delta-tracked transport through layered medium |
| Off | Surface-only shading |

### Sampling

| Strategy | Description |
|----------|-------------|
| Path tracing | Camera-path estimator |
| MIS | BSDF + light samples combined via power heuristic |
| Bidirectional | MIS plus explicit volume light-connection samples |
| Stored BDPT | Light-side surface vertex stored per invocation, connected with visibility-tested geometry term |

### Furnace Mode

Swaps the scene to a unit sphere under unit-white radiance. Pixels exceeding
energy conservation tolerance are tinted pink.

## MaterialX Skin Model

Skinny defines custom MaterialX nodedefs in `src/skinny/mtlx/`:

### Layer nodedefs (each produces a `scatteringlayer`)

- **`ND_skinny_skin_epidermis`** -- melanin absorption + scattering
- **`ND_skinny_skin_dermis`** -- hemoglobin + blood oxygenation + optional ink
- **`ND_skinny_skin_subcut`** -- fixed-physics fat layer
- **`ND_skinny_scattering_layer`** -- generic escape hatch for non-skin media

### Stack nodedef

- **`ND_skinny_layered_skin_stack`** -- combines three layers into
  `surfaceshader` + `volumeshader` outputs with GGX specular, detail
  (pores/hair), and per-layer volume transport

### Slang implementations

Function-form Slang implementations live in `src/skinny/mtlx/genslang/` and are
referenced by `<implementation target="genslang">` tags in the nodedef files.

## Implementation Map

### Python

| File | Purpose |
|------|---------|
| `app.py` | GLFW window, input handling, settings persistence (desktop entry point) |
| `web_app.py` | Panel web app, per-session renderer, Tornado video WebSocket (web entry point) |
| `params.py` | Shared parameter definitions (`ParamSpec`), get/set helpers |
| `hardware.py` | GPU enumeration, vendor detection, encoder selection |
| `video_encoder.py` | H264/JPEG encoding with hardware-aware fallback chain |
| `control_panel.py` | Tk control panel with collapsible per-material sections |
| `renderer.py` | Vulkan resources, uniforms, environment/mesh/texture upload, frame loop |
| `vk_compute.py` | Compute pipeline, descriptor layout, GPU buffer/image helpers |
| `vk_context.py` | Vulkan instance, device, queue setup (windowed + headless) |
| `scene.py` | Scene graph data classes (`MeshInstance`, `Material`, `Light*`, `Scene`) |
| `materialx_runtime.py` | MaterialX document loading, Slang code generation, uniform packing |
| `usd_loader.py` | USD stage to `Scene` conversion (with MaterialX API fallback) |
| `environment.py` | Built-in and HDR environment loading |
| `mesh.py` | OBJ loading, normalization, subdivision, displacement, BVH construction |
| `head_textures.py` | Head texture loading (normal, roughness, displacement) |
| `presets.py` | Fitzpatrick I--VI presets and user preset save/load |
| `settings.py` | User settings persistence |
| `tattoos.py` | Tattoo image loading |
| `fetch_hdrs.py` | Poly Haven HDRI download helper |

### Shaders (Slang)

| File | Purpose |
|------|---------|
| `main_pass.slang` | Primary camera path, composition, MIS, stored-BDPT |
| `common.slang` | Shared types, `FrameConstants`, `MtlxSkinParams` UBO layout |
| `bindings.slang` | Descriptor set bindings |
| `skin_material.slang` | Skin shading entry point (specular + BSSRDF + volume dispatch) |
| `skin_bssrdf.slang` | Layered skin optics, BSSRDF, GGX specular |
| `volume_render.slang` | Delta-tracked volume transport through layered medium |
| `material_eval.slang` | Per-hit material dispatch (`evalMaterial`) |
| `flat_material.slang` | Flat (non-skin) material evaluation |
| `mtlx_std_surface.slang` | MaterialX `standard_surface` approximation |
| `mtlx_closures.slang` | MaterialX closure helpers |
| `mtlx_noise.slang` | MaterialX noise functions |
| `scene_trace.slang` | TLAS/BLAS ray traversal |
| `scene_lights.slang` | Light sampling (distant, sphere, rect, emissive tri) |
| `mesh_head.slang` | BVH traversal and ray/triangle intersection |
| `sdf_head.slang` | Analytic SDF head |
| `environment.slang` | Environment lookup and furnace fallback |
| `detail.slang` | Statistical pores and vellus hair sheen |

## Papers and References

| Area | Files | Reference |
|------|-------|-----------|
| Subsurface transport | `skin_bssrdf.slang` | Jensen, Marschner, Levoy, Hanrahan, "A Practical Model for Subsurface Light Transport", SIGGRAPH 2001 |
| Quantized diffusion | `skin_bssrdf.slang` | d'Eon and Irving, "A Quantized-Diffusion Model for Rendering Translucent Materials", SIGGRAPH/TOG 2011 |
| Normalized diffusion | `skin_bssrdf.slang` | Christensen and Burley, "Approximate Reflectance Profiles for Efficient Subsurface Scattering", Disney/SIGGRAPH 2015 |
| Human skin optics | `skin_bssrdf.slang`, `presets.py` | Donner and Jensen, "A Spectral BSSRDF for Shading Human Skin", EGSR 2006 |
| Real-time skin pipeline | `renderer.py`, `mesh_head.slang`, `sdf_head.slang` | d'Eon and Luebke, "Advanced Techniques for Realistic Real-Time Skin Rendering", GPU Gems 3 Ch. 14, 2007 |
| MIS | `common.slang`, `main_pass.slang`, `volume_render.slang` | Veach, "Robust Monte Carlo Methods for Light Transport Simulation", PhD thesis, 1997 |
| Bidirectional path tracing | `main_pass.slang` | Veach and Guibas, "Bidirectional Estimators for Light Transport", 1995 |
| Bidirectional path tracing | `main_pass.slang` | Lafortune and Willems, "Bi-Directional Path Tracing", 1993 |
| GGX microfacet | `skin_bssrdf.slang` | Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007 |
| Fresnel approximation | `skin_bssrdf.slang`, `detail.slang` | Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering", 1994 |
| Henyey-Greenstein phase | `volume_render.slang` | Henyey and Greenstein, "Diffuse Radiation in the Galaxy", 1941 |
| Delta/Woodcock tracking | `volume_render.slang` | Woodcock et al., "Techniques Used in the GEM Code for Monte Carlo Neutronics Calculations", 1965 |
| Ray/triangle intersection | `mesh_head.slang` | Moeller and Trumbore, "Fast, Minimum Storage Ray/Triangle Intersection", 1997 |

Supporting techniques (ACES tone mapping, PCG hashing, median-split BVH,
Worley noise, Box-Muller sampling, Loomis-style head proportions) are standard
implementation building blocks.

## Development

Compile Python:

```powershell
.\Scripts\python -m py_compile src\skinny\app.py src\skinny\renderer.py
```

Compile main shader:

```powershell
slangc src\skinny\shaders\main_pass.slang -target spirv -entry mainImage -stage compute -o src\skinny\shaders\main_pass.spv -I src\skinny\shaders
```

Run tests:

```powershell
.\Scripts\python -m pytest
```

## License

MIT
