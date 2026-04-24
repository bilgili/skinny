# Skinny

Skinny is a real-time physically based human skin renderer built around a
Vulkan compute shader pipeline. It combines ray-traced head geometry, image
based lighting, surface microfacet specular, point-BSSRDF subsurface response,
volume ray marching through layered skin, skin detail maps, tattoos, and
energy-conservation checks.

## Requirements

- Python 3.11 or newer
- Vulkan-capable GPU and current graphics driver
- GLFW-compatible desktop environment
- Slang compiler (`slangc`) available on `PATH`

Python dependencies are declared in `pyproject.toml`:

- `numpy`
- `slangpy`
- `vulkan`
- `glfw`
- `Pillow`

## Setup

From the repository root:

```powershell
python -m venv .
.\Scripts\python -m pip install --upgrade pip
.\Scripts\python -m pip install -e .
```

If you want development tools:

```powershell
.\Scripts\python -m pip install -e ".[dev]"
```

Check that `slangc` is visible:

```powershell
slangc -version
```

## Running

Run via the project entry point:

```powershell
.\Scripts\skinny.exe
```

Or directly as a module:

```powershell
.\Scripts\python -m skinny.app
```

On startup the app creates a Vulkan window and a Tk control panel. Runtime
settings are restored from the user settings file managed by `skinny.settings`.

## Controls

Keyboard and mouse controls are also shown in the on-screen HUD.

- `Left mouse drag`: orbit camera, or look around in free-camera mode
- `Right mouse drag`: pan orbit target
- `Scroll`: zoom orbit camera, or adjust free-camera speed
- `C`: toggle orbit/free camera
- `W A S D`: move in free-camera mode
- `Q / E`: move down/up in free-camera mode
- `Tab / Shift`: next/previous parameter
- `Arrow keys`: adjust selected parameter
- `1` to `9`: jump to one of the first parameters
- `F`: recenter camera
- `R`: reset parameters
- `P`: print all parameters
- `H`: print help
- `Space` or `F1`: toggle HUD
- `Esc`: quit

## Assets

### HDR Environments

HDR files are loaded from `hdrs/`. The renderer expects Radiance `.hdr` files.
The helper script `src/skinny/fetch_hdrs.py` documents the curated Poly Haven
HDRIs used for portrait/skin lighting.

### Head Models

The analytic fallback head is an SDF head based on Loomis-style proportions.
Additional mesh heads are discovered from `heads/`:

- each immediate subdirectory containing an `.obj` becomes one model
- loose top-level `.obj` files are also loaded
- texture maps are attached by filename keywords:
  - normal: `normal`, `nrm`, `nor`
  - roughness: `rough`, `roughness`
  - displacement: `displacement`, `displace`, `disp`, `height`, `bump`

Displacement can be baked into the mesh after midpoint subdivision. Normal maps
can be baked into vertex normals when the mesh is dense enough, otherwise they
are sampled at shading time.

### Tattoos

Tattoo images are loaded from `tattoos/` when present. The alpha channel drives
ink density, and RGB drives the pigment color contribution in the dermis.

## Rendering Modes

### Scattering

The `Scattering` control selects which subsurface estimator is active:

- `BSSRDF + Volume`
- `BSSRDF only`
- `Volume only`
- `Off`

The BSSRDF path provides a smooth layered diffuse response. The volume path
performs delta tracking through the layered medium and can add near-surface
single/multi-scatter detail.

### Sampling

The `Sampling` control selects the integration strategy:

- `Path tracing`: original camera-path estimator
- `MIS`: surface environment samples combine BSDF-side and light-side samples
  with a power heuristic
- `Bidirectional`: MIS plus explicit volume light-connection samples
- `Stored BDPT`: builds a light-side surface vertex per shader sample and
  connects the camera hit to it with a visibility-tested geometry term

The current stored BDPT implementation is bounded to the existing single
compute dispatch. It stores a light-side vertex locally in the shader invocation
rather than using a separate global light-path pass and reusable vertex buffer.

### Furnace Mode

`Furnace mode` swaps the scene to a unit sphere and forces the environment to
unit-white radiance. Under this test a passive BRDF/BSSRDF should not emit more
than `1.0` radiance per channel. Pixels that exceed the tolerance are tinted
pink, which makes energy violations visible.

Use this mode after changing scattering, MIS, or BDPT logic.

## Development Checks

Compile Python files:

```powershell
.\Scripts\python -m py_compile src\skinny\app.py src\skinny\renderer.py
```

Compile the main shader manually:

```powershell
slangc src\skinny\shaders\main_pass.slang -target spirv -entry mainImage -stage compute -o src\skinny\shaders\main_pass.spv -I src\skinny\shaders
```

Run tests if development dependencies are installed:

```powershell
.\Scripts\python -m pytest
```

## Implementation Map

- `src/skinny/app.py`: GLFW window, input handling, parameter list, settings
- `src/skinny/control_panel.py`: Tk control panel generated from `ALL_PARAMS`
- `src/skinny/renderer.py`: Vulkan resources, uniforms, environment/mesh/tattoo upload
- `src/skinny/vk_compute.py`: compute pipeline, descriptor layout, GPU image/buffer helpers
- `src/skinny/environment.py`: built-in and HDR environment loading
- `src/skinny/mesh.py`: OBJ loading, normalization, subdivision, displacement, BVH construction
- `src/skinny/shaders/main_pass.slang`: primary camera path, composition, MIS, stored-BDPT connection
- `src/skinny/shaders/skin_bssrdf.slang`: layered skin optics, BSSRDF, GGX specular
- `src/skinny/shaders/volume_render.slang`: delta-tracked volume skin transport
- `src/skinny/shaders/mesh_head.slang`: BVH traversal and ray/triangle intersection
- `src/skinny/shaders/sdf_head.slang`: analytic SDF head
- `src/skinny/shaders/environment.slang`: environment lookup and furnace fallback
- `src/skinny/shaders/detail.slang`: statistical pores and vellus hair sheen

## Papers And References

The renderer uses or follows these papers and technical references:

| Area | Implementation | Reference |
|---|---|---|
| Subsurface transport model | `skin_bssrdf.slang` | Jensen, Marschner, Levoy, Hanrahan, "A Practical Model for Subsurface Light Transport", SIGGRAPH 2001 |
| Quantized diffusion | `skin_bssrdf.slang` | d'Eon and Irving, "A Quantized-Diffusion Model for Rendering Translucent Materials", SIGGRAPH/TOG 2011 |
| Normalized diffusion profile | `skin_bssrdf.slang` | Christensen and Burley, "Approximate Reflectance Profiles for Efficient Subsurface Scattering", Disney/SIGGRAPH 2015 |
| Human skin optical parameters | `skin_bssrdf.slang`, `presets.py` | Donner and Jensen, "A Spectral BSSRDF for Shading Human Skin", EGSR 2006 |
| Real-time skin pipeline | `renderer.py`, `mesh_head.slang`, `sdf_head.slang` | d'Eon and Luebke, "Advanced Techniques for Realistic Real-Time Skin Rendering", GPU Gems 3 Chapter 14, 2007 |
| Multiple importance sampling | `common.slang`, `main_pass.slang`, `volume_render.slang` | Veach, "Robust Monte Carlo Methods for Light Transport Simulation", PhD thesis, 1997 |
| Bidirectional path tracing | `main_pass.slang` | Veach and Guibas, "Bidirectional Estimators for Light Transport", 1995 |
| Bidirectional path tracing | `main_pass.slang` | Lafortune and Willems, "Bi-Directional Path Tracing", 1993 |
| GGX microfacet specular | `skin_bssrdf.slang` | Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007 |
| Fresnel approximation | `skin_bssrdf.slang`, `detail.slang` | Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering", 1994 |
| Henyey-Greenstein phase function | `volume_render.slang` | Henyey and Greenstein, "Diffuse Radiation in the Galaxy", 1941 |
| Delta/Woodcock tracking | `volume_render.slang` | Woodcock et al., "Techniques Used in the GEM Code for Monte Carlo Neutronics Calculations", 1965 |
| Ray/triangle intersection | `mesh_head.slang` | Moeller and Trumbore, "Fast, Minimum Storage Ray/Triangle Intersection", 1997 |

Some supporting pieces, such as ACES-style filmic tone mapping, PCG hashing,
median-split BVH construction, Worley noise, Box-Muller sampling, and
Loomis-style head proportions, are standard techniques used as implementation
building blocks rather than primary skin-rendering papers.
