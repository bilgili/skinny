# Skinny

> **Note:** This project is developed with [Claude Code](https://claude.ai/claude-code)
> and serves as a testbed for experimenting with new rendering algorithms.
> The codebase evolves rapidly and stability is not guaranteed.

Skinny is a physically based GPU renderer with Vulkan and native Metal compute
backends compiled from the same Slang shader sources. It started as a human skin
rendering testbed -- and retains first-class skin support -- but the core
pipeline handles arbitrary MaterialX materials, OpenUSD scenes, ray-traced
geometry, image-based lighting, microfacet specular, and energy-conservation
checks.

**[docs/README.md](docs/README.md) indexes every reference document**, one line
each. The entry points: renderer internals in [Architecture.md](docs/Architecture.md);
what the renderer can be told to do — backend, compatibility matrix, sampling
modes — in [RenderingModes.md](docs/RenderingModes.md); skin-specific rendering
(three-layer optics, scattering modes, MaterialX skin nodedefs, head geometry,
presets, tattoos) in [SkinRendering.md](docs/SkinRendering.md); the public
Python API in [PythonAPI.md](docs/PythonAPI.md).

## Gallery

<p align="center">
  <img src="docs/GUI.png" alt="Skinny GUI — Qt desktop app with viewport + sidebar" width="32%">
  <img src="docs/GUI2.png" alt="Skinny GUI — alternate dock layout" width="32%">
  <img src="docs/skinny.png" alt="Skinny render output — layered skin under HDR environment lighting" width="32%">
</p>
<p align="center">
  <img src="docs/skinny_lens.png" alt="Skinny render output — realistic thick-lens camera depth of field" width="48%">
</p>
<p align="center">
  These images are rendered from PBRT v4 scenes.
</p>
<p align="center">
  <img src="docs/bunny.png" alt="Skinny render output - PBRT v4 bunny scene" width="32%">
  <img src="docs/dragon.png" alt="Skinny render output - PBRT v4 dragon scene" width="32%">
  <img src="docs/bathroom.png" alt="Skinny render output - PBRT v4 bathroom scene" width="32%">
</p>
<p align="center">
  <img src="docs/crown.png" alt="Skinny render output - PBRT v4 crown scene" width="32%">
  <img src="docs/watercolor.png" alt="Skinny render output - PBRT v4 watercolor scene" width="32%">
  <img src="docs/cloud.png" alt="Skinny render output - PBRT v4 cloud scene" width="32%">
</p>

## Features

- **Layered skin rendering** -- three-layer biological optics (epidermis /
  dermis / subcutaneous), custom MaterialX skin nodedefs, scattering modes,
  Fitzpatrick presets, detail/pores, and tattoos. See
  [SkinRendering.md](docs/SkinRendering.md)
- **MaterialX nodegraph compute** -- arbitrary MaterialX nodegraphs (marble,
  wood, brass, custom standard_surface authoring) compiled per-material to
  Slang modules through `MaterialXGenSlang` plus a bindless `SamplerTexture2D`
  shim (`mtlx_gen_shim.slang`); SPIR-V cache (mtime-LRU, ~32 entries) skips
  recompilation
- **OpenUSD scene loading** -- meshes, transforms, `UsdShade.Material` bindings,
  lights (`DomeLight`, `DistantLight`, `SphereLight`, `RectLight`,
  `DiskLight`), and per-prim material assignment
- **USD light authority** -- any active authored USD light or emissive material
  exclusively owns scene lighting. Skinny adds its default DistantLight + IBL
  pair only when the active scene has no authored lighting; the pair's controls
  and synthetic scene-graph nodes are hidden otherwise
- **USD animation playback** -- time-sampled transform / camera / light tracks
  play in the viewport via a built-in transport (play/pause, scrubber, fps);
  cheap per-frame re-eval (TLAS/​light re-upload, no rebake) with a `usd` camera
  mode that follows an animated USD camera
- **UsdSkel skeletal skinning** -- skinned meshes deform per frame by linear
  blend skinning; on Vulkan a GPU skinning compute pass + GPU BVH refit keep the
  path tracer correct over deformed geometry with no readback (CPU fallback
  elsewhere)
- **USD-driven scene controls** -- a stage declares its own control panel via
  `skinny:ui:*` prims (slider / toggle / combo / color); each control binds to a
  renderer parameter, a material input, or a USD attribute and appears in a
  "Scene Controls" section across the Qt, web, and debug front-ends
- **Flat material support** -- USD prims bound to `UsdPreviewSurface`, MaterialX
  `standard_surface`, or `OpenPBR` render alongside skin materials in the same
  scene, with opacity / refraction, clear coat, and cutout-vs-alpha-blend
  masking. UsdPreviewSurface textures honour per-input channel selection,
  normal-map scale/bias (OpenGL vs DirectX Y), and wrap modes. The flat path now
  also consumes the richer `standard_surface` inputs **colored glass**
  (`transmission_color` tints the refracted delta-transmission branch),
  **tinted speculars** (`specular_color` scales the GGX spec response), and
  **Oren-Nayar diffuse** (`diffuse_roughness` drives a rough-diffuse response,
  `0` ⇒ exact Lambert) — all weight/response-only, so absent inputs reproduce the
  prior render exactly
- **Python-authored materials** -- SlangPile `python_materials/*.py` compile to
  GPU `IMaterial` structs, dispatched as material type 3; editable live in the
  Qt material editor
- **MIS path tracing** -- unified bounce loop with per-bounce NEE, Russian
  roulette, and sphere-light MIS; materials provide BSDF sample/evaluate
- **Environment importance sampling** -- equirect HDR sampled by a
  sin θ-weighted 2D distribution for env NEE + MIS, with VNDF GGX specular
  sampling, killing specular environment fireflies
- **Bidirectional path tracing** -- BDPT integrator with light-tracer splatting
  for caustics on flat materials; connections evaluate the real
  `standard_surface` BSDF, env importance sampling matched to the path tracer;
  Veach §10 MIS weighting
- **Stochastic Progressive Photon Mapping** -- caustic-efficient SPPM integrator
  (`--integrator sppm`): per-pass eye → spatial-hash grid → photon → radius/flux
  update, with the per-pixel estimator persisting across accumulation frames;
  wavefront-only, flat materials, supported on both Vulkan and native Metal. See
  [docs/PhotonMapping.md](docs/PhotonMapping.md)
- **Furnace mode** -- unit-sphere + white-environment energy conservation test;
  violations tinted pink; supports per-material furnace probes
- **Realistic lens camera** -- pinhole + PBRT-v3 thick-lens stack
  (`shaders/cameras/`); per-pixel exit-pupil bounding (`lens_optics.py`) so
  small f-stops don't shrink the rendered area; on-screen focus / vignette
  overlays (`L`, `V`)
- **Camera debug viewport** -- second window (or embedded dock) rendering
  frustum, lens rings, focus / DOF planes, mesh wireframes, AABBs, ground
  grid, and a camera-body glyph
- **Transform gizmo** -- screen-space gizmo (`gizmo.py`) for the selected mesh
  instance: rotate rings or translate arrows, in world or local space, cycled
  with `Space` (a `W`/`L` glyph hints the coordinate space); line list
  composited by `main_pass.slang`
- **Scene graph editing** -- add referenced USD models or authored Distant,
  Sphere, Dome, Rect, and Disk lights from the Qt or web scene-graph panel;
  transform and delete prims, then persist the non-destructive edit layer with
  **Save edits**
- **Exposure + tonemapping** -- EV-stop exposure and selectable tonemap
  operator (ACES filmic / Reinhard / Hable / linear) as post-process knobs that
  don't reset accumulation; HDR/EXR screenshot export
- **BVH caching** -- zstd-compressed mesh/BVH data cached to disk
  (`~/.skinny/mesh_cache/`) for fast reload
- **Qt desktop UI** -- single-window `skinny-gui` (PySide6) with render
  viewport docked alongside collapsible sidebar, BXDF visualiser, MaterialX
  graph editor, scene graph inspector, and debug viewport docks; sidebar
  open/closed state persists across sessions
- **Web mode** -- Panel (HoloViz) browser UI sharing the same widget-tree
  spec as Qt, with per-user server-side rendering, H264 streaming over
  WebSocket, hardware-accelerated encoding (NVENC / QSV / AMF), and
  WebCodecs decoding in the browser
- **Multi-user sessions** -- up to 4 concurrent browser sessions, each with
  independent renderer, camera, and parameters
- **GPU selection** -- `--gpu {intel,nvidia,amd,discrete,auto}` flag on all
  entry points
- **Persistent settings** -- parameter snapshots saved and restored between
  sessions

## Requirements

- Python 3.11 or newer
- A GPU supported by one of the compute backends: Vulkan 1.2 with a current
  graphics driver, or native Metal on Apple Silicon
- Slang compiler (`slangc`) on `PATH`
- MaterialX **built from source** with the Slang code generator enabled — the
  PyPI wheel does not ship `PyMaterialXGenSlang`. See
  [MaterialX from source (required for the Slang backend)](#materialx-from-source-required-for-the-slang-backend).
- GLFW-compatible desktop environment (only required for the `skinny`
  shader-debug entry; `skinny-gui` runs on Qt and `skinny-web` is headless)

Python dependencies (`pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `numpy` | Linear algebra, mesh processing |
| `slangpy` | Slang shader compilation and reflection |
| `vulkan` | Vulkan API bindings |
| `glfw` | Window creation and input (debug entry) |
| `PySide6` | Qt desktop UI |
| `Pillow` | Image I/O (HDR, textures, tattoos) |
| `imageio[freeimage]` | HDR / EXR screenshot output |
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

### Pre-commit hooks

`.pre-commit-config.yaml` runs `ruff-check` (lint, scoped to `src/`) plus
basic hygiene checks (trailing whitespace, EOF newline, YAML/TOML syntax,
merge conflicts) over the repo minus vendored build output, data/asset dirs,
generated Slang, and the openspec corpus — see the comment atop the config
for the exact exclude list. Install the `[dev]` extra (above), then enable
the git hook:

```bash
.venv/bin/pre-commit install
```

Run it manually against staged changes at any time:

```bash
.venv/bin/pre-commit run
```

If `core.hooksPath` is already customized in this repo (e.g. by another tool's
hook installer), `pre-commit install` refuses rather than clobbering it — run
`pre-commit run` manually in that case, or reconcile the hooks path first.

Verify the Slang compiler:

```powershell
slangc -version
```

### MaterialX from source (required for the Slang backend)

The MaterialX wheel published on PyPI (1.39.x) ships the GLSL, MDL, MSL, and
OSL code generators, but **not** the Slang code generator. Skinny's MaterialX
runtime (`materialx_runtime.py`) imports `PyMaterialXGenSlang` to compile both
the `ND_skinny_layered_skin_stack` skin shader and arbitrary nodegraphs
(`standard_surface`, marble, wood, brass, etc.) into Slang modules at runtime.
Without the Slang generator the renderer fails at import time with
`ImportError: cannot import name 'PyMaterialXGenSlang'`.

**On a supported platform you don't need to do anything below** — `pyproject.toml`
already pulls prebuilt wheels for both packages as base (non-extra) dependencies:

- `materialx-python-standalone` — MaterialX built with `MATERIALX_BUILD_GEN_SLANG=ON`,
  providing `import MaterialX` + `PyMaterialXGenSlang`.
- `openusd-materialx` — OpenUSD (v26.05) built with the `usdMtlx` plugin, providing
  `import pxr`.

Both are published as direct-URL GitHub Release wheels from
[`bilgili/openusd-materialx`](https://github.com/bilgili/openusd-materialx) (not
PyPI — the PyPI `MaterialX`/`usd-core` wheels lack GenSlang/usdMtlx), one entry per
`(python_version, sys_platform, platform_machine)` combination the release ships —
Python 3.12/3.13/3.14 × (`darwin`/`arm64` [Apple Silicon], `linux`/`x86_64`,
`win32`/`AMD64`), matching wheel filename tags `cp312`-`cp314` ×
`macosx_26_0_arm64`/`linux_x86_64`/`win_amd64`. `pip` resolves the matching
entry automatically from the environment markers, so a plain `pip install -e .`
(or `-e ".[dev]"`) installs the Slang- and usdMtlx-capable builds directly —
no compiler, no CMake.

The manual from-source build below is only needed if your platform isn't in that
matrix (e.g. Linux aarch64, Intel macOS) or you need a newer MaterialX than the
pinned `v1.0.11` release provides.

Build and install MaterialX with Python bindings + Slang generator enabled:

```bash
# 1. Clone upstream MaterialX (>= 1.39)
git clone --depth 1 https://github.com/AcademySoftwareFoundation/MaterialX.git
cd MaterialX

# 2. Configure with Python bindings and the Slang generator enabled.
#    Point MATERIALX_PYTHON_EXECUTABLE at the same interpreter you will use
#    to run skinny (your venv's python), so the bindings match its ABI.
cmake -S . -B build \
  -DMATERIALX_BUILD_PYTHON=ON \
  -DMATERIALX_BUILD_GEN_SLANG=ON \
  -DMATERIALX_PYTHON_EXECUTABLE="$(pwd)/../.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install"

# 3. Build and install
cmake --build build --parallel
cmake --install build

# 4. Install the Python package into skinny's venv. The build emits a
#    standard setup.py / pyproject under build/python (or install/python
#    depending on the version) — install it in place, do NOT `pip install
#    MaterialX` afterwards or the wheel will overwrite the source build.
../.venv/bin/pip install ./install/python
```

Verify the Slang generator is available:

```bash
.venv/bin/python -c "from MaterialX import PyMaterialXGenSlang; print(PyMaterialXGenSlang.__file__)"
```

Notes:

- On Windows use the same CMake invocation with the Visual Studio generator
  (`cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ...`) and install
  with `cmake --build build --config Release --target install`.
- If you previously installed the PyPI wheel into the venv, uninstall it first
  (`pip uninstall MaterialX`) before installing the from-source build.
- Keep the MaterialX checkout around — re-installing the venv requires
  re-running step 4 against the same `install/python` tree.

## Running

Three entry points share the renderer core:

| Command | UI | Use case |
|---------|----|----|
| `skinny-gui` | Qt (PySide6) | Primary desktop app — viewport dock, sidebar, tool docks |
| `skinny-web` | Panel + browser | Multi-user H264 streaming over WebSocket |
| `skinny` | GLFW + keyboard | Headless shader-debug loop (no widgets) |

### Qt desktop (`skinny-gui`)

```powershell
.\Scripts\skinny-gui.exe
.\Scripts\skinny-gui.exe assets/demo_head.usda
.\Scripts\skinny-gui.exe --gpu nvidia assets/Usd-Mtlx-Example/scene.usda
```

Layout:

- Central dock: render viewport (mouse drag = orbit, right-drag = pan,
  scroll = zoom)
- Left dock: collapsible parameter sidebar (Render / ReSTIR / Skin / Detail /
  Materials sections, generated from the shared widget-tree spec)
- View menu: BXDF visualiser, MaterialX graph editor, scene graph
  inspector, camera debug viewport (each a `QDockWidget`)

The scene-graph inspector's **Add light** menu authors a DistantLight,
SphereLight, DomeLight, RectLight, or DiskLight below the selected Xform/Scope
(or `/World`), assigns a unique name and explicit defaults, and refreshes the
render immediately. Adding the first authored light activates USD lighting
authority and removes both built-in fallback lights. DomeLight starts without
an HDR; select the new node and use its texture property to choose one.

The render viewport owns the renderer session on a Qt worker thread: GPU context
creation, `Renderer` construction, frame rendering, online-training ticks, and
cleanup all happen there. The GUI builds controls against a lightweight proxy
that reads immutable snapshots and posts renderer mutations (camera input,
zoom/focus toggles, scene loads, parameter edits, and render-target resize)
through a command queue. The main Qt thread paints the latest emitted frame and
stays responsive while the renderer is accumulating. All five View-menu tool
docks are proxy-backed and live: they read worker-built snapshots and post
mutations through the same queue.

Any `.usda` / `.usdc` / `.usdz` file with MaterialX-bound or
`UsdPreviewSurface`-bound materials will load. The renderer has been tested
with the [Usd-Mtlx-Example](https://github.com/pablode/Usd-Mtlx-Example)
repository.

### Web mode (`skinny-web`)

```powershell
.\Scripts\skinny-web.exe --port 8080
.\Scripts\skinny-web.exe --port 8080 --usd assets/Usd-Mtlx-Example/scene.usda
```

Open `http://localhost:8080/skinny` in a browser. Each tab gets an independent
renderer session with its own camera and parameters. Video is H264-encoded
server-side and decoded via WebCodecs in the browser.

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Server port |
| `--gpu` | auto | GPU selection: `intel`, `nvidia`, `amd`, `discrete`, `auto` |
| `--max-sessions` | 4 | Max concurrent browser sessions |
| `--usd` | — | Path to USD scene (alternative to positional arg) |
| `--usdMtlx` | off | Use USD's built-in usdMtlx plugin instead of MaterialX API fallback |

### GLFW shader-debug entry (`skinny`)

```powershell
.\Scripts\skinny.exe assets/demo_head.usda
```

Keyboard-driven loop with no Qt overhead. Useful for fast iteration on the
render pipeline or Slang code where the Qt event loop gets in the way.

### MCP scene control (`--mcp`)

Let an MCP client (Claude, or any other) inspect and edit the scene of a
**running** `skinny` or `skinny-gui` session while you watch the viewport.

```bash
pip install -e ".[mcp]"          # optional extra; --mcp errors without it
.venv/bin/skinny-gui --backend metal --mcp
```

Startup prints the registration command:

```bash
claude mcp add --transport http skinny http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $(cat ~/.skinny/mcp_token)"
```

13 tools, addressed by USD prim path:

| Tool | Purpose |
|------|---------|
| `scene_list(path, depth, kind)` | Tree structure — no property values. Filter by `kind` (`material`, `light_dir`, `light_sphere`, `light_env`, `instance`, `renderer_camera`) |
| `scene_get(path)` | One node's properties, with editable flags and bounds |
| `scene_set(path, property, value)` | Write one property |
| `scene_create(force)` | Start a fresh empty editable scene (a bare `/World`) so edits work with no scene loaded; refuses if one is already loaded unless `force=true` |
| `scene_add_model(usd_path, name, parent, translate/rotate_euler_deg/scale or matrix)` | Reference a USD file into the scene |
| `scene_import_glb(glb_path, name, parent, out_dir, overwrite, transform)` | Convert a GLB to USD (built-in pure-Python converter, works on macOS/Linux/Windows) and reference it in one call — the drop-in path for image-to-3D output (e.g. TRELLIS.2). Refuses out-of-scope glTF (Draco, skinning, animation) by name |
| `scene_add_primitive(type, color, roughness, metallic, material, name, parent, transform)` | Add a Sphere/Cube/Cylinder/Cone/Capsule/Plane with its own editable material, or bind `material` (a preset/template name, or an existing `/Materials/...` path) instead of the inline `color`/`roughness`/`metallic` seed |
| `scene_add_light(light_type, intensity, color, name, parent, transform)` | Add a DistantLight/SphereLight/DomeLight/RectLight/DiskLight |
| `material_list()` | Discovery: curated preset catalog (with editable inputs), the `preview`/`standard_surface` parametric schemas, the nodegraph node whitelist, and the procedural template schemas — renderer-free, everything needed to build a `scene_add_material` spec |
| `scene_add_material(spec, name)` | Create a `/Materials` holder from a curated preset, a parametric UsdPreviewSurface/standard_surface, or a procedural template; not live (not rendered, loaded, or editable) until bound |
| `scene_bind_material(prim_path, material_path)` | Bind (or rebind) a material to a geometry prim — the moment a material becomes live; replaces any file-authored binding |
| `scene_remove(path)` | Deactivate a node (non-destructive) |
| `scene_save(path)` | Write the USD edit layer — structural edits only, see below |
| `scene_job_status(job_id)` | Poll a structural tool that returned `{"status": "pending", ...}` |

Property edits take the same code path as the equivalent Scene Graph dock
edit, so both behave identically. (One exemption: the dock's *file-chooser*
flows for HDR and lens files keep their own dialog and async error handling;
the routing decision is still shared, so a client reaches the same renderer
verb.) Structural adds author into the same non-destructive USD edit layer
the dock's Add model / Add light buttons do. The docks stay live and enabled
while a client is connected; concurrent edits are last-write-wins, and every
result reports the current scene and material versions.

**Structural tools.** A model/primitive/light add waits briefly (~2s) for the
render thread and returns its result directly; a bigger add (a large
referenced scene) instead returns `{"status": "pending", "job_id": ...}` to
poll via `scene_job_status` rather than being cancelled — a cancelled-but-
already-running add would otherwise leave you unsure whether the scene
changed. `scene_add_primitive` always authors its own bound material (never a
bare gprim), so `color`/`roughness`/`metallic` — and later `scene_set` edits
on the same material node — actually take effect. `scene_remove` refuses the
root and the renderer's synthesized `/Skinny/*` nodes.

**Material authoring.** `scene_add_material` accepts exactly one spec form —
`{"preset": name}` (curated corpus under `assets/Usd-Mtlx-Example/materials/`,
resolved server-side by name, never a client-supplied path), `{"model":
"preview"|"standard_surface", "params": {...}, "graph": {...}?}` (flat
parameters, optionally an explicit MaterialX nodegraph on `standard_surface`),
or `{"template": name, "params": {...}}` (a server-owned procedural recipe —
`noise` or `marble_veins` — that expands to a `standard_surface` graph). A raw
nodegraph's node types are restricted to a generator-proven whitelist
(`fractal3d`, `noise2d`, `noise3d`, `position`, `texcoord`, `mix`, `multiply`,
`add`, `subtract`, `sin`, `power`, `dotproduct`, `ramplr`, `ramptb`) — no
`checker`/`checkerboard` node in v1, so no checker template either. Every
spec is validated and, for a synthesized document, run through a GPU-free
Slang generator dry-run before any prim or file is created; a rejected spec
never touches the stage. Adding the same preset twice returns the existing
`/Materials` holder instead of creating a duplicate (curated documents have
fixed element names); synthesized/template materials are never deduped.

The result of `scene_add_material` always reports `"live": false` —
participation is binding-driven, so a created material is loaded, rendered,
and exposes its editable properties only once `scene_bind_material` or
`scene_add_primitive(material=...)` binds it. A synthesized material's
*first* bind (not the add) changes the scene's graph-set signature and
rebuilds the render pipeline, so it degrades to a pollable job
(`scene_job_status`) more often than a plain structural add. On `scene_save`,
all curated presets keep absolute references into `assets/` rather than being
copied beside the saved scene (copying a texture-bearing doc such as
`wood_tiled` without its textures would silently break it); synthesized
documents (textureless by the v1 whitelist) are always copied alongside it.

**Filesystem allowlist.** Every path a structural tool touches — a model
reference, a save destination, an asset-typed `scene_set` write — must resolve
inside `--mcp-roots dir[,dir...]` / `SKINNY_MCP_ROOTS` (default: the platform
temp directories and the current working directory). For `scene_add_model` the
check also follows the reference: any USD layer or asset the referenced file
newly pulls in must stay inside the roots too, or the add is rolled back. This
guards against a misdirected tool call — the MCP client already has full
filesystem access on this machine, so it is not a sandbox against an
adversarial one.

**Security.** Off by default. Binds `127.0.0.1` only (the port option takes a
port number, never a host). Requests carrying an `Origin` header are refused and
`Host` is validated, so a page in your browser cannot drive the renderer. Every
request needs the bearer token from `~/.skinny/mcp_token` (overridable with
`SKINNY_MCP_TOKEN`). On POSIX the file is created mode `0600` and re-validated
on every read (no-follow open, owner and mode checked on the same descriptor).
**On Windows those checks do not apply** — the primitives they use do not exist
there, so the token is only as protected as your user profile directory. Refusing `Origin` also blocks
browser-hosted MCP clients such as the MCP Inspector — a deliberate trade.

`--mcp-port` overrides the default `8765`. If the port is already bound (a second
session), the renderer starts normally with MCP disabled rather than exiting.

**v1 limits:** no image tool, so the client edits without seeing the result —
watch the viewport. `scene_save` captures structural edits (adds, removes,
transforms) but **not** `scene_set` property edits, which mutate in-memory
render state without authoring to USD — the same partial-save behavior the
dock's own Save edits button has.

### Headless rendering (`skinny-render`)

Render a USD scene to a file (or frame sequence) with no window:

```bash
# Single image — path tracer, 256 samples, PNG
skinny-render assets/cornell_box_sphere.usda -o out/cornell.png \
    --width 1920 --height 1080 --samples 256

# Animation over USD timecodes → PNG frame sequence
skinny-render assets/animated_scene.usda --animate \
    --frames 1:96:1 --outdir out/frames --samples 64 --ext png
```

`skinny.headless.HeadlessRenderer` holds the GPU context across calls so you
can open a `Usd.Stage`, mutate it per frame (move prims, change camera xforms,
set USD time), and call `r.render_to_array(stage)` or `r.render_scene(stage,
path)` for each frame — the pipeline is compiled only once.

See `examples/` for minimal demo scripts. Full Python API reference (headless
interface, `Renderer`, parameters, scene loading, presets) is in
[PythonAPI.md](docs/PythonAPI.md); `skinny.headless` internals are in
[Architecture.md](docs/Architecture.md).

### Importing pbrt v4 scenes (`skinny-import-pbrt`)

Convert a [pbrt v4](https://pbrt.org) text scene into a skinny-loadable USD stage:

```bash
skinny-import-pbrt scene.pbrt -o scene.usda
skinny-render scene.usda -o out/scene.png --samples 256
```

The importer covers triangle/ply/sphere geometry + instancing, the common pbrt
materials/lights, the `perspective` camera, spectrum→RGB, and homogeneous
media/subsurface (best-effort), emitting an exact/approx/skipped report. Image
parity against pbrt v4 is validated by a relMSE/FLIP gate over a checked-in
corpus. See [PbrtImport.md](docs/PbrtImport.md) for the full feature/parity
matrix.

Pass `-mtlx` / `--materialx` to additionally write a portable MaterialX sidecar
(`scene.mtlx`, referenced from the stage) carrying the materials as
`standard_surface` networks:

```bash
skinny-import-pbrt scene.pbrt -o scene.usda -mtlx
```

The sidecar makes the export MaterialX-native for other MaterialX-aware tools and
captures the richer pbrt parameters (`transmission`/`transmission_color`,
separate `coat`/`coat_IOR`, `subsurface_radius`, `specular_anisotropy` from
`uroughness`/`vroughness`, `thin_walled`) that UsdPreviewSurface cannot express.
The production integrators consume the `FlatMaterial` subset of either export, so
for diffuse / conductor / dielectric materials `-mtlx` and the UsdPreviewSurface
output stay pixel-identical. The flat path now also reads
`transmission_color` (colored glass), `specular_color` (tinted speculars), and
`diffuse_roughness` (Oren-Nayar diffuse) from these richer slots, so those
inputs render rather than being dropped (Stage-2 Tier A); `specular_anisotropy`
and rough-glass transmission remain future work.

A pbrt `Material "subsurface"` now imports as a **volumetric subsurface
material** (Stage-2 Ch5) — a smooth dielectric boundary (`eta`) wrapping a
homogeneous interior medium (`σ_a`, `σ_s`, Henyey-Greenstein `g`), transported
by a delta-tracked (Woodcock / null-collision) interior random walk — rather
than being lowered to clear glass. Coefficients follow pbrt precedence (explicit
`sigma_a`/`sigma_s` × `scale` → named preset → `reflectance` + `mfp` via the
Jensen inversion; the `-mtlx` `standard_surface` maps identically). It runs in
**both execution modes** (megakernel + wavefront) and **both backends** (Vulkan
+ native Metal), and is energy-conserving (furnace `σ_a → 0` ≈ unity). pbrt
uses a dipole here, so the random walk is *milky*-matching rather than
bit-parity. Limitation: the walk lights from a single distant light + the
environment; area/emissive lights inside the medium, heterogeneous / NanoVDB
grids, and free-standing `MediumInterface` media are deliberate follow-ups.

### Mesh heads (legacy)

Place `.obj` files (with optional normal/roughness/displacement maps) in
`heads/<name>/` directories. They are discovered automatically at startup.

## Controls

Keyboard and mouse controls are shown in the on-screen HUD when running the
GLFW debug entry. Qt and web entries use widget-driven input plus the
shortcuts below forwarded to the viewport.

| Input | Action |
|-------|--------|
| Left drag | Orbit camera (orbit mode) / look around (free mode) |
| Right drag | Pan orbit target |
| Scroll | Zoom (orbit) / adjust speed (free) |
| `C` | Toggle orbit / free camera |
| `W A S D` | Move in free-camera mode |
| `Q / E` | Move down / up in free-camera mode |
| `Tab / Shift+Tab` | Next / previous parameter (debug entry) |
| Arrow keys | Adjust selected parameter (debug entry) |
| `1`--`9` | Jump to parameter (debug entry) |
| `F` | Recenter camera |
| `R` | Reset parameters |
| `P` | Print all parameters |
| `H` | Print help |
| `L` | Toggle lens focus overlay |
| `V` | Toggle lens vignette debug (green=ray valid, red=clipped) |
| `Z` | Arm zoom rectangle (drag in viewport, release to apply) |
| `X` | Reset zoom rectangle |
| `F2` | Toggle camera debug viewport dock / window |
| `Space` | Cycle transform gizmo mode (rotate/translate × world/local) |
| `F1` | Toggle HUD |
| `Esc` | Quit |

### Camera Debug viewport

Its own key map, identical in the GLFW window and the `skinny-gui` dock (the
recorded set is asserted by `tests/test_qt_debug_viewport_dock.py`). `W A S D
Q E` move the debug camera in free mode; `D` also toggles the depth-of-field
planes on press, served from a separate channel so a held strafe does not flip
it.

| Input | Action |
|-------|--------|
| `C` | Toggle orbit / free debug camera |
| `F` | Reset debug camera |
| `W A S D Q E` | Move (free mode) |
| `M` | Toggle mesh wireframes (AABBs invert) |
| `G` | Toggle ground grid |
| `P` | Toggle focus plane |
| `D` | Toggle depth-of-field planes |
| `I` | Toggle render-area outline |
| `O` | Toggle orthographic projection |
| `T` / `B` / `L` | Top / back / left view |
| `Space` | Toggle HUD |
| `Esc` | Close (GLFW window only — the Qt dock closes from its title bar) |

## Assets

### HDR Environments

Radiance `.hdr` (and discovered sibling `.exr` / `.pfm`) files in `hdrs/`. The
helper script `src/skinny/fetch_hdrs.py` documents the curated Poly Haven
HDRIs used for portrait/skin lighting. The Qt and web sidebars expose a
"Load HDR" picker that scans the chosen file's directory for additional
formats.

### Head Models

Head geometry (analytic SDF head + discovered `heads/*.obj` mesh heads with
detail maps) is documented in [SkinRendering.md](docs/SkinRendering.md).

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

## Rendering Modes

The GPU backend, the render resolution, the compatibility matrix, the sampling
modes (integrators, proposals, reuse), and furnace mode are documented in
[docs/RenderingModes.md](docs/RenderingModes.md).

## Implementation Map

The per-file map of the Python package, the shader tree, and the tests is in
[docs/ImplementationMap.md](docs/ImplementationMap.md).

## Papers and References

| Area | Files | Reference |
|------|-------|-----------|
| MIS | `samplers/mis_combine.slang`, `integrators/bdpt.slang` | Veach, "Robust Monte Carlo Methods for Light Transport Simulation", PhD thesis, 1997 |
| Bidirectional path tracing | `integrators/bdpt.slang` | Veach and Guibas, "Bidirectional Estimators for Light Transport", 1995 |
| Bidirectional path tracing | `integrators/bdpt.slang` | Lafortune and Willems, "Bi-Directional Path Tracing", 1993 |
| Stochastic progressive photon mapping | `integrators/wavefront_sppm.slang`, `integrators/sppm_state.slang` | Hachisuka and Jensen, "Stochastic Progressive Photon Mapping", SIGGRAPH Asia 2009 |
| GGX microfacet | `materials/flat/flat_shading.slang` | Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007 |
| Fresnel approximation | `materials/flat/flat_shading.slang` | Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering", 1994 |
| Realistic camera | `lens_optics.py`, `shaders/cameras/thick_lens.slang` | Pharr, Jakob, Humphreys, *Physically Based Rendering 3e*, Ch. 6 |

Skin-, subsurface-, volume-, and head-geometry references live in
[SkinRendering.md](docs/SkinRendering.md). Supporting techniques (ACES tone mapping,
PCG hashing, median-split BVH, Worley noise, Box-Muller sampling) are standard
implementation building blocks.

## Testing

The test suite covers shader math, sampling, lighting, volume rendering,
struct layout, MaterialX closures, MaterialX nodegraph compilation, skin
optics, headless rendering, SlangPile transpilation, the shared widget-tree
spec, and the web application. Tests are organized by subsystem with Slang
harness shaders in `tests/harnesses/` and reference kernels in
`tests/kernels/`.

```powershell
.\Scripts\python -m pytest
```

GPU-dependent tests are marked `@pytest.mark.gpu`; statistical Monte Carlo
tests are marked `@pytest.mark.slow`; SlangPile-specific tests are marked
`@pytest.mark.slangpile`.

## Development

Compile Python:

```powershell
.\Scripts\python -m py_compile src\skinny\app.py src\skinny\renderer.py
```

Compile main shader:

```powershell
slangc src\skinny\shaders\main_pass.slang -target spirv -entry mainImage -stage compute -o src\skinny\shaders\main_pass.spv -I src\skinny\shaders
```

## License

MIT
