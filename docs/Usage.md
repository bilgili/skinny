# Skinny — Running the Renderer

This document covers how to drive skinny: the interactive front ends, the
command-line flags, headless rendering, and importing pbrt v4 scenes.

For which combinations of integrator, execution mode, and backend are in the
envelope see [RenderingModes.md](RenderingModes.md). For installation see
[Install.md](Install.md). For keyboard and mouse bindings see
[Controls.md](Controls.md).

---

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
[PythonAPI.md](PythonAPI.md); `skinny.headless` internals are in
[FrontEnds.md](FrontEnds.md).

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
corpus. See [PbrtImport.md](PbrtImport.md) for the full feature/parity
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
