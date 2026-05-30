# Skinny — Architecture

Skinny is a single-kernel Vulkan compute path tracer specialised for
physically-based human skin rendering. It combines a three-layer biological
skin model (epidermis / dermis / subcutaneous) with a full MaterialX-based
standard_surface closure tree plus arbitrary MaterialX nodegraphs compiled
to per-material Slang modules — all driven from one
`[numthreads(8,8,1)]` compute dispatch.

The skin-specific subsystems (three-layer biological model, §1–§6 estimator
chain, volume transport, head geometry, and MaterialX skin codegen) are
documented in [SkinRendering.md](SkinRendering.md). This file covers the generic
renderer architecture.

---

## High-Level Pipeline

Three entry points share the same renderer core:

```
GLFW debug (skinny)            Qt desktop (skinny-gui)         Web (skinny-web)
┌─────────────┐                ┌────────────────────┐          ┌────────────────────┐
│ GLFW window │                │ QMainWindow        │          │ Browser tab        │
│ + keyboard  │                │  ├ RenderViewport  │          │ (Panel widgets +   │
│ params      │                │  ├ sidebar (spec)  │          │  WebCodecs canvas) │
└──────┬──────┘                │  └ tool docks      │          └─────────┬──────────┘
       │                       └─────────┬──────────┘                    │
       │                                 │                               │
       ▼                                 ▼                               ▼
                       ┌─────────────────────────────────────┐    ┌──────────────────┐
                       │ Renderer.py                          │    │ SkinnySession    │
                       │  packs UBO, uploads bufs,            │◀───│ per-user wrapper │
                       │  runs MaterialX codegen + cache      │    │ + H264 encoder   │
                       └────────┬─────────────────────────────┘    └────────┬─────────┘
                                │                                           │
                                ▼                                           ▼
                       ┌──────────────────┐                          ┌──────────────────┐
                       │ Vulkan Compute   │                          │ Vulkan Compute   │
                       │  Dispatch        │                          │  (headless)      │
                       │  → swapchain /   │                          │  → readback      │
                       │     offscreen    │                          │  → H264 encode   │
                       └──────────────────┘                          │  → WebSocket     │
                                                                     └──────────────────┘
```

`skinny-gui` and `skinny-web` share a **single widget-tree spec**
(`ui/spec.py` + `ui/build_app_ui.py`). The Qt backend
(`ui/qt/backend.py`) and the Panel backend (`ui/panel/backend.py`) walk
the same tree and instantiate their own widgets, so adding a slider in
the spec lights it up in both UIs.

### Per-Frame Render Loop (Qt desktop)

1. `MainWindow.QTimer` ticks at the swap interval → `renderer.update(dt)`
   → `renderer.render_offscreen()` → `RenderViewport` blits the latest
   image with `QImage::QImage(..., RGBA8888)`.
2. `renderer.update(dt)` detects dirty state (camera, params, env, scene
   graph), reuploads affected buffers, resets accumulation on change.
3. `renderer.render_offscreen()` packs FrameConstants + SkinParams + light
   + per-material data into the UBO, updates descriptor sets, dispatches
   `ceil(W/8) × ceil(H/8)`, and copies the storage image into a
   readback buffer for Qt.

### Per-Frame Render Loop (GLFW debug)

1. `app.main()`: GLFW poll → `InputHandler.update(dt)` → `renderer.update(dt)`
   → `renderer.render()`.
2. `renderer.render()` presents directly via the swapchain (windowed mode).

### Per-Frame Render Loop (Web)

1. `SkinnySession._render_loop()`: background thread per session.
2. `renderer.update(dt)`: same as desktop.
3. `renderer.render_headless()`: dispatches compute, copies result to
   `ReadbackBuffer` via staging buffer, returns raw RGBA bytes.
4. `VideoEncoder.encode_h264()`: RGBA → YUV420p → H264 AVCC packets.
5. Packets pushed to `frame_queue` → Tornado `VideoStreamHandler` sends
   binary WebSocket messages → browser decodes via WebCodecs.

---

## GPU Execution Flow

```
mainImage()                                          main_pass.slang
  ├─ generateCameraRay(fc, pixel, rng)               cameras/{pinhole,thick_lens}.slang
  ├─ traceScene(fc, ray)                             scene_trace.slang
  │    ├─ furnace → unit sphere
  │    ├─ mesh    → marchHeadMesh()                  mesh_head.slang
  │    └─ SDF     → marchHead()                      sdf_head.slang
  ├─ if BDPT + flat first-hit:
  │    └─ BDPTIntegrator.estimateRadiance()          integrators/bdpt.slang
  │         ├─ eye walk (4 verts, FlatMaterial)
  │         ├─ light walk (4 verts, sphere/emissive/dir)
  │         ├─ (s,t) connections with Lambertian approx
  │         └─ light-tracer splat (s=1) → lightSplatBuffer
  ├─ else:
  │    └─ PathTracer.estimateRadiance()              integrators/path.slang
  │         ├─ cutout transparency skip loop
  │         ├─ for bounce 0..5:
  │         │    ├─ evaluateBounce(h, r, bounce, rng)
  │         │    │    ├─ FLAT → allLightsNEE + sample (FlatMaterial)
  │         │    │    │    └─ if matlx_graph_id valid → evalSceneGraph(...)
  │         │    │    ├─ SKIN → evalSkinRadiance (§1-§6)
  │         │    │    └─ DEBUG → 0.5 + 0.5·N
  │         │    ├─ Russian roulette (bounce > 0)
  │         │    └─ sphere-light MIS on BSDF ray
  ├─ NaN / inf / negative guard
  ├─ progressive accumulation (running mean)
  ├─ + BDPT light-splat mean (Q22.10 → float)
  ├─ exposure (2^EV) → tonemap (ACES / Reinhard / Hable / linear) → sRGB gamma
  ├─ per-material furnace energy-violation overlay (pink)
  ├─ rotate-gizmo line composite (binding 22)
  └─ HUD alpha composite → outputBuffer
```

`evalSceneGraph(materialId, hit, ...)` is generated per material into
`shaders/generated/` by `MaterialXGenSlang` and dispatched via tag-switch
in `flat_material.slang`. See **MaterialX Nodegraph Compute Pipeline**
below.

---

## Pluggable Interface Architecture

All interfaces live in `shaders/interfaces.slang`. Per-material furnace
probes use `effectiveFurnaceMode()` from `bindings.slang`. Dispatch
strategies are chosen to avoid existential warp serialisation on GPUs.

### ICamera

Ray generation is split out behind a tiny interface in `interfaces.slang`:

```
generateRay(fc, pixel, rng) → Ray
```

| Implementation | File | Notes |
|---|---|---|
| `PinholeCamera` | `cameras/pinhole.slang` | Standard projective ray gen |
| `ThickLensCamera` | `cameras/thick_lens.slang` | PBRT-v3 RealisticCamera port; per-pixel exit-pupil bounds packed by `lens_optics.py` |

### ISampler

```
sample(float3 wo, float2 uv) → float3
pdf(float3 wi, float3 wo)    → float
```

Tangent-space sampler (N = +Z). Callers transform world↔tangent. Sampler
state (roughness, g, etc.) is stored in struct fields. Generic parameter on
estimators — compile-time monomorphised, zero runtime cost.

| Implementation | File | Purpose |
|---|---|---|
| `GGXSampler` | `samplers/ggx.slang` | Microfacet specular importance sampling — GGX **visible** normals (VNDF, Heitz 2018/2023); weight reduces to F·G₁, killing the grazing-angle spec fireflies of classical D(H) sampling |
| `LambertSampler` | `samplers/lambert.slang` | Cosine-hemisphere diffuse sampling |
| `UniformSphereSampler` | `samplers/uniform_sphere.slang` | MIS companion sampler |
| `HenyeyGreensteinSampler` | `samplers/henyey_greenstein.slang` | Phase-function importance sampling |

MIS utilities in `samplers/mis_combine.slang`: `misPrimaryWeight<TA,TB>`,
`misCompanionWeight<TA,TB>` (power heuristic).

### IMaterial

```
sample(float3 wo, inout RNG rng)  → BSDFSample
evaluate(float3 wo, float3 wi)    → BSDFEval
```

All directions in tangent space (N=+Z). `BSDFSample` carries `wi`, `weight`
(BSDF×cos/pdf), `pdf`, `emission`, `valid`, and `transmitted` (refraction
flag). `BSDFEval` returns `response` (f×cos) and `pdf`. Tag-switch
monomorphisation in `evaluateBounce()` (`integrators/path.slang`). Never
used as existential — divergent material hits in a warp would serialise.

Skin uses its own 6-estimator chain and returns full radiance via
`BounceResult.fullRadiance`; `bsdfSample.valid = false` terminates bouncing.

| Implementation | File | Type Code |
|---|---|---|
| `SkinMaterial` | `materials/skin/skin_material.slang` | 0 (default) — self-integrating, returns full radiance |
| `FlatMaterial` | `materials/flat/flat_material.slang` | 1 — opacity/refraction, coat, spec/diff MIS, optional MaterialX graph eval |
| `DebugNormalMaterial` | `materials/debug_normal_material.slang` | 2 — normal visualisation |
| Python material | `mtlx/genslang/python_materials/*.py` → generated dispatch | 3 — SlangPile-authored `IMaterial`, id in bits 24–31, switch-dispatched by `vk_compute._emit_python_dispatcher` |

Material type encoding in `materialTypes[id]` (binding 16):
- bits 0–7: type code (0 skin, 1 flat, 2 debug-normal, 3 python)
- bits 8–9: scatter mode for skin (bit 0 = BSSRDF, bit 1 = volume)
- bit 10: per-material furnace mode (energy-conservation probe)
- bits 16–23: MaterialX graph slot (`MATERIAL_GRAPH_SHIFT`; 0 = none)
- bits 24–31: Python-material id (`MATERIAL_PYMAT_SHIFT`; index into the
  `vk_compute._emit_python_dispatcher` switch, consulted only when
  type == python)

### ILight

```
samplePoint(float3 shadingPos, float2 u) → LightSample
pdfSolidAngle(float3 shadingPos, float3 direction) → float
```

`LightSample` carries: `point`, `normal`, `radiance`, `pdfArea`, `valid`.
Delta lights (directional) set `pdfArea = 0` as a sentinel — callers skip
geometry-term conversion.

| Implementation | File | Notes |
|---|---|---|
| `SphereLightImpl` | `lights/sphere_light.slang` | Uniform surface sample, ray-sphere for `pdfSolidAngle` |
| `EmissiveTriangleLightImpl` | `lights/emissive_triangle_light.slang` | Barycentric sample, selection-weighted PDF |
| `DirectionalLightImpl` | `lights/directional_light.slang` | Delta distribution adapter |

### IIntegrator

```
estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng) → float3
```

Two implementations, selected by `fc.integratorType`:

- **`PathTracer`** — 6-bounce loop with Russian roulette, cutout
  transparency traversal, per-bounce NEE via generic `allLightsNEE<TM>()`,
  and sphere-light MIS on BSDF-sampled rays. Material dispatch in
  `evaluateBounce()` returns `BounceResult` (direct light + full radiance
  + BSDF sample with world-space direction).
- **`BDPTIntegrator`** — bidirectional path tracer (Veach §10).
  4-vertex eye + light subpaths, connections that evaluate the real
  `standard_surface` BSDF (`FlatMaterial.evaluate`), environment
  importance sampling matched to the path tracer's env NEE, and
  light-tracer splatting (s=1) for caustics via atomic adds to
  `lightSplatBuffer` (binding 21, Q22.10 fixed-point). Flat materials
  only; skin hits fall through to PathTracer.

| Implementation | File | Mode |
|---|---|---|
| `PathTracer` | `integrators/path.slang` | `INTEGRATOR_PATH` (0) |
| `BDPTIntegrator` | `integrators/bdpt.slang` | `INTEGRATOR_BDPT` (1) |

### Adding a New Material (Two-File Add)

1. Create `shaders/materials/my_material.slang` —
   `struct MyMat : IMaterial { sample(), evaluate() }` + `loadMyMat(HitInfo)`.
2. In `integrators/path.slang` — add `import materials.my_material;` and a
   `case` in `evaluateBounce()`.
3. In `renderer.py` — add `MATERIAL_TYPE_MYMAT = N` constant + packing branch.

---

## Material & Integrator Pipeline

The skin material (`SkinMaterial`, type code 0) is self-integrating: a
six-estimator chain over a three-layer biological optics model. Its internals —
the layer model, the §1–§6 estimator order, volume transport, and the MaterialX
skin codegen — are documented in [SkinRendering.md](SkinRendering.md). The flat
material and the bidirectional integrator below are general-purpose.

### Flat Material BSDF (`materials/flat/flat_material.slang`)

Implements `IMaterial` interface — provides `sample()` / `evaluate()` in
tangent space. Bounce loop and NEE are in `PathTracer`. BSDF layers:

- Opacity / refraction (Fresnel-weighted reflect/refract split). Cutout vs
  alpha-blend opacity are split: cutout discards below `opacityThreshold`,
  alpha-blend attenuates — matching UsdPreviewSurface `opacityThreshold`
  semantics
- Clear coat (GGX, coat color tinting)
- Specular / diffuse MIS split (Schlick F0, luminance-weighted probability) —
  GGX specular uses VNDF sampling (`samplers/ggx.slang`)
- Cutout alpha masking via `isCutoutTransparent()` (in `flat_shading.slang`)
- **UsdPreviewSurface textures** — per-input channel selection (`channelMask`),
  normal-map `scale`/`bias` (`normalScale`/`normalBias`, for OpenGL vs DirectX
  Y convention), and wrap modes flow from each material's `TextureBinding`
  (binding 14 bindless textures)
- **MaterialX graph evaluation** when `materialTypes[id]` packs a graph
  slot — `evalSceneGraph(materialId, hit, ...)` (generated module) returns
  `StdSurfaceParams` overrides (base_color, roughness, metallic, etc.) before
  the BSDF math runs

### Python Material (`materials` type code 3)

SlangPile-authored materials (`mtlx/genslang/python_materials/*.py`) compile to
`IMaterial` structs. Their per-material id is packed into bits 24–31 of
`materialTypes[id]`; `vk_compute._emit_python_dispatcher` generates a
switch that routes `pythonMaterialId(matId)` to the right struct. Edited live
through the Qt material editor.

### Bidirectional Path Tracer (`integrators/bdpt.slang`)

Veach §10 BDPT with V1 simplifications for shader compile time:

- **Subpaths**: eye walk + light walk, each capped at 4 vertices
- **Connections**: (s ≥ 1, t ≥ 1) evaluate the real `standard_surface`
  BSDF via `FlatMaterial.evaluate()` (not the earlier Lambertian f ≈
  albedo/π approximation); `FlatMaterial.sample()` drives walk bounces
- **Environment**: env-miss and s=0 contributions use the same
  importance-sampled environment distribution + MIS as the path tracer,
  so BDPT and path-traced IBL converge to the same image
- **Light tracer** (s = 1): non-delta light vertices projected onto camera,
  atomic-added to `lightSplatBuffer` (binding 21, Q22.10 fixed-point per
  R/G/B channel). `main_pass.slang` composites the running mean after
  accumulation
- **Scope**: flat-material first-hit only; skin/debug hits fall through to
  PathTracer
- **MIS**: balance heuristic over all (s, t) strategies per path length;
  `convertSAtoArea()` handles geometry-term conversion

---

## MaterialX Nodegraph Compute Pipeline

> **Build prerequisite:** the Slang generator (`PyMaterialXGenSlang`) is **not**
> in the PyPI MaterialX wheel — you must build MaterialX from source with
> `-DMATERIALX_BUILD_PYTHON=ON -DMATERIALX_BUILD_GEN_SLANG=ON` and install the
> resulting `python/` tree into your venv. The whole pipeline below depends on
> it; see the *MaterialX from source* section in `README.md` for the full
> recipe.

Arbitrary MaterialX nodegraphs (e.g. marble, wood, brass — see
`assets/three_materials_demo.usda`) are compiled to per-material Slang
modules at scene-load time and again whenever a graph signature changes.

```
USD scene with MaterialX ─► usd_loader.py
                            └─► CompiledMaterial per UsdShade.Material
                                ├─ MaterialXGenSlang generates:
                                │     graph_<hash>.slang   (in shaders/generated/)
                                │     defines `evalSceneGraph_<hash>(hit, params)`
                                ├─ filename inputs replaced with bindless slot
                                │  indices via TexturePool (binding 14)
                                └─ generated_materials.slang switch-dispatches
                                   `evalSceneGraph(materialId, hit, params)` →
                                   the right per-graph function

Renderer._compile_pipeline()
  └─ ComputePipeline._run_codegen()    # walks slangpile + MaterialX graphs
  └─ Slang compile (SPIR-V)             # mtime-LRU cache, ≤32 entries
       └─ SPV cache key = source hash + entry point
  └─ Pipeline rebuild
       └─ skipped when graph set is content-identical
       └─ texture pool repopulated after rebuild
```

Key invariants:

- **`mtlx_gen_shim.slang`** wraps `SamplerTexture2D` so generated modules
  see `Texture2D + SamplerState`-style methods backed by binding-14
  bindless lookups. Sentinel slot `0xFFFFFFFFu` returns transparent black
  to avoid sampling unbound descriptors.
- **`SampleLevel` only** in generated modules — compute pipelines have no
  derivatives.
- **Slang fallback** on graph compile failure preserves the flat
  base_color so the scene still renders; an infinite slangc retry is
  guarded by skipping rebuilds for known-broken graph signatures.
- **`makeBSDF` + dual-author overrides**: `standard_surface` parameters
  authored both as nodes and as direct inputs are merged so graph-uniform
  sliders in the sidebar drive both code paths.

`MaterialLibrary` (`materialx_runtime.py`) owns the document, the
`MaterialXGenSlang` shadergen instance, the per-material reflection of
the uniform block, and the `pack_material_values()` byte serialiser.

---

## Environment Importance Sampling (`environment.slang`)

A 2D piecewise-constant distribution over the equirect environment map drives
next-event estimation toward bright sky/sun directions instead of relying on a
BSDF ray happening to land on them — the fix for specular environment
fireflies. The distribution is built CPU-side in
`environment.build_env_distribution()` (sin θ-weighted luminance) and uploaded
as two CDF buffers:

- binding 31 — `envMarginalCdf` (row marginal, `ENV_H + 1` floats)
- binding 32 — `envCondCdf` (per-row conditional, `H × (W + 1)` floats)

`sampleEnvDir(u, intensity)` importance-samples a direction + solid-angle PDF;
`envPdf(dir)` returns the PDF of an arbitrary direction so BSDF-sampled
env-miss hits can be MIS-weighted against env NEE. `ENV_DIST_W = 1024`,
`ENV_DIST_H = 512` must match `ENV_WIDTH`/`ENV_HEIGHT` in `environment.py`.
Both the path tracer and BDPT consume this distribution so their IBL
converges to the same image.

---

## Display: Exposure, Tonemap, and Tool Readback

`main_pass.slang` post-processes the accumulated linear-HDR image:

- **Exposure** — `fc.exposure` (EV stops, applied as `2^EV`) before tonemapping.
- **Tonemap operator** — `fc.tonemapMode`: 0 = ACES filmic (Narkowicz),
  1 = Reinhard, 2 = Hable/Uncharted 2, 3 = Linear clamp. Exposure and tonemap
  are post-process knobs — changing them does **not** reset accumulation.
- **Tool readback** (binding 30, `toolBuffer`) — one-shot probes that write
  per-pixel data back to the CPU: scene pick (`fc.pickArmed` + `fc.pickPixel`
  → `HitInfo`), the BXDF visualiser (`TOOL_MODE_BXDF`), and a BSSRDF probe
  (`TOOL_MODE_BSSRDF`). The CPU clears the arm flag after a single read.

---

## Scene System

Head geometry — the analytic SDF head (`sdf_head.slang`) and the two-level
mesh BVH (`mesh_head.slang`, TLAS/BLAS) that `traceScene()` dispatches to — is
documented in [SkinRendering.md](SkinRendering.md).

### Scene Dispatch (`scene_trace.slang`)

`traceScene()`:
- `furnaceMode` → unit sphere intersection
- `useMesh` → `marchHeadMesh()` (BVH traversal)
- else → `marchHead()` (SDF sphere tracing)

Shadow tests: `visibleSegment()` (point-to-point), `visibleDirectional()`
(point toward infinity). Both traverse up to 8 transparent surfaces
(cutout alpha or refractive) before declaring occlusion.

Transparency helpers (defined in `materials/flat/flat_shading.slang`):
- `isCutoutTransparent(h)` — alpha below `opacityThreshold`
- `isMaterialTransparent(materialId)` — opacity < 1 or opacity texture
- `isShadowTransparent(h)` — cutout or refractive

### USD Loading (`usd_loader.py`)

Walks USD stage for `UsdGeom.Mesh`, `UsdLux` lights (DistantLight,
SphereLight, DomeLight, RectLight), `UsdGeom.Camera`, `UsdShade.Material`
bindings with UsdPreviewSurface, MaterialX, and **OpenPBR** overrides.
Connected shader inputs are resolved to their authored constant when a node
graph drives them (the OpenPBR / `standard_surface` connection case), so
single-value parameters survive even when authored through a connection.
Converts `metersPerUnit` → `mm_per_unit`. CW-wound triangles (e.g.
`three_materials_demo.usda` quads) are flipped on import so normals are
consistent.

`UsdUVTexture` reads populate a `TextureBinding` (`scene.py`) per material
input — file path plus `inputs:scale`/`inputs:bias` (e.g. DirectX normal maps
author `scale.y = -2`, `bias.y = +1` to flip Y), channel selector
(`rgb`/`r`/`g`/`b`/`a`), `sourceColorSpace`, and `wrapS`/`wrapT`. The renderer
packs scale/bias into `FlatMaterialParams.normalScale`/`normalBias` and the
per-input channel selectors into `channelMask` (4 bits per input), so the
shader fetches the correct channel and applies the right normal-map convention
without per-texture branches.

### Runtime Scene-Graph Editing (`renderer.py`)

The loaded `Usd.Stage` is the authoritative scene model; the flat `Scene` +
GPU buffers are a derived cache. `Renderer._attach_edit_layer()` inserts an
in-memory anonymous `Sdf.Layer` as the strongest sublayer and sets it as the
edit target, so every runtime edit is authored there and the original file is
never written until `save_edits()`. The editing API — `add_model()` (define an
`Xform` + `AddReference`), `remove_node()` (`SetActive(False)`),
`set_transform()` (author `xformOp:transform`), `save_edits()`, `list_nodes()` —
authors inside a scoped `Usd.EditContext`. Add/remove trigger a geometry resync
(`_resync_geometry_from_stage`: re-read via `load_scene_from_stage`, mesh cache
keeps unchanged prims free, runtime `enabled` flags carried by prim path);
`set_transform` uses a transform-only fast path (`_reupload_instance_transforms`,
no re-bake). `MeshInstance.prim_path` + the `_prim_to_instances` index key all
edits by USD prim path; edits reset progressive accumulation via
`_material_version`. Headless callers pass `stage=` to `set_usd_scene`.

The geometry resync also re-reads lights + camera (so deleting a light/camera
prim drops it; `LightDir`/`LightSphere` carry `prim_path` to preserve runtime
`enabled` toggles across the re-read) and rebuilds the derived scene graph
(`build_scene_graph` + default-light injection) while bumping
`_scene_graph_version`, so the scene-graph panels repaint. Both front-ends drive
this from their scene-graph view — the Qt dock (`ui/qt/windows/scene_graph.py`)
and Panel card (`ui/panel/windows.py`) expose Add model / Delete node / Save
edits and route per-node TRS edits through `set_transform`. The decision logic
(add-parent resolution, deletability, TRS→matrix) lives in pure helpers
(`ui/scene_edit_actions.py`) shared by both and unit-tested without a display.

### USD Animation Playback (`playback.py`, `usd_loader.py`, `renderer.py`)

At load, `build_animation_index(stage)` scans for time-varying prims — transform
tracks (incl. ancestor-driven), animated lights, an animated camera — and
skinned meshes. `build_playback_clock(stage, index)` reads the stage's
`startTimeCode`/`endTimeCode`/`timeCodesPerSecond` into a `PlaybackClock` (pure
time logic: advance, loop, normalized scrub). The renderer keeps the stage alive
(`_usd_stage`) so prims can be re-evaluated at runtime.

Each frame, `Renderer.update(dt)` advances the clock and `_apply_animation_frame`
re-evaluates only the indexed prims at `current_time_code`: animated transforms
recompute the world matrix (`_world_transform`) and re-upload only those TLAS
`instance_buffer` records (no mesh rebake / BVH rebuild); animated lights are
re-extracted; an animated USD camera feeds a follower used in `camera_mode ==
"usd"`. `current_time_code` is folded into `_current_state_hash`, so playback
resets accumulation (1 spp in motion, converges when paused). A built-in
transport (play/pause, normalized scrubber, fps) lives in the shared spec tree,
shown only when the stage has animation.

### UsdSkel Skeletal Skinning (`usd_loader.py`, `vk_skinning.py`, `shaders/skin.slang`, `shaders/bvh_refit.slang`)

`extract_skeletal_bindings(stage)` returns a `SkeletalScene` (retaining the cache
+ stage) with one `SkinnedMeshBinding` per skinned mesh: rest points/normals,
`jointIndices`/`jointWeights`, influences, and the skel/skinning queries.
`compute_joint_matrices(binding, time)` builds per-joint matrices (mapper remap +
geomBindTransform fold), validated against pxr `ComputeSkinnedPoints`; deformed
points live in the authored-points space, so the loader's existing TLAS transform
places them (no identity-TLAS).

On Vulkan, `SkinningPasses` (`vk_skinning.py`) owns two standalone compute
pipelines with their **own descriptor sets** (the main 0–32 binding map is
untouched): `skin.slang` linear-blend-skins rest vertices into the shared vertex
buffer; `bvh_refit.slang` refits each skinned mesh's BVH in place (parallel leaf
AABBs are folded into a single-thread reverse-array-order pass — valid because the
depth-first build emits parents before children). They run as one isolated
submit (skin → barrier → refit) before the frame render — no edit to the shared
render recording, no GPU→CPU readback. Non-Vulkan backends fall back to CPU
skinning + BLAS rebuild.

### USD-Driven Scene Controls (`usd_loader.py`, `ui/build_app_ui.py`)

`extract_ui_controls(stage)` parses any prim with an authored `skinny:ui:type`
into a `ControlSpec` (type, prefix-typed `target`, label, range, choices,
default, order). `resolve_control_binding(renderer, spec)` maps the target prefix
to live get/set closures: `renderer:`/`mtlx:` → `_get_nested`/`_set_nested`;
`material:<name>:<input>` → `apply_material_override`; `usd:<prim>.<attr>` →
attribute `Get`/`Set` + a live-state refresh (lights/transforms/camera). A
data-driven "Scene Controls" `DynamicSection` in `build_main_ui` renders one
widget per control across all front-ends, shown only when the stage declares
controls. Authored `skinny:ui:default` values apply at load.

### Scene Graph Inspector (`scene_graph.py`, `ui/qt/windows/scene_graph.py`)

Preserves the USD prim hierarchy as a browsable tree with typed,
editable properties on each node. `SceneGraphNode` carries a
`RendererRef` (kind + index) mapping back to the flat renderer arrays
(material, light, instance). Property edits flow through
`apply_material_override` / `apply_light_override` /
`apply_instance_transform`. Qt presents tree-above-properties inside a
`QDockWidget`; the web UI (`ui/panel/windows.py` + `scene_tree.html`)
serves the same model in a Panel iframe.

---

## Camera, Lens, and Debug Viewport

### Camera ray gen (`shaders/cameras/`)

`pinhole.slang` is the default projective camera. `thick_lens.slang` is a
straight port of PBRT-v3's `RealisticCamera`, with two CPU-side helpers
in `lens_optics.py`:

- `trace_lenses_from_film()` — line-by-line PBRT port for verification.
- `bound_exit_pupil()` — packs per-radius exit-pupil rectangles so the
  shader can sample only directions the lens won't vignette. Without
  this, closing the iris collapses each pixel to a central pinhole and
  shrinks the rendered area.

### Debug viewport (`debug_viewport.py` + `shaders/debug_line.slang`)

Second view that rasterises wireframe visualisations of the render
camera, its lens elements, per-instance world-space AABBs (or full mesh
wireframes), a ground grid, and a small camera-body glyph. Lives in two
places:

- Standalone GLFW window (used by the GLFW debug entry; toggled with
  `F2`). Owns its own surface, swapchain, depth buffer, render pass,
  line-list pipeline, vertex buffer, and per-frame sync — sharing only
  the `VulkanContext` device/queue.
- Embedded Qt dock (`ui/qt/windows/debug_viewport.py`) that renders to
  an offscreen image and blits it via Qt — same pipeline, no GLFW.

Geometry is regenerated from live `Renderer` state every frame.

### Rotate gizmo (`gizmo.py`)

`RotateGizmo` tracks one selected mesh instance and exposes three
orthogonal screen-space rings (X/Y/Z, world-axis aligned) around its
pivot. The renderer rebuilds the line list per frame and uploads it to
binding 22; `main_pass.slang` draws each segment as an anti-aliased
line over the final tonemapped image.

### BXDF visualiser (`bxdf_math.py` + `ui/qt/windows/bxdf.py`)

CPU-side Lambert + GGX-Smith standard_surface evaluation, hemisphere
lobe rasterisation via Pillow. The Qt dock binds it to a material
picker so any scene material can be inspected in isolation.

### MaterialX graph editor (`mtlx_graph_view.py` + `ui/qt/windows/material_graph.py`)

Pure view-model (`NodeGraphView`, `NodeView`, `PortView`) extracted
from the legacy Tk editor so the Qt port and Panel port can share it.
Edits flow back through `MaterialLibrary` and trigger a graph rebuild
+ pipeline recompile.

---

## Headless Render API (`skinny.headless`)

`skinny.headless` is the public offscreen-render interface, driving
`Renderer.set_usd_scene()` + `usd_loader.load_scene_from_stage()` directly
with no window or event loop. Key symbols:

- `HeadlessRenderer(w, h)` — context-manager that owns `VulkanContext` +
  `Renderer`; pipeline compiles once, then `render_to_array(stage)` /
  `render_scene(stage, path)` / `render_animation(stage, outdir)` can be
  called repeatedly with a mutated `Usd.Stage` per frame.
- Module-level `render_scene` / `render_to_array` / `render_animation` —
  convenience wrappers that open and close the GPU context for one-shot calls.
- `skinny-render` CLI entry point wraps the same API; `--animate` renders a
  frame sequence over USD timecodes.

---

## Web Application Architecture

### Overview

`skinny-web` serves a Panel (HoloViz) web application with per-user
server-side rendering. Each browser session gets its own Vulkan renderer,
H264 encoder, and render thread. The Panel/Bokeh protocol handles widget sync
and session isolation; a custom Tornado WebSocket streams encoded video.
The sidebar widget tree comes from the same `ui/build_app_ui.build_main_ui`
spec that the Qt app uses.

### Session Lifecycle

```
Browser connects → Panel creates session → SkinnySession.__init__()
  ├─ VulkanContext(window=None)  # headless, no GLFW
  ├─ Renderer(vk_ctx, ..., usd_scene)
  ├─ VideoEncoder(w, h, gpu_info)  # H264 hw or sw
  └─ Thread(_render_loop)  # starts immediately

Browser disconnects → on_session_destroyed → session.cleanup()
  ├─ _running = False → thread joins
  ├─ encoder.close()
  ├─ renderer.cleanup()
  └─ ctx.destroy()
```

Max concurrent sessions capped (default 4) to bound GPU memory.

### Video Streaming Protocol

Binary WebSocket at `/video_ws/<session_id>`:

| Frame type | Byte 0 | Payload |
|------------|--------|---------|
| H264 keyframe | 0 | AVCC-framed NAL units |
| H264 delta | 1 | AVCC-framed NAL units |
| JPEG fallback | 2 | JPEG image |
| AVCC description | 3 | SPS/PPS for decoder init |

Header: `!BI` (1 byte type + 4 byte accum frame number) + payload.

On WebSocket open, stale frames are drained and encoder forced to emit a
keyframe so the browser decoder starts clean.

Browser-side: WebCodecs `VideoDecoder` for H264 → `<canvas>` blit. Falls back
to JPEG `<img>` when WebCodecs unavailable.

### Hardware Abstraction (`hardware.py`)

GPU selection is vendor-aware:

```
enumerate_gpus(vk_instance) → list[GpuInfo]
select_gpu(vk_instance, preference) → GpuInfo
```

`GpuInfo.preferred_h264_encoder` maps vendor → encoder:
- Intel (0x8086) → `h264_qsv`
- NVIDIA (0x10DE) → `h264_nvenc`
- AMD (0x1002) → `h264_amf`
- Fallback → `libx264`

All entry points accept `--gpu {intel,nvidia,amd,discrete,auto}`.

### H264 Encoder (`video_encoder.py`)

Wraps PyAV for H264 encoding with hardware-aware fallback chain:

1. Try `gpu_info.preferred_h264_encoder`
2. Fall back to `libx264`
3. If all fail, JPEG-only mode

Encoder outputs **Annex B** NAL units (PyAV default), converted to **AVCC**
framing for WebCodecs compatibility. AVCC description (SPS+PPS) sent once on
WebSocket open.

Key methods:
- `encode_h264(rgba_bytes)` → list of `(is_key, avcc_data)` tuples
- `encode_jpeg(rgba_bytes, quality)` → JPEG bytes (fallback)
- `force_keyframe()` → next frame forced as IDR (called on param/camera change)

### Headless Vulkan Path

`VulkanContext(window=None)`:
- No GLFW dependency, no surface/swapchain
- Compute queue only (no present queue)
- No surface extensions in instance creation

`Renderer.render_headless()`:
- Dispatches to persistent offscreen `StorageImage` (not swapchain image)
- Barrier → `ReadbackBuffer.record_copy_from()` → fence wait → `read()`
- Returns raw RGBA bytes

The Qt entry (`skinny-gui`) runs in the same headless mode and blits the
readback into a `QImage` via `RenderViewport`.

---

## Backend Abstraction (`gfx/`)

A new abstraction layer lets the renderer talk to a `Backend` instance
(`gfx/backend.py`) instead of touching Vulkan directly:

```
Backend
  ├─ caps: BackendCaps        # bindless / scalar layout / push descriptors
  ├─ device: Device            # queues, allocators, command recording
  ├─ presenter: Presenter|None # surface/swapchain (None = headless)
  └─ shader_target() -> "spirv" | "metal"
```

| Backend | Status |
|---------|--------|
| `gfx/vulkan/` | Production — wraps `vk_context.py` + `vk_compute.py` |
| `gfx/metal/` | Stub for a future native macOS path; MoltenVK still uses the Vulkan backend |

`vk_context.py` and `vk_compute.py` keep their direct Vulkan API; the
abstraction is layered above them so existing code keeps working while
new code paths (preview pass, debug viewport line pipeline) are
incrementally moved over.

---

## Descriptor Binding Map

| Binding | Type | Content | Owner |
|---------|------|---------|-------|
| 0 | UBO | FrameConstants + SkinParams + light uniforms | `bindings.slang` |
| 1 | RWTexture2D | Swapchain / offscreen output (RGBA8) | `bindings.slang` |
| 2 | RWTexture2D | HDR accumulation (RGBA32F) | `bindings.slang` |
| 3 | RWTexture2D | HUD alpha mask (R8) | `bindings.slang` |
| 4 | Sampler2D | HDR environment map (1024×512) | `environment.slang` |
| 5 | StructuredBuffer | Mesh vertices (32 B each) | `mesh_head.slang` |
| 6 | StructuredBuffer | Mesh indices (uint32) | `mesh_head.slang` |
| 7 | StructuredBuffer | BVH nodes (32 B each) | `mesh_head.slang` |
| 8 | Sampler2D | Tattoo map (512×512 RGBA) | `materials/skin/skin_shading.slang` |
| 9 | Sampler2D | Normal detail map (2048²) | `materials/skin/skin_shading.slang` |
| 10 | Sampler2D | Roughness detail map (2048²) | `materials/skin/skin_shading.slang` |
| 11 | Sampler2D | Displacement detail map (2048²) | `materials/skin/skin_shading.slang` |
| 12 | StructuredBuffer | TLAS instances (144 B each) | `mesh_head.slang` |
| 13 | StructuredBuffer | FlatMaterialParams (128 B each, scalar layout) | `bindings.slang` |
| 14 | Sampler2D[128] | Bindless material textures (PARTIALLY_BOUND) | `bindings.slang` |
| 15 | StructuredBuffer | MtlxSkinParams (164 B each, scalar layout) | `materials/skin/skin_shading.slang` |
| 16 | StructuredBuffer | Material type code + scatter + furnace + graph slot + python id (uint32 each) | `bindings.slang` |
| 17 | StructuredBuffer | SphereLight (32 B each) | `scene_lights.slang` |
| 18 | StructuredBuffer | EmissiveTriangle (64 B each) | `scene_lights.slang` |
| 19 | StructuredBuffer | StdSurfaceParams (256 B each) | `bindings.slang` |
| 20 | StructuredBuffer | DistantLight (analytic distant lights) | `scene_lights.slang` |
| 21 | RWStructuredBuffer | BDPT light-splat buffer (Q22.10 uint per R/G/B) | `bindings.slang` |
| 22 | StructuredBuffer | Rotate-gizmo line segments | `gizmo.py` |
| 23 | StructuredBuffer | Lens elements (thick-lens stack, float4) | `cameras/thick_lens.slang` |
| 24 | StructuredBuffer | Per-radius exit-pupil bounds (float4) | `cameras/thick_lens.slang` |
| 30 | RWStructuredBuffer | Tool readback (float4) — scene pick / BXDF / BSSRDF probe | `bindings.slang` |
| 31 | StructuredBuffer | Env importance-sampling marginal CDF (ENV_H+1 floats) | `environment.slang` |
| 32 | StructuredBuffer | Env importance-sampling conditional CDF (H×(W+1) floats) | `environment.slang` |

Light uniforms (part of UBO, not separate bindings):
- `lightDirection` (float3) — analytic directional light toward-light vector
- `lightRadiance` (float3) — analytic directional light colour × intensity

`ProceduralParams` (formerly binding 20) was removed; procedural flat colour is
now derived inside `flat_shading.slang` without a dedicated buffer.

---

## SlangPile — Embedded Python→Slang Codegen

Located in `src/skinny/slangpile/`. Python functions decorated with `@sp.shader`
are transpiled to Slang source code. Used for rapid material prototyping — not
a build-time requirement (generated `.slang` files are checked into git).

### Modules

| File | Purpose |
|------|---------|
| `api.py` | `@shader`, `extern()`, `compile_module()`, `build_module()`, `load_module()` |
| `types.py` | `SlangType` hierarchy — scalars, vectors (float2/3/4), matrices |
| `registry.py` | Global shader/extern function registries |
| `runtime.py` | `SlangPyModule`, `RuntimeConfig`, `call_shader()` |
| `compiler/module.py` | AST-walking transpiler: `ModuleCompiler` + `FunctionEmitter` |
| `diagnostics.py` | `Diagnostic`, `SlangPileError` |
| `verification.py` | `slangc` invocation wrapper for syntax checking |
| `cli.py` | CLI: `build`, `check`, `verify` subcommands |

### Codegen Hook

`ComputePipeline._run_codegen()` (in `vk_compute.py`) runs before every
`_compile_slang()`. Walks `mtlx/genslang/python_materials/*.py`, calls
`build_module()`, writes to `mtlx/genslang/generated_*.slang` plus a
`slangpile_manifest.json`. Failures are non-fatal (debug log only).

---

## Python Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `app.py` | `InputHandler` | GLFW shader-debug entry |
| `ui/qt/app.py` | `MainWindow` | `skinny-gui` entry — dock layout, menu, file open |
| `ui/qt/viewport.py` | `RenderViewport` | Qt widget that blits the renderer's offscreen image |
| `ui/qt/backend.py` | `QtTreeBuilder` | Walks the spec tree, instantiates Qt widgets |
| `ui/spec.py` | `Section`, `DynamicSection`, `Slider`, … | Pure dataclass widget tree |
| `ui/build_app_ui.py` | `AppCallbacks`, `build_main_ui` | Builds shared sidebar tree |
| `ui/panel/backend.py` | `PanelTreeBuilder` | Walks the spec tree, instantiates Panel widgets |
| `web_app.py` | `SkinnySession`, `VideoStreamHandler` | Panel web app, per-session renderer, Tornado video WS |
| `renderer.py` | `Renderer`, `SkinParameters`, `OrbitCamera`, `FreeCamera`, `TexturePool` | GPU resource orchestration, per-frame dispatch |
| `vk_context.py` | `VulkanContext`, `SwapchainInfo` | Vulkan 1.3 instance, device, swapchain (+ headless mode) |
| `vk_compute.py` | `ComputePipeline`, `UniformBuffer`, `StorageImage`, `StorageBuffer`, `SampledImage`, `ReadbackBuffer`, `HudOverlay` | Shader compilation (Slang→SPIR-V), GPU resource types |
| `gfx/backend.py` | `Backend`, `BackendCaps` | Backend ABC |
| `gfx/device.py` | `Device` | Device abstraction |
| `gfx/presenter.py` | `Presenter` | Surface/swapchain abstraction |
| `gfx/vulkan/*` | — | Vulkan implementation |
| `scene.py` | `Scene`, `Material`, `MeshInstance`, `LightDir`, `LightSphere`, `LightEnvHDR` | Scene description dataclasses |
| `materialx_runtime.py` | `MaterialLibrary`, `CompiledMaterial`, `UniformField` | MaterialX loading, GenSlang codegen, uniform reflection |
| `mesh.py` | `Mesh`, `MeshSource` | OBJ loading, subdivision, displacement, BVH construction |
| `mesh_cache.py` | — | On-disk BVH cache (zstd-compressed vertex/index/BVH blobs, `~/.skinny/mesh_cache/`) |
| `environment.py` | `Environment` | HDR env map loading (.hdr decoder), built-in presets |
| `params.py` | `ParamSpec` | Shared parameter definitions, get/set helpers, persistence |
| `hardware.py` | `GpuInfo`, `GpuVendor` | GPU enumeration, vendor detection, encoder selection |
| `video_encoder.py` | `VideoEncoder` | H264/JPEG encoding with hw-aware fallback, Annex B→AVCC |
| `scene_graph.py` | `SceneGraphNode`, `SceneGraphProperty`, `RendererRef` | USD prim hierarchy tree model with typed editable properties |
| `mtlx_graph_view.py` | `NodeGraphView`, `NodeView`, `PortView` | View-model for MaterialX nodegraph editor |
| `bxdf_math.py` | — | CPU BSDF eval + lobe rasterisation |
| `gizmo.py` | `RotateGizmo` | Rotate gizmo math + line-list buffer |
| `lens_optics.py` | — | PBRT-v3 thick-lens helpers |
| `debug_viewport.py` | `DebugViewport` | Camera/lens/wireframe debug renderer |
| `presets.py` | `Preset` | 12 built-in skin presets (Fitzpatrick I–VI × Female/Male) |
| `settings.py` | — | Persistent storage at `~/.skinny/` (JSON) |
| `tattoos.py` | `Tattoo` | Procedural + image-based tattoo loading |
| `head_textures.py` | `TextureStats` | Detail map loading (normal, roughness, displacement) at 2048² |
| `usd_loader.py` | — | USD stage → Scene (meshes, lights, cameras, materials, MaterialX fallback) |
| `fetch_hdrs.py` | — | Downloads CC0 HDRIs from Poly Haven |

---

## Shader Module Dependency Graph

```
                           common.slang
                          ╱     |     ╲
              interfaces      bindings    materials/skin/skin_bssrdf
              ╱  |  |  ╲       |              |       ╲
        ICamera  ISampler   IMaterial  scene_trace  materials/skin/skin_shading
            |        |        ILight        |    ╲       |
       cameras/  samplers/    BSDFSample sdf_head  mesh_head
            |              |  BSDFEval                 ╲
       lights/                                      materials/skin/{direct,ibl_*,
            |                                            volume,transmission,hair_sheen}
                                                          |
                                            materials/skin/skin_material ─┐
                                            materials/flat/flat_material ─┤
                                            materials/debug_normal_material┤
                                            materials/flat/flat_shading ──┤
                                            shaders/generated/graph_*.slang┤
                                              (via mtlx_gen_shim)         │
                                                                          ▼
                                              integrators/path.slang ──────┐
                                              (evaluateBounce + bounce loop)
                                              integrators/bdpt.slang ──────┤
                                              (eye/light walks + MIS)      │
                                                                           ▼
                                                                  main_pass.slang
```

---

## FrameConstants Layout

Compiled with `-fvk-use-scalar-layout` — float3 has 4-byte alignment.

| Offset | Type | Field |
|--------|------|-------|
| +0 | Camera | viewInverse, projInverse, position, fov |
| ... | uint | frameIndex |
| ... | uint | accumFrame |
| ... | float | time |
| ... | uint | width, height |
| ... | uint | numDistantLights (active count in binding 20; 0 = IBL only) |
| ... | uint | useMesh |
| ... | float | tattooDensity |
| ... | float | envIntensity |
| ... | uint | furnaceMode |
| ... | float | mmPerUnit |
| ... | uint | detailFlags |
| ... | float | normalMapStrength |
| ... | float | displacementScaleMM |
| ... | uint | numInstances |
| ... | uint | numSphereLights |
| ... | uint | numEmissiveTriangles |
| ... | uint | integratorType (0 = path, 1 = BDPT) |
| ... | (lens) | numLensElements (0 = pinhole, else thick-lens), film/aperture/pupil + focus-overlay + zoom-rect + vignette-debug fields |
| ... | uint2 | pickPixel; uint pickArmed (one-shot scene pick → toolBuffer) |
| ... | float | exposure (EV stops, 2^EV) |
| ... | uint | tonemapMode (0 ACES, 1 Reinhard, 2 Hable, 3 linear) |

`cameraType` was removed — camera selection is implied by `numLensElements`
(0 ⇒ pinhole). `exposure` and `tonemapMode` are post-process knobs and do not
reset accumulation.

---

## Key Invariants

- **`mmPerUnit`**: only `loadSkin` / volume code converts mm→world units.
  Estimators receive already-converted σ values.
- **Scalar block layout**: all UBOs and SSBOs use `-fvk-use-scalar-layout`.
  float3 has 4-byte alignment (no 16-byte promotion). Struct packing on the
  Python side must match exactly.
- **Progressive accumulation**: running mean in linear HDR. One NaN permanently
  poisons a pixel — guarded in `main_pass.slang` (reject NaN / inf / negative
  before accumulation).
- **Furnace mode**: `main_pass.slang` flags energy-conservation violations
  in pink. Supports both global (`fc.furnaceMode`) and per-material (bit 10
  in `materialTypes[]`) furnace probes via `effectiveFurnaceMode()`. Every
  material must converge to L=1.0 under a white unit-sphere environment.
- **Material dispatch**: tag-switch monomorphisation in `evaluateBounce()`
  (`integrators/path.slang`) and `BDPTIntegrator` (`integrators/bdpt.slang`).
  Never existential `IMaterial`. NEE is generic (`allLightsNEE<TM>`) —
  monomorphised per material type.
- **MaterialX graph dispatch**: `evalSceneGraph(materialId, ...)` is also a
  switch statement, generated into `shaders/generated/generated_materials.slang`
  with one case per active graph hash.
- **RNG order**: skin estimators (§1–§6) are called in fixed sequence so RNG
  state stays pixel-identical across refactors.
- **BVH caching**: `mesh_cache.py` stores zstd-compressed vertex/index/BVH
  blobs keyed by content hash. Cache hit skips subdivision + BVH build.
- **SPIR-V cache**: bounded to ~32 entries via mtime-LRU eviction. Pipeline
  rebuilds skip when the graph set is content-identical; texture pool is
  repopulated after every rebuild.
- **MaterialX texture sampling**: generated graph modules must use
  `SampleLevel` (no derivatives in compute pipelines) and must guard against
  the bindless `SENTINEL` slot via `mtlx_gen_shim.slang`.
- **Single widget tree**: the Qt and Panel UIs both consume
  `ui/build_app_ui.build_main_ui`. New parameters added there appear in
  both UIs without per-backend code.

---

## File Listing

### Python (`src/skinny/`)

```
__init__.py              app.py                 renderer.py
vk_context.py            vk_compute.py          scene.py
materialx_runtime.py     mesh.py                mesh_cache.py
environment.py           params.py              hardware.py
video_encoder.py         web_app.py             scene_graph.py
mtlx_graph_view.py       bxdf_math.py           gizmo.py
lens_optics.py           debug_viewport.py      presets.py
settings.py              tattoos.py             head_textures.py
fetch_hdrs.py            usd_loader.py
```

### Backend abstraction (`src/skinny/gfx/`)

```
backend.py               device.py              presenter.py
pipeline.py              command.py             resources.py
shader_compiler.py       types.py
vulkan/{backend.py, device.py, presenter.py, command.py,
        resources.py, sync.py, _helpers.py}
metal/                   (placeholder)
```

### UI (`src/skinny/ui/`)

```
spec.py                  build_app_ui.py        direction_math.py
qt/app.py                qt/backend.py          qt/viewport.py
qt/camera_input.py       qt/direction_picker.py
qt/windows/scene_graph.py     qt/windows/material_graph.py
qt/windows/bxdf.py            qt/windows/debug_viewport.py
panel/backend.py         panel/windows.py
```

### Web Templates (`src/skinny/web_templates/`)

```
video_player.html        WebCodecs decoder + camera controls (JS)
scene_tree.html          USD scene graph tree + property editor (web)
```

### SlangPile (`src/skinny/slangpile/`)

```
__init__.py              api.py                 types.py
registry.py              runtime.py             diagnostics.py
verification.py          cli.py                 compiler/__init__.py
compiler/module.py
README.md
```

### Shaders (`src/skinny/shaders/`)

```
common.slang             interfaces.slang        bindings.slang
main_pass.slang          preview_pass.slang
scene_trace.slang        scene_lights.slang
mesh_head.slang          sdf_head.slang
environment.slang        volume_render.slang
mtlx_closures.slang      mtlx_std_surface.slang  mtlx_noise.slang
mtlx_gen_shim.slang      generated_materials.slang
debug_line.slang
cameras/{pinhole.slang, thick_lens.slang}
materials/debug_normal_material.slang
materials/flat/{flat_material.slang, flat_shading.slang}
materials/skin/{skin_material.slang, skin_bssrdf.slang, skin_shading.slang,
                skin_direct.slang, skin_ibl_specular.slang, skin_ibl_diffuse.slang,
                skin_volume.slang, skin_transmission.slang, skin_hair_sheen.slang,
                detail.slang}
samplers/{ggx.slang, lambert.slang, uniform_sphere.slang,
          henyey_greenstein.slang, mis_combine.slang}
lights/{sphere_light.slang, emissive_triangle_light.slang, directional_light.slang}
integrators/{path.slang, bdpt.slang}
generated/                          # MaterialXGenSlang output, gitignored
lib/mx_closure_type.glsl
```

### MaterialX (`src/skinny/mtlx/`)

```
skinny_defs.mtlx                    skinny_skin_default.mtlx
genslang/skinny_genslang_impl.mtlx
genslang/skinny_skin_epidermis_genslang.slang
genslang/skinny_skin_dermis_genslang.slang
genslang/skinny_skin_subcut_genslang.slang
genslang/skinny_scattering_layer_genslang.slang
genslang/skinny_skin_layered_bsdf_genslang.slang
genslang/skinny_skin_layered_vdf_genslang.slang
genslang/python_materials/*.py     # SlangPile material drafts
genslang/slangpile_manifest.json
```

### Tests (`tests/`)

```
conftest.py              helpers.py               __init__.py
test_environment.py      test_headless.py         test_integration.py
test_intersections.py    test_lights.py           test_math.py
test_materialx_graph.py  test_mis.py              test_mtlx_closures.py
test_sampling.py         test_skin_optics.py      test_slangpile.py
test_slangpile_execution.py  test_struct_layout.py
test_ui_spec.py          test_volume.py           test_web.py
harnesses/test_common_harness.slang   harnesses/test_environment_harness.slang
harnesses/test_light_harness.slang    harnesses/test_sampler_harness.slang
harnesses/test_skin_harness.slang     harnesses/test_volume_harness.slang
kernels/energy_ref.py    kernels/sampling_ref.py
```
