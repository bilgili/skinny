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

The renderer has **two GPU execution modes** for the *same* light-transport
integral, selected once at startup with `--execution-mode megakernel|wavefront`
and fixed for the session. Each has its own technical deep-dive:

- **[Megakernel.md](Megakernel.md)** — the default: one monolithic
  `[numthreads(8,8,1)]` dispatch of `main_pass.slang`, one thread traces a whole
  path in a register-resident bounce loop.
- **[Wavefront.md](Wavefront.md)** — the same estimator torn across many small
  per-stage / per-material kernels connected by GPU queues; better material
  coherence and small enough to compile on MoltenVK.

Both are unbiased and A/B-verified to match; see
[Wavefront.md § Megakernel vs wavefront](Wavefront.md#megakernel-vs-wavefront)
for the side-by-side comparison and the rationale for having both. The section
below describes the megakernel GPU flow in detail.

---

## High-Level Pipeline

Three entry points share the same renderer core:

![High-level pipeline: GLFW / Qt / Web front-ends feed Renderer.py, which dispatches Vulkan compute to swapchain or headless readback.](diagrams/high_level_pipeline.svg)

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

![GPU execution flow: mainImage generates a ray, traces the scene, branches to BDPT (flat first-hit) or PathTracer, then guards, accumulates, tonemaps, and composites overlays into outputBuffer.](diagrams/gpu_execution_flow.svg)

The detailed per-step substructure (furnace/mesh/SDF trace dispatch, BDPT eye/light
walk + (s,t) connections + s=1 splat, the path tracer's per-bounce
`evaluateBounce` → FLAT/SKIN/DEBUG dispatch, Russian roulette, sphere-light MIS):

- **`traceScene`** — furnace → unit sphere; mesh → `marchHeadMesh()`; SDF → `marchHead()`.
- **BDPT** — eye walk (FlatMaterial) + light walk (sphere/emissive/dir) + (s,t)
  connections + light-tracer splat (s=1) → `lightSplatBuffer`.
- **PathTracer** — cutout-transparency skip loop, then `for bounce 0..5`:
  `evaluateBounce` dispatches FLAT (`allLightsNEE` + sample, optional
  `evalSceneGraph`), SKIN (`evalSkinRadiance` §1–§6), or DEBUG (`0.5 + 0.5·N`);
  Russian roulette after bounce 0; sphere-light MIS on the BSDF ray.
- **Post** — NaN/inf/neg guard → running-mean accumulation (+ BDPT light-splat
  mean, Q22.10 → float) → exposure (2^EV) → tonemap → sRGB → furnace overlay →
  gizmo line composite (binding 22) → HUD alpha → `outputBuffer`.

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

### Flat Material BSDF (`materials/flat/flat_material.slang` + `flat_lobes.slang`)

![Flat BSDF unified lobe model](diagrams/flat_bsdf_lobes.svg)

Implements `IMaterial` — `sample()` (draw a bounce direction) and `evaluate()`
(response + solid-angle pdf, consumed by NEE, BDPT connections + reverse pdfs,
ReSTIR, and the directional-proposal mixture). Both walk **one** lobe set
(`{coat, spec, diffuse}`) over a single param source, so `sample().pdf ==
evaluate().pdf` structurally and `evaluate().response / evaluate().pdf` reduces to
the bounded native per-lobe weight (`F·G₁` for the GGX lobes, the diffuse albedo
term for Lambert). This makes one canonical BSDF for the path tracer **and** BDPT
in **both** megakernel and wavefront modes. The lobe model lives in
`flat_lobes.slang` (`flatBsdfPdf`, `flatBsdfResponse`, the per-lobe sampler
dispatch); `flat_material.slang` assembles it. BSDF layers:

- Opacity / refraction (Fresnel-weighted reflect/refract split; delta lobe).
  Cutout vs alpha-blend opacity are split: cutout discards below
  `opacityThreshold`, alpha-blend attenuates — matching UsdPreviewSurface
  semantics
- Clear coat (GGX VNDF, coat-color tinting)
- Specular / diffuse MIS split (Schlick F0, luminance-weighted probability) —
  GGX specular uses VNDF sampling (`samplers/ggx.slang`), diffuse is Lambert
- **Per-lobe runtime-pluggable sampler seam** — each lobe resolves a sampler id
  to a draw/density strategy, defaulting to native (2023 spherical-cap VNDF for
  coat/spec, cosine for diffuse). The host registry (`sampling/lobe_samplers.py`)
  also ships the Heitz-2018 basis-form VNDF (coat/spec — a different warp of the
  *same* GGX visible-normal distribution, so its pdf is shared and parity is
  structural) and uniform-hemisphere (diffuse). `sample()` and `evaluate()` read
  the same per-lobe id from `fc.flatLobeSamplers`, so pdf agreement — hence
  unbiasedness and the bounded `F·G₁` weight — holds for *any* registered
  strategy; only `flatLobeSamplers`' diffuse byte changes a pdf (cosine vs
  uniform). Selectable per lobe via `--lobe-samplers` / the GUI. Adding a strategy
  is a dispatch case in `flat_lobes.slang` + a registry entry — `sample()` /
  `evaluate()` stay untouched
- Cutout alpha masking via `isCutoutTransparent()` (in `flat_shading.slang`)
- **UsdPreviewSurface textures** — per-input channel selection (`channelMask`),
  normal-map `scale`/`bias` (`normalScale`/`normalBias`, for OpenGL vs DirectX
  Y convention), and wrap modes flow from each material's `TextureBinding`
  (binding 14 bindless textures)
- **MaterialX graph evaluation** when `materialTypes[id]` packs a graph
  slot — `evalSceneGraphBaseColor(materialId, hit, ...)` (generated module)
  drives the lobe model's albedo before the BSDF math runs

The full MaterialX `std_surface` closure (`evalStdSurfaceBSDF`, binding-19
`StdSurfaceParams`) is **no longer** used by the path-traced / BDPT estimator — it
is retained only for the raster `preview_pass`. Unifying `evaluate()` onto the
same lobe model `sample()` draws from removed the proposal-mixture bias on layered
coat+metal materials (brass under the BSDF+Env / Env presets).

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

![MaterialX nodegraph pipeline: USD MaterialX → CompiledMaterial → MaterialXGenSlang emits graph_<hash>.slang and generated_materials.slang switch-dispatch; ComputePipeline runs codegen then Slang→SPIR-V with an mtime-LRU cache, skipping rebuild when the graph set is identical.](diagrams/materialx_pipeline.svg)

Details: filename inputs are replaced with bindless slot indices via `TexturePool`
(binding 14); `evalSceneGraph_<hash>(hit, params)` is switch-dispatched by
`generated_materials.slang`; the SPV cache key is `source hash + entry point`
(≤32 entries); the texture pool is repopulated after each rebuild.

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

### Transform gizmo (`gizmo.py`)

`TransformGizmo` tracks one selected scene instance — any baked instance,
including analytic gprims (Sphere/Cube/Cylinder/…) the loader tessellates,
not just `UsdGeom.Mesh` prims — and has four modes —
rotate and translate, each in world or local space — cycled with `Space`
(`(index+1) % 4`, grouped by type). Rotate modes draw three orthogonal
rings, translate modes draw three axis arrows, and a `W`/`L` glyph above
the pivot hints the coordinate space. World modes align to the canonical
X/Y/Z axes; local modes align to the instance's current orientation.
Rotation drag is a true axis-angle rotation about the (world or local)
ring axis composed as a matrix and re-decomposed to Euler; translate drag
projects the mouse onto the screen-projected axis. The renderer rebuilds
the line list per frame and uploads it to binding 22; `main_pass.slang`
draws each segment as an anti-aliased line over the final tonemapped
image. The active mode persists in `~/.skinny/settings.json`.

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

> Full signatures, return types, and examples for the whole programmatic
> surface are in **[PythonAPI.md](PythonAPI.md)**. This section is the
> architectural overview.

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

![Session lifecycle: a browser connect builds a SkinnySession (headless VulkanContext, Renderer, VideoEncoder, render thread); disconnect runs cleanup (thread join, encoder close, renderer cleanup, ctx destroy).](diagrams/session_lifecycle.svg)

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
| 19 | StructuredBuffer | StdSurfaceParams (256 B each) — raster `preview_pass` only; the path-traced / BDPT flat BSDF uses the `flat_lobes` model, not `evalStdSurfaceBSDF` | `bindings.slang` |
| 20 | StructuredBuffer | DistantLight (analytic distant lights) | `scene_lights.slang` |
| 21 | RWStructuredBuffer | BDPT light-splat buffer (Q22.10 uint per R/G/B) | `bindings.slang` |
| 22 | StructuredBuffer | Transform-gizmo line segments | `gizmo.py` |
| 23 | StructuredBuffer | Lens elements (thick-lens stack, float4) | `cameras/thick_lens.slang` |
| 24 | StructuredBuffer | Per-radius exit-pupil bounds (float4) | `cameras/thick_lens.slang` |
| 30 | RWStructuredBuffer | Tool readback (float4) — scene pick / BXDF / BSSRDF probe | `bindings.slang` |
| 31 | StructuredBuffer | Env importance-sampling marginal CDF (ENV_H+1 floats) | `environment.slang` |
| 32 | StructuredBuffer | Env importance-sampling conditional CDF (H×(W+1) floats) | `environment.slang` |
| 33 | StructuredBuffer | Neural-proposal flat Linear weights (`NF_WT`, row-major — `float` by default, `half` in the fp16 precision modes) | `sampling/neural_proposal.slang` |
| 34 | StructuredBuffer | Neural-proposal flat Linear biases (`NF_WT` — `float`/`half`) | `sampling/neural_proposal.slang` |
| 35 | StructuredBuffer | Neural-proposal per-Linear-layer headers (`NfLayerHeader`: weightOffset, biasOffset, inDim, outDim — precision/size-agnostic) | `sampling/neural_proposal.slang` |
| 36 | RWStructuredBuffer | Neural training-record append buffer (`PathRecord`, 64 B) — written by the `mainImageRecord` dump entry **and** the wavefront path integrator (when `fc.recordMode` is set) | `integrators/path_record_common.slang` |
| 37 | RWStructuredBuffer | Record append counter (`uint[2]` = `[count, capacity]`) | `integrators/path_record_common.slang` |

Bindings **25–29** are reserved for the MaterialX nodegraph buffers
(`GRAPH_BINDING_BASE = 25`), so the neural-proposal weight buffers sit at **33+**,
above the graph range, the tool buffer (30), and the env CDFs (31/32). All three
are **always bound** — the renderer seeds them with a full-sized all-zero ("dummy")
net so the inline flow inverse referenced by `sampling/proposal.slang` has valid
descriptors on every pipeline, **including the megakernel** (which never sets the
neural bit); real per-scene weights overwrite them when the neural proposal is
activated. The full reference is in
[Wavefront.md § Neural directional proposal](Wavefront.md#proposal-seam-neural-directional-proposal-proposal-bit2-wavefront-only).

The network **size and precision are build-time configurable** (study change
`neural-precision-size-study`): bindings 33/34 keep their slots but their **element
type follows `NF_WT`** — `float` in the default fp32 mode, `half` in the
fp16-storage / fp16-compute modes (the host casts the fp32 NFW1 file to half at
upload, halving the GPU footprint). Their byte size tracks the configured
`(layers, bins, hidden)`. The header buffer (35) is precision- and size-agnostic.
No binding slot moves; the shader's `NF_WT`/`NF_CT` aliases + `NF_LAYERS/BINS/HIDDEN`
`#ifndef` defaults reproduce the shipped net byte-for-byte when no override is
given. See [Wavefront.md § Neural size & precision](Wavefront.md#neural-size--precision-tuning-neural-precision-size-study).

Under **`--neural-handoff interop`** (online training, change
`neural-online-training`) bindings 33/34/35 are allocated as **externally-shared
memory** (`VK_KHR_external_memory`, **dedicated allocation** — required for the
CUDA import on NVIDIA) so the CUDA trainer can write freshly-baked weights (33)
and biases (34) straight into them with no CPU round-trip — the slots and element
types are unchanged, only the buffers' memory backing differs. A companion
exportable **timeline semaphore** (`VK_KHR_timeline_semaphore` +
`VK_KHR_external_semaphore_win32`/`_fd`) orders the CUDA write against the Vulkan
read so a frame never tears. The default `--neural-handoff file` keeps them as
ordinary device-local buffers the host re-uploads on a hot-reload. See
[Online neural training](#online-neural-training).

Bindings **36/37** back the per-vertex training-record stream
(`PathRecord`/`emitRecord`, shared in `integrators/path_record_common.slang`).
Two producers append to them: the offline dump via a second megakernel entry
`mainImageRecord` (`Renderer.dump_path_records` → a `.nrec` file), and — for the
**live online-training drain** — the wavefront path integrator itself, which
emits the same records during the normal render whenever `fc.recordMode` is set
(`wavefront/wf_records.slang`; a per-lane vertex stack in the path pass's own
set-1 bindings 9/10 carries the snapshots, splatted at termination). `mainImage`
never references 36/37 (dead-stripped → byte-identical), so they are seeded with
1-element dummies and only reallocated to per-frame capacity during a dump or
while the wavefront drain is armed. `Renderer.drain_path_records_to_replay` is
source-selectable (`_record_source`: `auto` → wavefront for the wavefront path
integrator, else the megakernel dispatch): the wavefront source needs **no**
megakernel dispatch — removing the ~400 s-compile / 2 s-TDR seam that loses the
device on NVIDIA/Windows — and reads the buffers the render already filled via
the shared `records_from_buffer` reader. See
[Online neural training](#online-neural-training).

Light uniforms (part of UBO, not separate bindings):
- `lightDirection` (float3) — analytic directional light toward-light vector
- `lightRadiance` (float3) — analytic directional light colour × intensity

`ProceduralParams` (formerly binding 20) was removed; procedural flat colour is
now derived inside `flat_shading.slang` without a dedicated buffer.

### Wavefront pass-local descriptor sets (set 1)

The wavefront passes bind the scene set above as **set 0** and add a pass-local
**set 1** for their stream state (these are NOT part of the scene set):

| Set 1 binding | Owner | Content |
|---|---|---|
| 0 | `WavefrontPathPass` / `RestirDiPass` | `WavefrontPathState[]` (per-lane path state) |
| 1 | `WavefrontPathPass` / `RestirDiPass` | `HitInfo[]` (per-lane primary/bounce hit) |
| 2–7 | `WavefrontPathPass` | counting-sort queues (lane-slot / counts / offsets / queue / cursor / indirect args) |
| 8 | `WavefrontPathPass` | `WfNeuralSample[]` (per-lane neural forward sample `{wi, pdf, version, valid}`, 32 B) — written by the neural pre-pass, read by the flat shade |

The **neural pre-pass** (`WavefrontNeuralProposalPass`) binds set 0 verbatim and a
3-binding set 1 of its own (0 path-state, 1 hit, 2 the `wfNeural` output buffer above);
see [Wavefront.md § Neural directional proposal](Wavefront.md#proposal-seam-neural-directional-proposal-proposal-bit2-wavefront-only).

**ReSTIR DI** (`RestirDiPass`) uses its own set-1 layout — it shares bindings 0–1
(path-state + hit, over the same buffers as the path pass) and adds three
ReSTIR-owned per-pixel buffers:

| Set 1 binding | Owner | Content |
|---|---|---|
| 2 | `RestirDiPass` | `Reservoir[]` A (ping-pong; fill writes, spatial reads) |
| 3 | `RestirDiPass` | `Reservoir[]` B (ping-pong; spatial writes, resolve reads; persists across frames for temporal) |
| 4 | `RestirDiPass` | G-buffer `{pos, normal}[]` (spatial-neighbour domain check; per-neighbour material is re-loaded from `wfHits` for the GRIS p̂ re-eval) |

`RestirPC` push constant (36 B scalar): `streamSize, flags (bit0 spatial / bit1
temporal / bit2 biased), mLight, spatialK, spatialRadius, normalThresh,
depthThresh, mCap, mBsdf`.

The full ReSTIR DI reference — pipeline stages, equations, the equation→shader
map, design choices, and GUI controls — is in **[ReSTIR.md](ReSTIR.md)**.

---

## Online neural training

The neural directional proposal (the `{bsdf,neural}` wavefront proposal) can be
trained **continuously while the scene animates**, so the net adapts instead of
staying frozen on a per-scene offline bake (change `neural-online-training`,
Stage 2). The renderer runs a four-stage loop whose only render-thread cost is
the frame-end weight swap; training itself happens off the render path.

![Online neural training loop: the renderer emits path records to bindings 36/37, a recency-weighted ReplayBuffer feeds the NeuralTrainer, the NeuralWeightPublisher stages weights through a file or interop backend, and the frame-end swap promotes them into the binding-33/34/35 weight buffers while bumping networkVersion.](diagrams/neural/online_training_loop.svg)

1. **Drain** — `Renderer.drain_path_records_to_replay()` reads the per-vertex
   `PathRecord`s the producer appended to bindings 36 (append) / 37 (counter)
   into a recency-weighted `ReplayBuffer`
   (`sampling/neural_replay.py`), via the shared reader `records_from_buffer`
   (`sampling/path_records.py`). The record producer is the `mainImageRecord`
   **megakernel**, which device-losts under the 2 s TDR watchdog on
   NVIDIA/Windows, so the live GPU drain is an NVIDIA-box seam (a follow-up
   proposal `wavefront-native-path-records` moves emission onto the wavefront
   path to remove it).
2. **Train** — `Renderer.online_train_and_publish()` runs one warm-started cycle
   of `NeuralTrainer.train_cycle` (`sampling/neural_trainer.py`):
   contribution-weighted MLE on the replay batch, reusing `spline_flow`'s
   `ConditionalSplineFlow2D` + `render_records` loss. Device branch: CPU/MPS for
   CI, CUDA + `autocast(fp16)` + `GradScaler` on the NVIDIA box (linear GEMMs in
   fp16 on tensor cores, the RQ-spline math in fp32); torch-free venvs fall back
   to a placeholder. It bakes the new weights and `publish()`es them.
3. **Publish** — a `NeuralWeightPublisher` (`sampling/neural_handoff.py`) stages
   the pending weights through one of **two handoff backends**, selected by
   `--neural-handoff` (env `SKINNY_NEURAL_HANDOFF`, persisted):
   - **`file`** (default, `neural_handoff_file.py`) — the trainer writes an NFW1
     file, the renderer hot-reloads it: a CPU round-trip that works on **any**
     platform.
   - **`interop`** (`neural_handoff_interop.py`) — CUDA writes weights (33) and
     biases (34) straight into the Vulkan-exported buffers via
     `cudaImportExternalMemory` → `cudaExternalMemoryGetMappedBuffer` →
     `cudaMemcpyAsync`, with no CPU round-trip, then signals the exported timeline
     semaphore at the staged version (`cudaSignalExternalSemaphoresAsync`). The
     real-time path, **CUDA-only** (needs `cuda-python`) and guarded — it raises a
     clear `NotImplementedError` and `--neural-handoff file` is used instead where
     CUDA or the external-memory Vulkan extensions are absent. Implemented and
     verified on an RTX 4090 (`tests/test_neural_interop.py`); the interop
     `publish()` is **~54x faster** than the file backend's NFW1 round-trip
     (~0.5 ms vs ~29 ms, `tests/bench_neural_online.py`).
4. **Swap** — `Renderer._online_frame_end_swap()` runs at the **frame boundary**
   (after the fence wait in `render_headless`, after present in `render`):
   `publisher.swap()` promotes pending→render and `networkVersion` is incremented
   in both `FrameConstants.neuralNetworkVersion` **and** the
   `WavefrontNeuralProposalPass` push-constant stamp. For the **file** backend the
   swap also `_apply_render_weights`-uploads the new weights to bindings 33/34/35;
   for **interop** there is no re-upload — `acquire_for_render()` returns no
   host-side weights because the CUDA write already populated the bound buffers, so
   the swap only **host-waits the timeline** to the staged version (the CUDA write
   is provably resident) and re-stamps the version.

Render weights stay **frozen during a frame**, so each sample's density is always
evaluated against the network version that drew it. An asynchronous swap
therefore raises **variance only, never bias** — mixture-MIS unbiasedness is
preserved exactly as it is for an untrained net. The wavefront-side commitment
discipline is detailed in
[Wavefront.md § Online neural training: frame-end weight swap](Wavefront.md#online-neural-training-frame-end-weight-swap).

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
| `gizmo.py` | `TransformGizmo` | Transform gizmo math (rotate/translate × world/local) + line-list buffer |
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

![Shader module dependency graph: common.slang feeds interfaces/bindings/scene-trace, which feed cameras/samplers/lights and the material implementations (flat, skin, debug, generated graphs), which feed the path and BDPT integrators, which feed main_pass.slang.](diagrams/shader_dependency_graph.svg)

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
| ... | uint | proposalMask; uint reuseMode; float4 proposalAlpha (scene-sampling seam) |
| ... | uint | flatLobeSamplers — per-lobe flat-BSDF sampler ids, 8 bits/lobe (`coat \| spec<<8 \| diff<<16`; 0 = native). Unpacked by `flat_material.slang`; no new binding |
| ... | float3 | sceneBoundsMin; float3 sceneBoundsExtent — scene AABB for the neural-proposal condition's position normalisation |
| ... | uint | neuralNetworkVersion — active frozen-net version (baseline 0; per-sample network-version hook for future online training) |

`cameraType` was removed — camera selection is implied by `numLensElements`
(0 ⇒ pinhole). `exposure` and `tonemapMode` are post-process knobs and do not
reset accumulation. The neural-proposal scalar tail (`sceneBoundsMin` /
`sceneBoundsExtent` / `neuralNetworkVersion`) brings the UBO to **508 B** (was
480); it is read only when the neural proposal bit is active, so the default
`{bsdf}` path stays bit-identical.

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
scene_trace.slang        scene_lights.slang      nee.slang
mesh_head.slang          sdf_head.slang
environment.slang        volume_render.slang
mtlx_closures.slang      mtlx_std_surface.slang  mtlx_noise.slang
mtlx_gen_shim.slang      generated_materials.slang
debug_line.slang
cameras/{pinhole.slang, thick_lens.slang}
materials/debug_normal_material.slang
materials/flat/{flat_material.slang, flat_lobes.slang, flat_shading.slang}
materials/skin/{skin_material.slang, skin_bssrdf.slang, skin_shading.slang,
                skin_direct.slang, skin_ibl_specular.slang, skin_ibl_diffuse.slang,
                skin_volume.slang, skin_transmission.slang, skin_hair_sheen.slang,
                detail.slang}
samplers/{ggx.slang, lambert.slang, uniform_sphere.slang,
          henyey_greenstein.slang, mis_combine.slang}
sampling/{proposal.slang, reuse.slang,               # scene-sampling seam
          neural_flow.slang,                          # pure spline flow (coupling/RQ/MLP, fwd/inverse)
          neural_proposal.slang}                      # renderer adapter (weight buffers 33/34/35, world map)
lights/{sphere_light.slang, emissive_triangle_light.slang, directional_light.slang}
integrators/{path.slang, bdpt.slang,
             path_record.slang}                       # mainImageRecord training-record dump (5.1)
wavefront/{wavefront_path.slang, wavefront_bdpt.slang, wf_shade_common.slang,
           flat_bounce.slang, wavefront_state.slang,
           neural_proposal_pass.slang,                # WavefrontNeuralProposalPass pre-pass
           build_args.slang, scatter.slang, compaction.slang, indirect_paint.slang}
restir/{reservoir.slang, light_ris.slang, restir_primary.slang}
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
