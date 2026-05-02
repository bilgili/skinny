# Skinny — Architecture

Skinny is a single-kernel Vulkan compute path tracer specialised for
physically-based human skin rendering. It combines a three-layer biological
skin model (epidermis / dermis / subcutaneous) with a full MaterialX-based
standard_surface closure tree, all driven from one `[numthreads(8,8,1)]`
compute dispatch.

---

## High-Level Pipeline

```
┌─────────────┐      ┌────────────────┐      ┌──────────────────┐
│  GLFW + UI  │─────▶│ Renderer.py    │─────▶│ Vulkan Compute   │
│  (Tkinter   │      │  packs UBO,    │      │  Dispatch        │
│   panel)    │      │  uploads bufs  │      │  mainImage()     │
└─────────────┘      └────────────────┘      └──────────────────┘
                            │                         │
                     SkinParameters              Camera Ray
                     MaterialX codegen           ─▶ Trace
                     USD scene loading           ─▶ Material Eval
                     Mesh + BVH baking           ─▶ Progressive Accum
                                                 ─▶ Tonemap + sRGB
                                                 ─▶ HUD composite
```

### Per-Frame Render Loop

1. **`app.main()`**: GLFW poll → `InputHandler.update(dt)` → `panel.tick()` →
   `renderer.update(dt)` → `renderer.render()`.
2. **`renderer.update(dt)`**: detects dirty state (camera, params, env), reuploads
   affected buffers, resets accumulation counter on change.
3. **`renderer.render()`**: packs FrameConstants + SkinParams + light data into
   UBO, updates descriptor sets, records command buffer, dispatches
   `ceil(W/8) × ceil(H/8)`, presents via swapchain.

---

## GPU Execution Flow

```
mainImage()                                          main_pass.slang
  ├─ generateCameraRay(fc, pixel, rng)               common.slang
  ├─ traceScene(fc, ray)                             scene_trace.slang
  │    ├─ furnace → unit sphere
  │    ├─ mesh    → marchHeadMesh()                  mesh_head.slang
  │    └─ SDF     → marchHead()                      sdf_head.slang
  ├─ evalMaterial(hit, ray, rng)                     material_eval.slang
  │    ├─ MATERIAL_TYPE_FLAT  → FlatMaterial          flat_material.slang
  │    ├─ MATERIAL_TYPE_DEBUG → DebugNormalMaterial   debug_normal_material.slang
  │    └─ default (SKIN)      → SkinMaterial          skin_material.slang
  ├─ NaN / inf / negative guard
  ├─ progressive accumulation (running mean)
  ├─ ACES filmic tonemap → sRGB gamma
  ├─ furnace energy-violation overlay (pink)
  └─ HUD alpha composite → outputBuffer
```

---

## Pluggable Interface Architecture

All interfaces live in `shaders/interfaces.slang`. Dispatch strategies are
chosen to avoid existential warp serialisation on GPUs.

### ISampler

```
sampleDirection(float2 u) → float3
pdf(float3 L)             → float
```

Sampler state (N, V, roughness, g, etc.) is stored in struct fields. Generic
parameter on estimators — compile-time monomorphised, zero runtime cost.

| Implementation | File | Purpose |
|---|---|---|
| `GGXSampler` | `samplers/ggx.slang` | Microfacet specular importance sampling |
| `LambertSampler` | `samplers/lambert.slang` | Cosine-hemisphere diffuse sampling |
| `UniformSphereSampler` | `samplers/uniform_sphere.slang` | MIS companion sampler |
| `HenyeyGreensteinSampler` | `samplers/henyey_greenstein.slang` | Phase-function importance sampling |

MIS utilities in `samplers/mis_combine.slang`: `misPrimaryWeight<TA,TB>`,
`misCompanionWeight<TA,TB>` (power heuristic).

### IMaterial

```
evalRadiance(Ray ray, inout RNG rng) → float3
```

Tag-switch monomorphisation in `material_eval.slang` — each `case` loads a
concrete struct and calls `runEstimators<TM>(mat, ray, rng)`. Never used as
an existential value (divergent material hits in a warp would serialise).

| Implementation | File | Type Code |
|---|---|---|
| `SkinMaterial` | `skin_material.slang` | 0 (default) |
| `FlatMaterial` | `flat_material.slang` | 1 |
| `DebugNormalMaterial` | `debug_normal_material.slang` | 2 |

Material type encoding in `materialTypes[id]` (binding 16):
- bits 0–7: type code
- bits 8–9: scatter mode for skin (bit 0 = BSSRDF, bit 1 = volume)
- bits 10–31: reserved

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

Uniform per dispatch — selected by `fc.integratorMode` in `main_pass.slang`.
All three currently delegate to `evalMaterial()` pending divergence.

| Implementation | File | Mode |
|---|---|---|
| `PathTracer` | `integrators/path.slang` | 0 |
| `MISPathTracer` | `integrators/mis.slang` | 1 |
| `LightTracer` | `integrators/light.slang` | 2 |

### Adding a New Material (One-File Add)

1. Create `shaders/my_material.slang` — `struct MyMat : IMaterial { ... }` + `loadMyMat(HitInfo)`.
2. In `material_eval.slang` — add `import my_material;` and one `case` line.
3. In `renderer.py` — add `MATERIAL_TYPE_MYMAT = N` constant + packing branch.

---

## Skin Material Pipeline

### Three-Layer Biological Model (`skin_bssrdf.slang`)

```
         ┌──────────────────────┐
         │   Epidermis          │  melanin absorption (Donner & Jensen 2006)
         │   thickness, σs, g   │
         ├──────────────────────┤
         │   Dermis             │  hemoglobin absorption (oxy/deoxy)
         │   + tattoo pigment   │  optional ink overlay
         ├──────────────────────┤
         │   Subcutaneous       │  fixed-optics fat layer
         │   σa, σs, g          │
         └──────────────────────┘
```

`SkinLayerStack` (3 × `SkinLayer`) built from `MtlxSkinParams` (164 bytes, 27
fields). Each layer has: σa, σs, thickness, g (anisotropy), IOR.

Key functions:
- `melaninAbsorption(melanin)` — wavelength-dependent absorption
- `hemoglobinAbsorption(hemoglobin, oxygenation)` — oxy/deoxy spectral mix
- `evaluateSubsurfaceScattering()` — Christensen-Burley diffusion BSSRDF
- `evaluateSkinSpecular()` — GGX specular with Schlick fresnel

### Skin Material Orchestrator (`skin_material.slang`)

Chains 7 estimator terms in fixed RNG order:

| Term | Estimator | File |
|------|-----------|------|
| §1 | Direct + area + emissive lights | `estimators/skin_direct.slang` |
| §2 | IBL specular (GGX sampling) | `estimators/skin_ibl_specular.slang` |
| §3 | IBL diffuse via BSSRDF | `estimators/skin_ibl_diffuse.slang` |
| §4 | Volume march (delta tracking) | `estimators/skin_volume.slang` |
| §5 | Thin-geometry translucency | `estimators/skin_transmission.slang` |
| §6 | Vellus hair sheen | `estimators/skin_hair_sheen.slang` |
| §7 | Stored light-subpath BDPT | `estimators/skin_bdpt.slang` |

`buildSkinShading()` handles detail maps (normal, roughness, displacement from
bindings 9–11), tattoo ink (binding 8), and pore perturbation (Worley noise).

### Flat Material Bounce Loop (`flat_material.slang`)

6-bounce path tracer with:
- Opacity / refraction (Fresnel-weighted reflect/refract split)
- Clear coat (GGX, coat color tinting)
- Specular / diffuse split (Schlick F0, luminance-weighted probability)
- Russian roulette (bounce > 0)
- MIS: sphere light intersection on BSDF-sampled rays via `intersectSphereLights`
- Procedural color via `ProceduralParams` (marble 3D noise)

---

## Volume Rendering (`volume_render.slang`)

Delta-tracking (Woodcock tracking) through heterogeneous skin:

- `layerAtDepth(depth, stack)` — depth-stratified σa/σs/g lookup
- Henyey-Greenstein phase function + importance sampler
- `MAX_VOLUME_STEPS = 128`, `VOLUME_DEFAULT_BOUNCES = 8`
- `mmPerUnit` converts between mm-valued optical coefficients and world-unit
  ray distances (FrameConstants offset +156)

---

## Scene System

### SDF Head (`sdf_head.slang`)

Loomis-proportioned head from SDF primitives (sphere, ellipsoid, capsule,
round box) combined with smooth boolean operators. Sphere-traced with
gradient-based normals. Approximate bounds: x ∈ [-0.85, +0.85],
y ∈ [-0.97, +1.05], z ∈ [-0.90, +1.00].

### Mesh Head (`mesh_head.slang`)

Two-level acceleration structure:
- **TLAS**: loop over `Instance` records (144 B each: two 4×4 matrices +
  BLAS offsets + materialId)
- **BLAS**: SAH BVH per mesh, `BVH_LEAF_SIZE = 4` triangles per leaf
- **MeshVertex**: 32 B (float3 pos + u + float3 nrm + v)
- **BvhNode**: 32 B (AABB + left/count + right/first)
- Ray-triangle: Möller-Trumbore. Ray-AABB: slab test.

### Scene Dispatch (`scene_trace.slang`)

`traceScene()`:
- `furnaceMode` → unit sphere intersection
- `useMesh` → `marchHeadMesh()` (BVH traversal)
- else → `marchHead()` (SDF sphere tracing)

Shadow tests: `visibleSegment()` (point-to-point), `visibleDirectional()`
(point toward infinity).

### USD Loading (`usd_loader.py`)

Walks USD stage for `UsdGeom.Mesh`, `UsdLux` lights (DistantLight,
SphereLight, DomeLight), `UsdGeom.Camera`, `UsdShade.Material` bindings with
UsdPreviewSurface parameter overrides. Converts `metersPerUnit` → `mm_per_unit`.

---

## Descriptor Binding Map

| Binding | Type | Content | Owner |
|---------|------|---------|-------|
| 0 | UBO | FrameConstants + SkinParams + light uniforms | `bindings.slang` |
| 1 | RWTexture2D | Swapchain output (RGBA8) | `bindings.slang` |
| 2 | RWTexture2D | HDR accumulation (RGBA32F) | `bindings.slang` |
| 3 | RWTexture2D | HUD alpha mask (R8) | `bindings.slang` |
| 4 | Sampler2D | HDR environment map (1024×512) | `environment.slang` |
| 5 | StructuredBuffer | Mesh vertices (32 B each) | `mesh_head.slang` |
| 6 | StructuredBuffer | Mesh indices (uint32) | `mesh_head.slang` |
| 7 | StructuredBuffer | BVH nodes (32 B each) | `mesh_head.slang` |
| 8 | Sampler2D | Tattoo map (512×512 RGBA) | `skin_shading.slang` |
| 9 | Sampler2D | Normal detail map (2048²) | `skin_shading.slang` |
| 10 | Sampler2D | Roughness detail map (2048²) | `skin_shading.slang` |
| 11 | Sampler2D | Displacement detail map (2048²) | `skin_shading.slang` |
| 12 | StructuredBuffer | TLAS instances (144 B each) | `mesh_head.slang` |
| 13 | StructuredBuffer | FlatMaterialParams (96 B each) | `bindings.slang` |
| 14 | Sampler2D[16] | Bindless material textures (PARTIALLY_BOUND) | `bindings.slang` |
| 15 | StructuredBuffer | MtlxSkinParams (164 B each, scalar layout) | `skin_shading.slang` |
| 16 | StructuredBuffer | Material type codes (uint32 each) | `bindings.slang` |
| 17 | StructuredBuffer | SphereLight (32 B each) | `scene_lights.slang` |
| 18 | StructuredBuffer | EmissiveTriangle (64 B each) | `scene_lights.slang` |
| 19 | StructuredBuffer | StdSurfaceParams (256 B each) | `bindings.slang` |
| 20 | StructuredBuffer | ProceduralParams (96 B each) | `bindings.slang` |

Light uniforms (part of UBO, not separate bindings):
- `lightDirection` (float3) — analytic directional light toward-light vector
- `lightRadiance` (float3) — analytic directional light colour × intensity

---

## MaterialX Integration

```
skinny_defs.mtlx                     6 custom nodedefs
        │                            (epidermis, dermis, subcut,
        ▼                             scattering_layer, layered_bsdf,
skinny_skin_default.mtlx              layered_vdf)
        │
        ▼
MaterialXGenSlang                    Code generation
        │
        ▼
mtlx/genslang/*.slang                6 generated Slang implementations
        │
        ▼
skinny_genslang_impl.mtlx           Binds nodedefs → Slang source
```

`MaterialLibrary` (`materialx_runtime.py`):
- Loads stdlib + skinny custom libraries
- Runs `MaterialXGenSlang` to produce Slang source from material graph
- Reflects uniform block → `MtlxSkinParams` struct (164 B, 27 fields)
- `pack_material_values()` serialises param overrides to GPU-ready bytes

Generated genslang files:
- `skinny_skin_epidermis_genslang.slang`
- `skinny_skin_dermis_genslang.slang`
- `skinny_skin_subcut_genslang.slang`
- `skinny_scattering_layer_genslang.slang`
- `skinny_skin_layered_bsdf_genslang.slang`
- `skinny_skin_layered_vdf_genslang.slang`

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
`_compile_slang()`. Walks `python_materials/*.py`, calls `build_module()`,
writes to `mtlx/genslang/generated_*.slang`. Failures are non-fatal (debug log
only).

---

## Python Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `app.py` | `InputHandler`, `ParamSpec` | GLFW window, render loop, camera + param hotkeys |
| `renderer.py` | `Renderer`, `SkinParameters`, `OrbitCamera`, `FreeCamera`, `TexturePool` | GPU resource orchestration, per-frame dispatch |
| `vk_context.py` | `VulkanContext`, `SwapchainInfo` | Vulkan 1.3 instance, device, swapchain |
| `vk_compute.py` | `ComputePipeline`, `UniformBuffer`, `StorageImage`, `StorageBuffer`, `SampledImage`, `HudOverlay` | Shader compilation (Slang→SPIR-V), GPU resource types |
| `scene.py` | `Scene`, `Material`, `MeshInstance`, `LightDir`, `LightSphere`, `LightEnvHDR` | Scene description dataclasses |
| `materialx_runtime.py` | `MaterialLibrary`, `CompiledMaterial`, `UniformField` | MaterialX loading, GenSlang codegen, uniform reflection |
| `mesh.py` | `Mesh`, `MeshSource` | OBJ loading, subdivision, displacement, BVH construction |
| `environment.py` | `Environment` | HDR env map loading (.hdr decoder), built-in presets |
| `control_panel.py` | `ControlPanel` | Tkinter UI: arcball light widget, auto-generated sliders |
| `presets.py` | `Preset` | 12 built-in skin presets (Fitzpatrick I–VI × Female/Male) |
| `settings.py` | — | Persistent storage at `~/.skinny/` (JSON) |
| `tattoos.py` | `Tattoo` | Procedural + image-based tattoo loading |
| `head_textures.py` | `TextureStats` | Detail map loading (normal, roughness, displacement) at 2048² |
| `usd_loader.py` | — | USD stage → Scene (meshes, lights, cameras, materials) |
| `fetch_hdrs.py` | — | Downloads CC0 HDRIs from Poly Haven |

---

## Shader Module Dependency Graph

```
                          common.slang
                         ╱     |     ╲
              interfaces    bindings    skin_bssrdf
              ╱  |  |  ╲      |         |       ╲
        ISampler ILight  IMaterial  scene_trace  skin_shading
          |              IIntegrator   |    ╲         |
     samplers/*                    sdf_head mesh_head  ╲
          |                                         estimators/*
     lights/*                                           |
                                                   skin_material ─┐
                                                   flat_material ─┤
                                                   debug_normal ──┤
                                                                  ▼
                                              material_eval.slang
                                                      |
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
| ... | uint | useDirectLight |
| ... | uint | useMesh |
| ... | float | tattooDensity |
| ... | float | envIntensity |
| ... | uint | furnaceMode |
| +156 | float | mmPerUnit |
| ... | uint | detailFlags |
| ... | float | normalMapStrength |
| ... | float | displacementScaleMM |
| +192 | uint | integratorMode |
| ... | uint | numInstances |
| ... | uint | numSphereLights |
| ... | uint | numEmissiveTriangles |

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
- **Furnace mode**: `main_pass.slang` lines 79–86 flag energy-conservation
  violations in pink. Every integrator × every material must converge to L=1.0
  under a white unit-sphere environment.
- **Material dispatch**: always tag-switch monomorphisation (never existential
  `IMaterial`). Keeps warp occupancy uniform.
- **RNG order**: skin estimators are called in fixed sequence so RNG state
  stays pixel-identical across refactors.

---

## File Listing

### Python (`src/skinny/`)

```
__init__.py              app.py                 renderer.py
vk_context.py            vk_compute.py          scene.py
materialx_runtime.py     mesh.py                environment.py
control_panel.py         presets.py              settings.py
tattoos.py               head_textures.py        fetch_hdrs.py
usd_loader.py
```

### SlangPile (`src/skinny/slangpile/`)

```
__init__.py              api.py                 types.py
registry.py              runtime.py             diagnostics.py
verification.py          cli.py                 compiler/__init__.py
compiler/module.py
```

### Shaders (`src/skinny/shaders/`)

```
common.slang             interfaces.slang        bindings.slang
main_pass.slang          scene_trace.slang        material_eval.slang
skin_material.slang      flat_material.slang      debug_normal_material.slang
skin_bssrdf.slang        skin_shading.slang       flat_shading.slang
volume_render.slang      sdf_head.slang           mesh_head.slang
environment.slang        scene_lights.slang       detail.slang
mtlx_closures.slang      mtlx_std_surface.slang   mtlx_noise.slang
```

### Estimators (`shaders/estimators/`)

```
skin_direct.slang         skin_ibl_specular.slang  skin_ibl_diffuse.slang
skin_volume.slang         skin_transmission.slang  skin_hair_sheen.slang
skin_bdpt.slang           flat_direct.slang
```

### Samplers (`shaders/samplers/`)

```
ggx.slang                lambert.slang            uniform_sphere.slang
henyey_greenstein.slang   mis_combine.slang
```

### Lights (`shaders/lights/`)

```
sphere_light.slang        emissive_triangle_light.slang
directional_light.slang
```

### Integrators (`shaders/integrators/`)

```
path.slang               mis.slang                light.slang
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
```
