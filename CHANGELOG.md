# Changelog

All notable changes to Skinny are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Rendering

- Unified flat / `std_surface` BSDF onto one composable lobe set —
  `FlatMaterial.sample()` and `.evaluate()` now share `{coat, spec, diffuse}`
  (`materials/flat/flat_lobes.slang`), so `sample().pdf == evaluate().pdf` and
  `response / pdf` stays bounded (`F·G₁` / Lambert, no clamp). This is the
  canonical BSDF for the path tracer **and** BDPT in both megakernel and wavefront
  modes, and removes the directional-proposal-mixture bias on layered coat+metal
  materials (brass under the BSDF+Env / Env presets: +4.6% → −0.2% vs the BDPT
  reference; megakernel and wavefront converge identically). The MaterialX
  `std_surface` closure (`evalStdSurfaceBSDF`, binding 19) is retained only for the
  raster preview pass. Each lobe carries a runtime-pluggable sampler id (native
  strategies only for now) — the seam a later `per-lobe-sampler-registry` change
  populates with a host registry + alternative samplers.

- Per-lobe sampler registry for the flat / `std_surface` BSDF — populates that
  `samplerId` seam. Each lobe's draw/density strategy is runtime-selectable per
  lobe (`--lobe-samplers coat=…,spec=…,diff=…` / GUI / `SKINNY_LOBE_SAMPLERS`),
  folded into one `FrameConstants.flatLobeSamplers` uint with **no** new
  descriptor bindings. Registered alternates: the Heitz-2018 basis-form VNDF
  (coat/spec — a different warp of the *same* GGX visible-normal distribution, so
  the pdf is shared and converged radiance is identical to native, mega ≡ wave,
  PT ≡ BDPT) and uniform-hemisphere (diffuse — unbiased, bounded weight,
  demonstrably higher-variance than cosine). `sample()` and `evaluate()` read the
  same per-lobe id so unbiasedness holds for any strategy; `flatBsdfResponse`
  (= f·cos) is sampler-invariant. Registry in `sampling/lobe_samplers.py`; gate in
  `tests/test_sampling_parity.py`.

- Neural directional proposal (wavefront-only) — a learned, position-conditioned
  rational-quadratic **neural spline flow** (proposal bit2,
  `--proposals bsdf,neural` / GUI "BSDF + Neural") that proposes the BSDF bounce
  direction with an exact solid-angle pdf and MIS-mixes into the scene-sampling
  proposal seam. Frozen, offline-trained per scene (standalone `spline_flow` repo)
  in the **NFW1** weight format. Architecture is **Option A** — a compute pre-pass
  (`WavefrontNeuralProposalPass`, `wavefront/neural_proposal_pass.slang`) draws one
  forward sample per live lane between scatter and shade; the flat shade kernel
  reads it and evaluates the arbitrary-direction inverse pdf inline
  (`sampling/{neural_flow,neural_proposal,proposal}.slang` + `nee.slang`). Scoped to
  the flat wavefront shade kernel; unbiased regardless of net quality
  (`proposalWeights` renormalises, the same effective weights drive the bounce and
  NEE companion pdfs). New weight buffers at **bindings 33/34/35** (above the
  MaterialX graph range) — always bound with an all-zero dummy net so the inline
  inverse resolves everywhere; the megakernel strips the bit and falls back to its
  analytic proposal subset (mirroring ReSTIR DI → identity). FrameConstants gained
  a scalar tail (scene AABB + net version), UBO now 508 B.

- Neural proposal — offline training pipe (Stage 1b/1c). `Renderer.dump_path_records`
  emits per-vertex `(position, normal, wo, wiLocal, contribution)` training records
  to a `.nrec` file via a second megakernel entry `mainImageRecord`
  (`integrators/path_record.slang`, an RR-free path tracer that backward-attributes
  the tail radiance `contribution = (L_final−L_k)/beta_in_k = f·cos·Li`); records
  land on new **bindings 36/37** (`mainImage` never references them → byte-identical).
  The standalone `spline_flow/render_records.py` trains the flow from those records by
  contribution-weighted MLE (`q ∝ f·Li·cos`) using the exact `neuralCondition`
  encoding and bakes NFW1. Verified end-to-end on Mac MPS (4.36M Cornell records →
  trained net, pdf ∫≈1). The equal-time gate is **measured, not won on Mac**: the
  net is unbiased (mixture-MIS) but the MLP pre-pass is ~28× a bsdf bounce on
  MoltenVK/MPS (the deferred CUDA-perf goal) and the flat ceiling-lit Cornell box is
  broad-indirect (cosine already near-optimal), so the guide ≈ cosine and adds a
  firefly tail with no offsetting win. Follow-ups: GPU-optimised inference,
  guiding-iteration training, a concentrated-indirect scene, a pdf-floor / lower-α
  firefly measure.

### Fixed

- USD scenes used a degenerate `(0,1)` AABB for the neural-proposal condition's
  position normalisation (the per-frame `Scene` snapshot has no instances for USD —
  geometry streams straight to the GPU); `_neural_scene_bounds` now falls back to the
  streamed `_usd_scene` instances, used by both inference and the dump header.
- `Renderer.cleanup()` never freed the env importance-sampling CDF buffers
  (bindings 31/32) — fixed (surfaced by the record-dump's clean-teardown check).
- ReSTIR DI reuse mode (wavefront-only) — reservoir resampling of primary-hit
  direct lighting over the unified light set (sphere + emissive-triangle + env,
  light- and BSDF-sampled candidates) with deferred visibility. Spatial reuse
  uses the unbiased generalized balance heuristic (GRIS) and reduces variance on
  many-light scenes; the path tracer's depth-0 BSDF-hits-light terms are gated so
  ReSTIR owns primary direct (converges to stock NEE). Selectable regimes
  (default Spatial only; Spatial+Temporal / Temporal only are progressive-limited,
  reprojected temporal is a follow-on), a biased ΣM toggle, and live tuning
  (candidate counts, neighbours, radius, M-cap). Capability-gates to identity on
  megakernel/Metal. Built on the pluggable scene-sampling reuse seam.
- Environment importance sampling: equirect HDR sampled by a sin θ-weighted
  2D piecewise-constant distribution (CDF buffers at bindings 31/32) for env
  next-event estimation + MIS — both the path tracer and BDPT consume it
- GGX specular now uses visible-normal (VNDF) importance sampling
  (Heitz 2018/2023); the BRDF×cos/pdf weight reduces to F·G₁, eliminating
  grazing-angle specular fireflies
- BDPT connections evaluate the real `standard_surface` BSDF instead of a
  Lambertian approximation, and use the same env importance sampling as the
  path tracer so the two integrators converge to the same image
- Exposure (EV stops) and a selectable tonemap operator (ACES filmic /
  Reinhard / Hable / linear) as post-process knobs that do not reset
  accumulation

### Materials

- OpenPBR material support in the USD loader, including resolution of
  connected shader inputs to their authored constant
- UsdPreviewSurface texture bindings: per-input channel selection, normal-map
  `scale`/`bias` (OpenGL vs DirectX Y convention), wrap modes, and source
  colour space, carried on a new `TextureBinding` (`scene.py`)
- `FlatMaterialParams` grew 96 B → 128 B to carry `normalScale`, `normalBias`,
  and a packed `channelMask`
- Cutout vs alpha-blend opacity split in `fetchFlatHitData` to match
  UsdPreviewSurface `opacityThreshold` semantics
- Python-authored materials: SlangPile `python_materials/*.py` compile to GPU
  `IMaterial` structs dispatched as material type 3 (id in bits 24–31 of
  `materialTypes`), editable live in the Qt material editor
- Removed the dedicated `ProceduralParams` buffer (was binding 20); binding 20
  is now `DistantLight`

### Animation

- USD animation playback: a `PlaybackClock` maps wall-clock time onto the
  stage's authored time range and re-evaluates *cheap* time-sampled prims each
  frame — transform tracks, camera, and lights — by re-uploading only the TLAS
  instance records / light buffers (no mesh rebake). A load-time animated-prim
  index keeps per-frame cost proportional to the animated set. `current_time_code`
  feeds the accumulation hash, so playback renders at 1 spp and converges when
  paused (`playback.py`, `usd_loader.build_animation_index`)
- A `usd` camera mode follows an animated USD camera; the user can switch back to
  Orbit/Free at any time
- UsdSkel skeletal animation: skinned meshes deform per frame via linear blend
  skinning. CPU computes per-joint matrices (pxr, validated against
  `UsdSkelSkinningQuery.ComputeSkinnedPoints`); a GPU compute pass
  (`shaders/skin.slang`) blends rest vertices into the shared vertex buffer and a
  GPU BVH refit (`shaders/bvh_refit.slang`) keeps the path tracer correct over
  the deformed geometry — no readback. Standalone Vulkan pipelines
  (`vk_skinning.py`) with their own descriptor sets leave `main_pass` untouched;
  a CPU skinning path is the fallback on non-Vulkan backends

### Scene editing

- Runtime scene-graph editing API on `Renderer`: `add_model()` (reference a USD
  file under a parent prim), `remove_node()` (non-destructive deactivation),
  `set_transform()` (fast transform-only resync), `save_edits()`, and
  `list_nodes()`. The loaded `Usd.Stage` is the source of truth; edits are
  authored to an in-memory edit sublayer so the original file is never modified
  until `save_edits()`. `MeshInstance` now carries its `prim_path`, and
  `apply_instance_transform` / `apply_node_enabled` are keyed by prim path
- Scene-graph panels are now live editors in both front-ends (Qt dock + Panel
  card): an "Add model" picker (references a USD file under the selected group
  or `/World`), a "Delete node" action (context-menu + `Delete` key in Qt;
  button in Panel; guarded against synthesized `/Skinny/*` nodes), a "Save edits"
  button, and per-node transform edits that author to the stage (persisted by
  "Save edits"). Add/remove now rebuild the scene graph and re-read lights +
  cameras, so deleting a light or camera drops it and the panels refresh; lights
  carry a `prim_path` so a runtime enable-toggle survives an unrelated edit

### UI and interaction

- ReSTIR controls now live in a dedicated **ReSTIR** sidebar group — the `Reuse`
  selector plus the regime, biased-combine, M-light/M-bsdf, neighbours, radius,
  and M-cap tuning — split out of the general **Render** group and placed
  directly after it. Defined once in `ui/build_app_ui.build_main_ui`
  (`_classify`), so the Qt, web, and debug front-ends pick it up identically; the
  group stays visible at identity reuse so you can switch into ReSTIR from it
- Transform gizmo: the viewport manipulator now has four modes — rotate and
  translate, each in world or local space — cycled with `Space`
  (rotate-world → rotate-local → translate-world → translate-local). A `W`/`L`
  glyph above the pivot hints the coordinate space; rings vs arrows hint the
  type. The active mode persists in `~/.skinny/settings.json`. **`Space` no
  longer toggles the HUD — use `F1`.** Rotation drag is now a true axis-angle
  rotation about the (world or local) ring axis rather than a per-Euler-axis
  add, so world-axis rotation behaves slightly differently (more correct)
- Built-in animation transport (play/pause, normalized time scrubber, fps) in the
  shared control tree, shown only when the loaded stage has animation
- USD-driven Scene Controls: a stage can declare its own control panel via
  `skinny:ui:*` prims (slider/toggle/combo/color). Each control's prefix-typed
  target binds to a renderer parameter (`renderer:`/`mtlx:`), a material input
  (`material:`), or a USD attribute (`usd:`); editing a `usd:` control writes the
  stage and refreshes the live light/transform/camera state. Controls appear in a
  "Scene Controls" section across the Qt, web, and debug front-ends
  (`usd_loader.extract_ui_controls` + `resolve_control_binding`)
- Live Python material editing in the Qt material editor
- Camera debug viewport (`F2`) with frustum, lens rings, focus plane, DOF
  planes, render-area outline, ground grid, mesh wireframes, AABBs, and
  camera-body glyph
- Screen-space HUD inside the debug viewport listing its keyboard
  shortcuts; toggleable with `Space`
- Lens focus overlay (`L`), lens-vignette debug visualisation (`V`),
  zoom-rectangle drag (`Z` to arm, `X` to reset) hotkeys on the main
  window
- Updated on-screen HUD and `H` help text to list the full key set

### Tooling

- Unified render-selection flags across every front-end (`skinny`,
  `skinny-gui`, `skinny-web`, `skinny-render`), defined once in
  `skinny.cli_common`: `--integrator {path,bdpt}`, `--execution-mode
  {megakernel,wavefront}`, and `--bdpt-walk {fused,eye,eye_light}`. The
  interactive front-ends gained `--integrator` (sets the initial integrator,
  still runtime-cycleable); `skinny-render` gained `--execution-mode` /
  `--bdpt-walk`
- The wavefront-bdpt single-kernel subpath build is now `--bdpt-walk fused`
  (was `megakernel`), so `megakernel` names only the execution mode. The old
  `megakernel` walk value is still accepted as a deprecated alias for `fused`
  (CLI, `SKINNY_BDPT_WALK` env, and persisted values)
- Headless render API (`skinny.headless`) and `skinny-render` CLI for offscreen
  USD rendering — accepts a file path or a live `Usd.Stage` mutated per frame;
  saves PNG/JPEG/BMP/EXR/HDR or returns a numpy array; supports USD-time and an
  animation loop. `examples/render_image.py` and `examples/render_turntable.py`
  are thin wrappers over the new API.

### Fixes

- Windowed Vulkan swapchain is now created at the surface's `currentExtent`
  instead of the window point-size. On MoltenVK/Retina the two differ (backing
  pixels), which made `vkAcquireNextImageKHR` return `VK_SUBOPTIMAL_KHR` and the
  windowed app crash on the first frame; the offscreen→swapchain blit already
  scales, so decoupled render resolution is unaffected (`vk_context.py`)

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
