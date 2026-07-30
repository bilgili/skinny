# Skinny — Scene System

This document covers the scene side of the renderer: USD stage intake, the
scene graph, instancing, lights, textures, skinning, and the camera, lens,
and debug viewport.

For the renderer overview see [Architecture.md](Architecture.md).

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

### Scene intake (`scene_intake.py`, change `scene-intake-interface`)

**`scene_intake.py` is the one interface from a USD stage to the renderer.** It
reads a stage — in full, as a streamed batch, or at a time code — and returns a
`SceneUpdate` value. It holds no reference to the renderer, imports nothing
from it, and never mutates it. `usd_loader.py` stays the extractor underneath;
intake is what the rest of the codebase talks to.

Before this change the renderer reached into the loader through **15
function-local imports of 9 private symbols**, so the module-level import graph
showed no coupling at all. The dependency also ran backwards:
`usd_loader.resolve_control_binding` took a renderer, read its scene, called
`apply_material_override` and set `_usd_live_dirty`. Intake and renderer were
circular at runtime, resolved only by laziness.

**The interface.** Four calls produce an update, one reads a time code, two
serve per-frame needs:

| Call | Produces |
|------|----------|
| `read_stage(path)` | `SceneUpdate` with `pending_prims` unbaked — safe to call off the render thread |
| `read_open_stage(stage, …)` | `SceneUpdate`, meshes baked inline (post-edit resync; `replaces=` marks a force-replace) |
| `adopt_scene(scene)` | `SceneUpdate` wrapping an already-loaded `Scene` (headless / parity harness) |
| `bake_pending(pending_prims)` | Baked instances, yielded in completion order |
| `read_at_time(stage, tc, …)` | `TimeSample` — instance transforms, lights, camera, all up-axis corrected |
| `deform_skinned_mesh(binding, source, t)` | LBS-deformed `MeshSource` with re-smoothed normals |
| `resolve_control_binding(spec, scene=, stage=)` | `ControlBinding` — a *description*, never a write |

`read_at_time` takes a frame number **or** a `Usd.TimeCode`. The sentinel
matters: `Usd.TimeCode.Default().GetValue()` is NaN, so rounding it through a
float silently asks for frame NaN.

**One application path.** `Renderer.apply_scene_update(update)` is the only
place a scene is adopted; it replaced three paths (`set_usd_scene`, the
streaming poll, `_resync_geometry_from_stage`) that each did a different subset
of the work in a different order. The order is stated once:

1. enter the USD-active state (if the update carries a label)
2. carry runtime state off the outgoing scene (resync only)
3. take the stage + its derived state (graph, animation index, clock, up-axis
   rotation, skeletal handle, controls)
4. film clamp, then `mm_per_unit`
5. `_sync_volume_grid`, then `_gen_scene_materials`
6. camera framing + the USD camera follower
7. authored `skinny:ui:default` control values
8. `_upload_usd_scene`
9. `_inject_default_lights_into_scene_graph`, `_refresh_camera_node`
10. bump `_material_version`, `_scene_graph_version`, `_scene_version`

Three of those orderings are load-bearing: runtime state is read *before* the
swap; `mm_per_unit` is final before the grid sync and the grid before the
upload that packs the volume σ folds; the camera is framed before the camera
node snapshots any authored thick lens.

**Per-trigger steps are fields, not branches.** `SceneUpdate` has one
constructor per trigger — `streamed`, `adopted`, `resynced`, `replacing` —
and nothing else fills the flags by hand. A resync keeps the user's camera and
skips control defaults; a full load adopts `mm_per_unit` (unless it is `Scene`'s
`120.0` sentinel) and does not carry runtime state, because
`parameter_overrides` mixes authored loader values with live edits and the old
authored value would beat the newly authored one.

**Runtime-state carry-over is a stated property.** Instance-enabled flags,
light-enabled flags and live material overrides are never authored to USD, so a
stage re-read would drop them — edit a colour, add a light, the edit vanishes
("finding #7"). `Renderer._carry_runtime_state_into` moves them, keyed by the
material's stable prim path with a fallback to its leaf name, so `/ScopeA/Foo`
and `/ScopeB/Foo` do not cross-apply.

**`renderer.scene_version` replaced `id(renderer._usd_scene)`.** Six sites used
object identity as a change token — two `DynamicSection` rebuild tokens, two
Panel repopulate polls, and two renderer-internal caches. An id only changes on
a swap, so it went stale the moment a path mutated the scene in place; that is
precisely why the post-edit path hand-copied eight fields instead of swapping,
and why it forgot the film clamp among them. The counter is bumped once per
applied update — **and once by `_clear_model_state`**, because a scene going
away is a scene change too. With `id()` that transition was noticed by
accident (`id(None)` differs from the old scene's id); a counter has no such
accident, so the clear also drops the stage-derived state it orphans (controls,
animation index, clock, up-axis rotation, skeletal handle), which is the mirror
of `SceneUpdate.replaces_stage_state` on the way in.

**Threading.** The USD streaming thread is a pure producer: it calls
`read_stage` then `bake_pending` and writes nothing to the renderer. Every
renderer write happens in `apply_scene_update` on the render thread.

Gates: `tests/test_scene_intake.py` (intake values, the time-code identity
fixture, and AST source gates that fail if a function-local loader import or a
renderer back-reference returns) and `tests/test_scene_update_apply.py` (the
adoption order and the carry-over, asserted directly rather than as a side
effect of one path).

### USD Loading (`usd_loader.py`)

Walks USD stage for `UsdGeom.Mesh`, `UsdLux` lights (DistantLight,
SphereLight, DomeLight, RectLight, DiskLight), `UsdGeom.Camera`,
`UsdShade.Material` bindings with UsdPreviewSurface, MaterialX, and **OpenPBR**
overrides.
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

The `file` value is resolved through `GetValueProducingAttributes`
(`_resolve_texture_binding`), so a `UsdUVTexture.inputs:file` authored as a
*connection* to a Material interface input — the shape Apple's glTF→USD
conversion and many DCC exporters emit (`file <- Material0.baseColorTexture`) —
resolves instead of dropping to flat white (change `glb-asset-import`, spec
`usd-texture-intake`). A `UsdTransform2d` on a texture's `st` chain
(`_resolve_st_transform`) is captured as `TextureBinding.uv_transform` and
**baked into the mesh UVs at load** (`_bake_uv_transform`), applied in raw USD
st-space before the loader's existing USD→skinny V-convention flip: the net
`flip(T(flip(uvs)))` collapses the glTF `scale (1,-1)`/`translation (0,1)`
V-flip and the convention flip back to the raw glTF texcoords. Identity/absent
transform short-circuits (UV output bit-identical); the per-prim build loop
resolves the material and bakes UVs **before** computing the mesh content hash,
so shared geometry under materials with differing transforms keys distinct mesh
cache entries. The converter that produces such assets from a GLB is
`glb_import.py` (pure-Python pygltflib + pxr), reachable one-call through the
`scene_import_glb` MCP tool.

### Default-Light Synthesis Policy (`renderer.py`)

The central `Renderer.uses_default_lights` decision grants lighting authority
to exactly one source set:

- If the active USD scene has any authored Distant, Sphere, Dome, Rect, or Disk
  light, or an emissive material, only authored USD lighting contributes.
  Presence expresses author intent: zero-intensity and runtime-disabled lights
  still suppress fallback; inactive/deactivated prims do not.
- Otherwise Skinny synthesizes its default DistantLight and built-in IBL
  together. Their own controls can disable contributions while fallback
  authority remains active.

The decision is re-evaluated rather than cached, is gated by the active model
(a retained inactive USD scene cannot affect an OBJ/default head), and is
shared by distant-light upload, environment selection, headless options, UI
visibility, and scene-graph projection. Authored mode without a DomeLight uses
a black environment; fallback `env_intensity` and `direct_light_index` cannot
alter authored sources. Stage resync copies or clears the authored environment,
so adding the first light and removing the last light transition both
contributions and controls at runtime. Furnace mode remains an explicit
diagnostic override.

### Runtime Scene-Graph Editing (`renderer.py`)

The loaded `Usd.Stage` is the authoritative scene model; the flat `Scene` +
GPU buffers are a derived cache. `Renderer._attach_edit_layer()` sets the
stage's **session layer** as the edit target (change `session-edit-layer`), so
every runtime edit is authored there and the original file is never written
until `save_edits()`. The session layer is used because it is stronger than the
whole root layer stack — a root *sublayer* (the earlier design) is weaker than
the root layer and so cannot override a file-authored opinion: `set_transform`
on a prim whose `xformOp:transform` lives in the loaded file would raise a
duplicate-op error and any value it authored would be silently ignored.
`set_transform` authors via `_author_local_transform`, which reuses an existing
single non-inverse `xformOp:transform` op with `op.Set()` (a value-over that
wins from the session layer) and falls back to clear+add only for the fresh /
inverse / multi-op cases (`skinny.usd_edit.author_local_transform`). The editing API — `add_model()` (define an
`Xform` + `AddReference`, optional `validate(stage, added_prim)` callback run
post-recompose/pre-resync so a policy layer can veto and roll back before the
resync pays for itself), `add_primitive()` (change `mcp-scene-structure`:
define one of the six analytic gprims `usd_gprims.tessellate_gprim` meshes,
plus a dedicated bound `UsdShade`/`UsdPreviewSurface` material — never authored
bare, since an unbound prim resolves to the protected fallback material slot),
`add_light()` (define one of the five supported `UsdLux` schemas with
editor-friendly defaults, optionally overridden by `intensity`/`color` args so
a caller isn't limited to a post-creation edit that a save wouldn't capture),
`remove_node()` (`SetActive(False)`), `set_transform()` (author
`xformOp:transform`), `save_edits()`, and `list_nodes()` — authors inside a
scoped `Usd.EditContext`. `add_model`'s and `add_light`'s failure/veto rollback
removes not just the authored prim but every parent `Xform` the call itself
created, so a rolled-back add under a not-yet-existing parent path leaves the
edit layer exactly as it was. Add/remove/light/primitive creation trigger a
geometry resync (`_resync_geometry_from_stage`: re-read via
`load_scene_from_stage`, mesh cache keeps unchanged prims free, runtime
`enabled` flags carried by prim path). `set_transform` uses a transform-only
fast path for geometry (`_reupload_instance_transforms`, no re-bake) and a full
light resync for authored light prims so analytic positions/directions refresh.
`MeshInstance.prim_path` + the `_prim_to_instances` index key all edits by USD
prim path; edits reset progressive accumulation via `_material_version`.
Headless callers pass `stage=` to `set_usd_scene`.

The geometry resync also re-reads lights + camera (so deleting a light/camera
prim drops it; `LightDir`/`LightSphere` carry `prim_path` to preserve runtime
`enabled` toggles across the re-read) and rebuilds the derived scene graph
(`build_scene_graph` + default-light injection) while bumping
`_scene_graph_version`, so the scene-graph panels repaint. Both front-ends drive
this from their scene-graph view — the Qt dock (`ui/qt/windows/scene_graph.py`)
and Panel card (`ui/panel/windows.py`) expose Add model / Add light / Delete
node / Save edits and route per-node TRS edits through `set_transform`. The Add
light menu offers DistantLight, SphereLight, DomeLight, RectLight, and DiskLight;
the first authored light naturally switches the existing all-or-nothing light
authority and removes the fallback pair. The decision logic (supported types,
add-parent resolution, deletability, TRS→matrix) lives in pure helpers
(`ui/scene_edit_actions.py`) shared by both and unit-tested without a display.

### USD Animation Playback (`playback.py`, `scene_intake.py`, `renderer.py`)

At load, `build_animation_index(stage)` scans for time-varying prims — transform
tracks (incl. ancestor-driven), animated lights, an animated camera — and
skinned meshes. `build_playback_clock(stage, index)` reads the stage's
`startTimeCode`/`endTimeCode`/`timeCodesPerSecond` into a `PlaybackClock` (pure
time logic: advance, loop, normalized scrub). The renderer keeps the stage alive
(`_usd_stage`) so prims can be re-evaluated at runtime.

Each frame, `Renderer.update(dt)` advances the clock and `_apply_animation_frame`
asks intake for one `scene_intake.read_at_time(stage, tc, …)` `TimeSample`
covering only the indexed prims, then writes it through `_apply_time_sample`:
animated transforms re-upload only those TLAS `instance_buffer` records (no
mesh rebake / BVH rebuild); animated lights replace the scene's light sets; an
animated USD camera feeds a follower used in `camera_mode == "usd"`. The
animation index selects which of the three the sample carries, so per-frame
cost still scales with the animated set — and `TimeSample.read_lights` says
whether lights were read *at all*, which is not the same as their coming back
empty (a transform-only re-read must not clear the scene's lights). The same
call serves `_refresh_usd_live_state` at the default time code after a raw USD
attribute edit, and `_resync_instance_transforms` for a subtree.

`_apply_time_sample` matches instances by the attribute the caller keyed
`xform_paths` from (`name` for playback, `prim_path` for a subtree resync);
the two are not interchangeable, because a synthetic area-light instance
carries the light prim's *leaf* name in `name` and its full path in
`prim_path`. `current_time_code` feeds the `usd_time_code` accumulation-state
provider (`params.py:ACCUM_STATE_PROVIDERS` → `_current_state_hash`), so
playback resets accumulation (1 spp in motion, converges when paused). A built-in
transport (play/pause, normalized scrubber, fps) lives in the shared spec tree,
shown only when the stage has animation.

### UsdSkel Skeletal Skinning (`scene_intake.py`, `usd_loader.py`, `vk_skinning.py`, `shaders/skin.slang`, `shaders/bvh_refit.slang`)

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

### USD-Driven Scene Controls (`scene_intake.py`, `usd_controls.py`, `ui/build_app_ui.py`)

`extract_ui_controls(stage)` parses any prim with an authored `skinny:ui:type`
into a `ControlSpec` (type, prefix-typed `target`, label, range, choices,
default, order). Resolution and application are **two owners**, split by change
`scene-intake-interface`: `scene_intake.resolve_control_binding(spec, scene=,
stage=)` looks the target up against a scene and a stage and returns a
`ControlBinding` *description* — which material index, which live
`Usd.Attribute` — and `usd_controls.control_accessors(renderer, spec)` turns
that into the get/set closures the UI uses. `renderer:`/`mtlx:` →
`_get_nested`/`params.set_param_value`; `material:<name>:<input>` →
`apply_material_override`; `usd:<prim>.<attr>` → attribute `Get`/`Set` + a
live-state refresh (lights/transforms/camera). Unresolvable targets return an
inert binding plus a warning, so a bad declaration leaves the widget
present-but-dead instead of breaking the panel.

`usd_controls.py` carries no GPU dependency, so the UI and its tests bind
controls without importing `skinny.renderer`; and the writes go through
`params.set_param_value`, never `_set_nested`, because the Qt and web
front-ends pass a marshalling proxy that `_set_nested` would resolve straight
through. A data-driven "Scene Controls" `DynamicSection` in `build_main_ui`
renders one widget per control across all front-ends, shown only when the stage
declares controls. Authored `skinny:ui:default` values apply at load.

The shared UI tree (`build_app_ui.py`) has no `IBL` or `Direct Light`
sections — those fallback-light params (`env_index`, `env_intensity`,
`direct_light_index`, `light_*`) are excluded from the Qt/Panel sidebar
entirely (`_group_params` skips any `is_fallback_light_param`); the light
color/direction dedicated widgets were removed along with them. The GLFW
debug host is unaffected — it still filters the same fallback parameters
from its own keyboard/HUD list via `build_visible_params`, conditional on
`Renderer.uses_default_lights`.

### Scene Graph Inspector (`scene_graph.py`, `ui/qt/windows/scene_graph.py`)

Preserves the USD prim hierarchy as a browsable tree with typed,
editable properties on each node. `SceneGraphNode` carries a
`RendererRef` (kind + index) mapping back to the flat renderer arrays
(material, light, instance). Property edits flow through
`apply_material_override` / `apply_light_override` /
`set_transform`. Authored light transforms are editable for all five supported
schemas; analytic-light transforms trigger a USD re-read while geometry keeps
the fast TLAS-only path. Qt presents tree-above-properties inside a
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
