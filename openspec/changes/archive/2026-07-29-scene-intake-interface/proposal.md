# Change: scene-intake-interface

## Why

Scene intake — USD stage to host `Scene` — has four public loaders in
`usd_loader.py`, and the renderer uses none of them for most of its work.
Instead `renderer.py` imports **9 private symbols across 15 function-local
import sites**:

```
renderer.py:3004  _read_usd_stage, bake_usd_prim, build_animation_index, build_playback_clock
renderer.py:3025  _up_axis_rt, extract_skeletal_bindings, extract_ui_controls
renderer.py:5129  _extract_distant_light, _extract_sphere_light
renderer.py:5175  _extract_camera, _world_transform
renderer.py:5235  compute_joint_matrices
renderer.py:5248  _smooth_normals, lbs_points
renderer.py:5322  resolve_control_binding
renderer.py:5342  _extract_camera, _world_transform
renderer.py:5649  _world_transform
renderer.py:5978  _prim_has_mtlx_reference          (also :6108)
renderer.py:8111  _light_color_radiance
renderer.py:8200  _extract_lens_system
```

Because every import is inside a method body, the module-level import graph
shows no coupling at all.

The dependency also runs **backwards**: `usd_loader.resolve_control_binding`
takes a renderer, reads `renderer._usd_scene.materials[mid]`
(`usd_loader.py:2720`, `:2732`), calls `renderer.apply_material_override`,
sets `renderer._usd_live_dirty` (`:2758`), reads `renderer._usd_stage`
(`:2743`), and imports `skinny.params._get_nested/_set_nested` to string-path
into renderer attributes. Intake and renderer are circular at runtime,
resolved only by laziness.

There are **three adoption paths** that each re-implement scene adoption with
different orderings:

- `set_usd_scene` (`renderer.py:5406`) — synchronous; its own docstring records
  that the scene graph is simply not built.
- `_poll_usd_streaming` (`:4912`) — the async path; also does
  `_apply_control_defaults`, `_inject_default_lights_into_scene_graph`,
  `_refresh_camera_node`, which the synchronous path does not.
- `_resync_geometry_from_stage` (`:5532`) — post-edit; hand-copies 8 fields
  onto the existing `Scene` rather than swapping it (because `id(_usd_scene)`
  is a UI change token), and must snapshot and re-apply runtime instance-enabled
  flags, light-enabled flags, and live material overrides keyed by
  `source_prim_path` with a fallback to `name` — a carry-over its own comment
  labels "finding #7".

Per-frame animation re-extraction (`_reextract_animated_lights:5117`,
`:5342`) re-derives lights and camera from loader privates instead of asking
intake to re-read at time *t*.

## What Changes

- Declare one scene-intake interface returning a value: read a stage (whole,
  streamed, or at a time code) and get back a `SceneUpdate` describing what
  changed — instances, materials, lights, camera, volume, controls, skeletal
  bindings, film clamp — with no reference to a renderer.
- The renderer applies a `SceneUpdate`. One application path replaces the three
  adoption paths, so the ordering (`film_max_component`, `_sync_volume_grid`,
  `_gen_scene_materials`, `_frame_camera_to_scene`, control defaults, default
  lights, camera node) is stated once.
- The runtime-state carry-over that `_resync_geometry_from_stage` performs —
  instance-enabled, light-enabled, material overrides by prim path — becomes
  an explicit part of applying an update, not a rescue step in one of three
  paths.
- Invert `resolve_control_binding`: intake returns the binding description;
  the renderer performs the override. `usd_loader` stops importing
  `skinny.params` and stops touching renderer state.
- Promote the 9 privates the renderer needs into the interface (as re-read
  at time *t*, or as part of the update), and delete the 15 lazy imports.
- Add hostless tests asserting a `SceneUpdate` built from a synthetic stage,
  including the runtime-state carry-over — none of which needs a GPU today.

## Capabilities

### New Capabilities

- `scene-intake`: one interface from USD stage to a `SceneUpdate` value; one
  application path in the renderer; no back-reference from intake to renderer;
  time-indexed re-read as a first-class call instead of per-frame reuse of
  extractor privates.

## Impact

- New: intake interface module (or a restructured public surface inside
  `src/skinny/usd_loader.py`), `SceneUpdate` value type, hostless tests.
- Modified: `src/skinny/usd_loader.py` (public surface, back-reference
  removed), `src/skinny/renderer.py` (three adoption paths collapse; 15 lazy
  imports deleted; per-frame re-extraction routed through the interface),
  `src/skinny/scene.py` if `SceneUpdate` lives beside `Scene`.
- Watch: `id(renderer._usd_scene)` is a UI change token at
  `ui/build_app_ui.py:361`, `:553`, `:566`. Whether an update swaps or mutates
  the `Scene` is therefore observable — see design.
- Unchanged: the `Scene` shape itself, the USD edit layer, MCP tools,
  parity-matrix behaviour.
- Docs: `docs/Architecture.md` intake section; `docs/PythonAPI.md` if the
  public loader surface changes.
