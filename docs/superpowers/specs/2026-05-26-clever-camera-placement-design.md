# Clever camera placement on model load

**Date:** 2026-05-26
**Status:** Approved design

## Problem

When a model loads, the camera should land in a position where the whole model
is visible and the model appears upright — its up axis aligned with the scene's
up (world +Y). Today:

- `_frame_camera_to_scene` / `_frame_camera_to_mesh` already fit the bounding
  sphere into the vertical FOV, but hardcode `yaw=0, pitch=0` (a flat,
  straight-on view).
- `_look_at` hardcodes `world_up = [0, 1, 0]`, and USD geometry is never rotated
  to match. `GetStageUpAxis` is read only for scene-graph display
  (`scene_graph.py:237`). So a Z-up USD stage loads lying on its side.

## Goals

1. Loaded model appears upright: model up = scene up (+Y).
2. Camera frames the whole model.
3. Camera views from a 3/4 hero angle (yaw 30°, pitch 15° elevation).

## Non-goals

- Detecting a true "front" of a model (no metadata exists for this).
- Inferring up-axis for OBJ — OBJ is assumed Y-up (decision below). Only USD
  carries `upAxis`.
- A scene-level root transform applied at render time (rejected — too many
  call sites read `instance.transform` directly).

## Decisions

- **Achieve uprightness by rotating geometry**, not by re-orienting the camera.
  Keeps scene up = +Y everywhere; lights/IBL/floor stay consistent; no
  per-camera up bookkeeping.
- **OBJ assumed Y-up** (no rotation). Correct for existing `heads/` assets and
  most exporters. Rotation logic only fires for Z-up USD.
- **Bake the up-axis correction at USD load** (Approach A), not at render time.
- **Hero angle:** yaw +30°, pitch +15° (camera turned right, raised, looking
  slightly down).
- **F key (`reset_camera`) re-applies the framing** for the loaded model/scene,
  falling back to the current fixed default when nothing is loaded.

## Architecture

### 1. Up-axis correction (USD load)

Injection point: `usd_loader._read_open_stage` — the single function that holds
the open `stage` (so `GetStageUpAxis`), all prim world transforms, lights, and
the camera override, and assembles the `Scene`.

```
up = UsdGeom.GetStageUpAxis(stage)        # "Y" or "Z"
R  = identity            if up == "Y"
     rotate −90° about X if up == "Z"     # Z→Y: +Z→+Y, +Y→−Z
```

Stored transforms in this codebase are math-transpose
(`p_world_row = p_local_row @ M_stored`; see `MeshInstance.world_bounds`
`scene.py:163`). A world-space rotation about the origin composes as:

```
M_new = M_stored @ Rᵀ
```

Apply `R` (about the world origin) to every stage-authored entity before the
`Scene` returns, via one helper `_apply_up_axis_correction(prim_data, scene, R)`:

- **prim transforms** in `prim_data` → flow into baked `MeshInstance.transform`
- **emissive instance** transforms (`scene.instances` already populated with
  emissive instances at this point)
- **lights**: `LightSphere.position` → `R · position`;
  `LightDir.direction` → `R · direction`
- **camera override**: `position` → `R · position`; `forward` → `R · forward`

Y-up stages get `R = identity` → no-op. OBJ never enters this path.

Note: mesh instances from `prim_data` are baked *after* `_read_open_stage`
returns (in `load_scene_from_usd` / `load_scene_from_stage`). Correcting the
transform inside `prim_data` (which the helper does) means the baked instances
inherit the corrected transform — no second pass needed after baking.

### 2. Camera framing — 3/4 hero angle

`_frame_camera_to_scene` (`renderer.py:1222`) and `_frame_camera_to_mesh`
(`renderer.py:1312`) keep their existing bounding-sphere fit:

```
distance = radius / tan(fov_v/2) · margin   # margin = 1.4, clamped [0.5, 50]
target   = bbox center
```

Change only the orientation (currently `yaw=0, pitch=0`):

```
yaw   = radians(30)
pitch = radians(15)
```

With geometry now Y-up, the elevated 3/4 view is upright and reveals depth.
Factor the shared yaw/pitch/target/distance assignment so both functions stay
in sync (small private helper or shared constant).

### 3. Reset (F key)

`reset_camera` (`renderer.py:1111`) currently builds a fresh `OrbitCamera()`
(yaw=0/pitch=0, head-centroid target). Change it to re-run the framing path for
the currently loaded model/scene:

- USD scene loaded → `_frame_camera_to_scene(self._usd_scene)`
  (which still honors an authored camera override first).
- OBJ mesh source loaded → `_frame_camera_to_mesh(active source)`.
- Nothing loaded → current default `OrbitCamera()`.

Preserve the existing post-reset behavior: re-apply the scene's camera override
and `_refresh_camera_node()`.

## Edge cases

- Degenerate bbox (`radius < 1e-6`): keep existing early-return; no orientation
  change.
- Authored USD camera (`camera_override`): still wins via `_apply_camera_override`;
  hero angle applies only to the auto-frame fallback. The override's
  position/forward are `R`-corrected so an authored Z-up camera lands upright.
- Scale / `mm_per_unit`: unaffected — the correction is a rigid rotation.

## Testing

Unit (`usd_loader` correction math):
- `R` maps `(0,0,1) → (0,1,0)` and `(0,1,0) → (0,0,-1)`.
- Y-up stage → `R == identity` (correction is a no-op).
- A synthetic Z-up instance transform, after `M @ Rᵀ`, has its top vertex on +Y.

Headless (extend `tests/test_headless.py`):
- Load a Z-up USD; assert `scene.world_bounds()` Y-extent > Z-extent post-load
  (model now stands up).
- Assert `orbit_camera.yaw == radians(30)` and `pitch == radians(15)` after
  auto-frame.

## Files touched

- `src/skinny/usd_loader.py` — `_read_open_stage` + new
  `_apply_up_axis_correction` helper.
- `src/skinny/renderer.py` — `_frame_camera_to_scene`, `_frame_camera_to_mesh`,
  `reset_camera`.
- `tests/` — new unit test for correction math; headless assertions.
