# Camera distance: settable to any value, 10× initial cap

**Date:** 2026-05-27
**Status:** Approved design

## Problem

The orbit-camera distance is hard-clamped to `OrbitCamera.max_distance` at every
entry point — the scene-graph field (slider + spinbox), the explicit setter, and
wheel zoom. `max_distance` is set at model load to `max(50, 4× longest AABB edge)`
(commit `66bb746`). Two consequences:

1. The distance field cannot be set beyond the cap, even when the user
   deliberately wants a far wide shot.
2. The `4×` multiplier and the stale `50.0` clamp in settings-restore are tuned
   conservatively; large scenes still feel pinned.

We want the distance field to accept **any value**, while the cap becomes the
**initial** bound (seed + starting slider range) rather than a hard ceiling, and
the multiplier moves to `10×`.

## Decisions

- **Cap scope — initial value + auto-grow.** The cap clamps the auto-framed start
  distance and seeds the slider range. Typing a larger number, or wheel-zooming
  past the current ceiling, raises the live ceiling so the slider, spinbox, and
  wheel-zoom stay mutually consistent. The ceiling never shrinks on its own; only
  a re-frame / model-load resets it to `cap`.
- **Keep the floor.** Formula becomes `cap = max(50, 10× longest edge)`. Small
  models (the head OBJs) keep a generous 50-unit initial range; auto-grow still
  lets the user exceed it.
- **Single source of truth + live rescale (Approach 1).** `OrbitCamera.max_distance`
  remains the one live ceiling. Writes grow it; the field widgets rescale in place
  to follow it — no fixed huge max, no widget rebuild.
- **All GUIs consistent.** Both the Qt scene-graph dock and the Panel web scene
  graph get free-entry + auto-grow.

## Behaviour spec

| Interaction | Before | After |
| --- | --- | --- |
| Model load auto-frame | seed + ceiling = `max(50, 4×)` | seed + ceiling = `max(50, 10×)` |
| Distance field (slider) | hard `[0.5, max_distance]` | slider range `[0.5, max_distance]`, grows when value exceeds it |
| Distance field (numeric) | hard `[0.5, max_distance]` | any value `≥ 0.5` (guarded at `1e9`); raises ceiling |
| Wheel zoom-out | clamped at ceiling | grows ceiling past it |
| Settings restore | clamped to `[0.5, 50.0]` | `max(0.5, restored)`; lifts ceiling to match |
| Lower bound | `0.5` | `0.5` (unchanged) |

## Changes

### 1. Renderer core (`src/skinny/renderer.py`) — shared by all front-ends

- `_orbit_distance_cap` (line ~620): multiplier `4.0` → `10.0`; keep the
  `max(50.0, …)` floor; update the docstring (`4×` → `10×`).
- New method `OrbitCamera.set_distance(v)`:
  - `v = clamp(float(v), 0.5, 1e9)` — `1e9` is a degeneracy guard (avoids
    NaN/inf and int-slider precision loss); effectively unbounded for real scenes.
  - if `v > self.max_distance`: `self.max_distance = v`.
  - `self.distance = v`.
- `OrbitCamera.zoom` (line ~736): route through `set_distance` so zoom-out past
  the ceiling grows it instead of clamping.
- `Renderer.apply_camera_param` distance branch (line ~4722):
  `cam.set_distance(v)` instead of `cam.distance = clip(v, 0.5, cam.max_distance)`.
- The three auto-frame paths (`_frame_camera_to_mesh` and the two
  override/USD framers) already assign `max_distance = cap` and clamp the seeded
  distance to `[0.5, cap]`; unchanged apart from inheriting the new `10×`.

### 2. Settings restore (`src/skinny/app.py`, line ~78)

Replace `o.distance = clip(restored, 0.5, 50.0)` with:

```python
o.distance = max(0.5, _flt(orbit_raw.get("distance"), o.distance))
o.max_distance = max(o.max_distance, o.distance)
```

So a persisted large distance survives instead of being clamped to 50.

> Note: model-load auto-frame may re-frame over a restored camera (it overwrites
> `distance` and `max_distance` with the freshly-computed `cap`). That is
> pre-existing behaviour and out of scope here.

### 3. Qt field widget (`src/skinny/ui/qt/windows/scene_graph.py`, `_add_float`)

Gated on `prop.metadata.get("growable")` (see §5) so only the distance property
is affected; other float properties keep their fixed metadata `[min, max]` range.

- Spinbox: `spin.setRange(lo, 1e9)` so any numeric value is typeable.
- Slider: keep mapping `[lo, max_distance]`. Make `hi` / `span` mutable
  (close over a small holder rather than fixed locals).
- Extend the existing per-tick `pull()`: when the camera's live `max_distance`
  differs from the captured `hi`, rescale the slider mapping (`hi`, `span`,
  `to_int`/`from_int`) in place so the thumb tracks the grown range. No widget
  rebuild — preserves the slider grab during drag-to-grow.

### 4. Panel field widget (`src/skinny/ui/panel/windows.py`)

- For the `growable` float property (see §5), use `pn.widgets.EditableFloatSlider`
  instead of `FloatSlider`: soft `end = max_distance`, `fixed_start = 0.5`,
  `fixed_end = 1e9`. Typing past the soft `end` is allowed up to `fixed_end`.
- In the change watcher, when the applied value exceeds `end`, set `w.end = value`
  so the slider track follows the grown ceiling.
- Other float properties keep the plain bounded `FloatSlider`.

### 5. Scene-graph model (`src/skinny/scene_graph.py`, line ~925)

Add `"growable": True` to the distance property's `metadata` (alongside the
existing `min`/`max`). This one flag gates the free-entry + auto-grow behaviour
in both field widgets (§3, §4), decoupling them from the property name. The
`metadata["max"]` still reads `getattr(camera, "max_distance", 50.0)` at build
time; stale-at-build is fine because the live widgets rescale from the camera.

## Non-issues verified

- **Far plane.** The main pass is ray-traced; primary rays originate at the
  camera with directions from the inverse view — there is no rasterizer far-clip.
  `OrbitCamera.far = 100.0` stays. Commit `66bb746` already proved large camera
  distances render correctly.
- **Lower bound.** The `0.5` floor is retained at every write site.

## Testing (`tests/test_camera_placement.py`, extends `TestDistanceCap`)

Some existing `TestDistanceCap` assertions **change meaning** and must be
rewritten, not merely renumbered:

- `_orbit_distance_cap`: `10×` scaling and the `50` floor. The `4×`-era boundary
  case `_orbit_distance_cap(12.5) == 50.0` becomes `== 125.0`; `(20.0)` becomes
  `== 200.0`; the floor case `(2.0) == 50.0` is unchanged.
- `test_frame_mesh_sets_cap`: the 40-unit mesh now yields `max_distance == 400.0`
  (was `160.0`).
- `test_zoom_respects_dynamic_cap` **inverts**: zoom-out past the ceiling now
  *grows* it, so the old `distance <= 200` assertion is wrong. Rewrite to assert
  the ceiling and distance rose past the starting `max_distance`.
- `test_scene_graph_slider_uses_max_distance`: still valid (metadata `max` reads
  `max_distance`).

New cases:

- `set_distance` raises `max_distance` when the value exceeds it; lower-clamps to
  `0.5`; clamps the upper guard at `1e9`.
- Settings-restore honours a `> 50` distance and lifts `max_distance` to match.

Renderer-importing tests stay gated behind the existing `@needs_renderer`
(Vulkan-SDK) marker (commit `8c09519`).
