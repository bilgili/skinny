# Design — pbrt-camera-up-axis

## Current pipeline (where the up is lost)

1. **Import** (`pbrt/api.py`): authors the full camera-to-world matrix
   `T.to_skinny(cam.camera_to_world)` onto the `UsdGeom.Camera`. The up vector is
   present here.
2. **Extract** (`usd_loader.py:_extract_camera`): reads `position = world[3,:3]`
   and `forward = (0,0,-1,0) @ world`. **Drops up.** Builds `CameraOverride`.
3. **Apply** (`renderer.py:_override_to_orbit`): `s = -forward`,
   `pitch = arcsin(s.y)`, `yaw = arctan2(s.x, s.z)` — a Y-up spherical decomposition,
   no roll term.
4. **Basis** (`renderer.py:_look_at` / `OrbitCamera.matrix` / `FreeCamera`):
   `world_up = (0,1,0)` hardcoded; `right = cross(forward, world_up)`.

The view basis is therefore `forward` (correct) + an up forced into the Y-up
plane. Authored roll about the forward axis is destroyed. For a Y-up camera the
forced up equals the real up → identical result; for a Z-up camera it is wrong by
the roll between them (~90° on sssdragon).

## Approach — carry the authored up, build the basis from it

A small, localized plumbing change; **no change to the yaw/pitch interaction
model**. The authored up rides alongside `forward` and is consumed by the same
`_look_at` that already builds every camera basis.

### 1. Extract the up (`usd_loader.py`)
In `_extract_camera`, after computing `forward`, compute
`up = normalize((0,1,0,0) @ world)[:3]` (USD camera local +Y → world), the exact
analogue of the existing forward extraction (local −Z → world). Both come from
the **same** world matrix, so they stay mutually consistent (including under the
improper/mirrored `Scale -1` case, which is handled orthogonally by the existing
`mirrored` ndc.x flip).

### 2. `CameraOverride.up` (`scene.py`)
Add `up: np.ndarray` (unit world-space up). Default `(0,1,0)` for any code path
that constructs an override without one, so existing call sites are unchanged.

### 3. `_look_at` takes an explicit up (`renderer.py`)
`_look_at(eye, forward, world_up=_DEFAULT_WORLD_UP)` where `_DEFAULT_WORLD_UP =
(0,1,0)`. Gram-Schmidt as today: `right = normalize(cross(forward, world_up))`,
`up' = cross(right, forward)`. Guard the degenerate case (up ∥ forward) by
falling back to a secondary axis, so a camera looking straight up/down a Z-up
world doesn't produce a zero `right`.

### 4. Honor the up on the authored camera (`renderer.py`)
`OrbitCamera` (and `FreeCamera`, for camera-mode parity) gain an `up` field
(default `(0,1,0)`) used in their `matrix()` call to `_look_at`.
`_override_to_orbit` sets `cam.up = ov.up`. The orbit position/yaw/pitch math is
unchanged (eye point + forward are already correct); only the up fed to
`_look_at` changes, which supplies the missing roll. When the user orbits, the
authored up persists as the world-up reference (a fixed tilt), which is the
sensible behavior for an authored viewpoint; the interaction model itself is
untouched.

### Back-compat
Every default / interactive camera and the entire Y-up parity corpus has
`up ≈ (0,1,0)`, so `_look_at` produces the identical basis and the rendered
output is byte-unchanged. The change is inert unless the authored up differs from
+Y.

## Verification

- **Math gate (cheap, definitive):** for the sssdragon `LookAt` params, the
  skinny authored-camera basis `(forward, up, right)` must equal pbrt's
  `T.look_at` basis mapped through `B` (the change-of-basis), to fp tolerance.
  Fails today (up forced to Y), passes after the fix. A Y-up control camera must
  stay identical pre/post.
- **`_extract_camera` unit:** a synthetic `UsdGeom.Camera` with a Z-up world
  matrix yields `CameraOverride.up ≈ +Z` (not `+Y`).
- **`_look_at` unit:** with `world_up=(0,0,1)`, the returned basis' up row tracks
  +Z; degenerate up∥forward does not produce NaN/zero rows.
- **Back-compat:** the pbrt parity corpus gate (Y-up scenes) is unchanged.
- **Visual confirmation:** the sssdragon render orientation matches the pbrt v4
  reference (dragon lies, not rolled 90°) — shown as an image; orientation only
  (brightness/env out of scope).

## Risks

- **up ∥ forward** (camera looking along the world-up axis): handled by the
  `_look_at` degenerate fallback.
- **Mirrored cameras:** up is extracted from the same authored matrix as forward,
  so the existing `mirrored` ndc.x handling is unaffected (sssdragon is mirrored
  *and* Z-up — both must compose).
- **Orbit-after-load tilt:** orbiting an authored Z-up camera keeps the authored
  up as the reference; acceptable and out of scope to "re-level".
