## 1. Failing tests first (TDD)

- [x] 1.1 Math gate: assert skinny's authored-camera basis `(forward, up, right)`
  for the sssdragon `LookAt`/`Scale -1` params equals pbrt `T.look_at` mapped
  through `B`, to fp tolerance. Plus a Y-up control camera that must stay
  identical. (Fails today — up forced to +Y.)
- [x] 1.2 `_extract_camera` unit: a synthetic `UsdGeom.Camera` with a Z-up world
  matrix yields `CameraOverride.up ≈ +Z` (not `+Y`).
- [x] 1.3 `_look_at` unit: with `world_up=(0,0,1)` the basis up tracks +Z;
  degenerate up∥forward yields a finite orthonormal basis (no NaN/zero row).

## 2. Carry the up through the pipeline

- [x] 2.1 `CameraOverride` (`scene.py`): add `up: np.ndarray` field, default
  `(0,1,0)`; document it alongside `forward`/`mirrored`.
- [x] 2.2 `_extract_camera` (`usd_loader.py`): compute
  `up = normalize((0,1,0,0) @ world)[:3]` and pass to `CameraOverride`.

## 3. Build the basis from the authored up

- [x] 3.1 `_look_at` (`renderer.py`): add `world_up` param (default `(0,1,0)`),
  Gram-Schmidt with a degenerate up∥forward fallback.
- [x] 3.2 `OrbitCamera` (+ `FreeCamera` for camera-mode parity): add an `up`
  field (default `(0,1,0)`) used in `matrix()`'s `_look_at` call.
- [x] 3.3 `_override_to_orbit` (`renderer.py`): set `cam.up = ov.up` (eye/yaw/
  pitch math unchanged — only the up fed to `_look_at` changes).

## 4. Verify

- [x] 4.1 Make 1.1–1.3 pass; confirm no NaN and the degenerate fallback.
- [x] 4.2 Back-compat: pbrt parity corpus (Y-up) unchanged (relMSE delta ≈ 0);
  default/interactive camera renders byte-identical.
- [x] 4.3 Visual: re-render sssdragon (Z-up) vs the pbrt v4 reference — dragon
  lies (matches reference orientation), not rolled 90°. Show the image
  (orientation only; brightness/env out of scope).

## 5. Docs

- [x] 5.1 Update the camera section (docs/Architecture.md and/or the camera doc):
  authored camera up/roll is now honored; `CameraOverride.up`; `_look_at`
  `world_up` param; default `(0,1,0)` keeps Y-up byte-identical.
- [x] 5.2 `CHANGELOG.md` entry; note pbrt Z-up cameras now orient correctly
  (env-intensity / env-map rotation remain follow-ups).
