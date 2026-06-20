## Why

A pbrt scene imported into skinny renders with the **wrong camera orientation**
whenever the authored camera's up vector is not ≈ +Y. Concretely, the `sssdragon`
scene (camera up = `(-0.317, 0.312, 0.895)`, i.e. essentially +Z up — pbrt is a
Z-up convention) renders ~90° rolled vs the pbrt v4 reference: the dragon stands
upright in skinny where it lies down in pbrt.

Root cause (verified end-to-end): the camera pipeline has **no roll / up degree
of freedom** and silently assumes world-up = +Y.

- `_extract_camera` (`usd_loader.py`) pulls only **position** and **forward**
  from the authored `UsdGeom.Camera` — it discards the up vector.
- `CameraOverride` (`scene.py`) carries `position`, `forward`, `mirrored`, lens —
  **no up**.
- `_override_to_orbit` (`renderer.py`) decomposes `forward` into pure
  **yaw/pitch** (a Y-up spherical model — no roll term).
- `_look_at`, `OrbitCamera`, and `FreeCamera` (`renderer.py`) all hardcode
  `world_up = (0, 1, 0)`.

So an authored camera whose up ≈ +Y reconstructs correctly — which is why the
entire pbrt parity corpus (all Y-up `LookAt … up 0 1 0` cameras) passes — but a
camera with a different up (the common pbrt Z-up case) is rebuilt on the wrong
basis and rolls. The authored up vector *is* preserved into USD (the importer
writes the full camera-to-world matrix), then dropped at extraction.

## What Changes

- `_extract_camera` SHALL also read the camera's world-space **up** vector
  (local +Y transformed by the camera world matrix) and carry it on
  `CameraOverride`.
- `CameraOverride` SHALL gain an `up` field (unit world-space up).
- The authored-camera basis SHALL be built from `(position, forward, up)` — the
  renderer SHALL honor the authored up/roll instead of assuming `(0, 1, 0)`.
  `_look_at` SHALL accept an explicit `world_up`; the camera-override application
  SHALL feed the authored up (derived as a roll the orbit camera reproduces, so
  the live OrbitCamera/FreeCamera interaction model is preserved).
- Back-compat: when the authored up is already ≈ +Y (the corpus, and every
  interactive/default camera), the result SHALL be byte-identical to today — the
  pbrt parity corpus is unchanged.
- Verification: the `sssdragon` (Z-up) camera SHALL reproduce the pbrt v4 frame
  orientation (the dragon lies, matching the reference, not rolled 90°), and the
  Y-up parity corpus SHALL remain within its existing tolerances.

## Non-Goals

- **Environment-light intensity / exposure** parity (the ~5.45× brightness gap
  on sssdragon) — a separate env-intensity calibration matter.
- **Environment-map rotation** for Z-up scenes (the IBL may also need the Z-up
  basis) — separate follow-up.
- A full 6-DoF interactive roll control on the GUI cameras; this change adds the
  roll needed to honor an *authored* camera, not a new interactive control
  (though the basis plumbing makes that a later, smaller step).
- Subsurface / glass material fidelity (covered by other changes).
