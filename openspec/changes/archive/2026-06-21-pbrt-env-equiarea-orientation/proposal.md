## Why

The equal-area→equirect reprojection lands the env at the wrong world
orientation. On `sss_dragon_small.pbrt` the ground plane (and the whole
background) renders a uniform dim neutral grey, while pbrt shows the road HDRI's
blue sky — the camera's view cone samples a neutral band of the map instead of
the sky.

Root cause: the reprojection's axis map `equiarea._apply_axis` used `Rx(+90)`
`(x,y,z)→(x,-z,y)`, rotating the env's sky onto skinny `+y`. That *looks* upright
when the env is viewed in isolation under a `+y`-up assumption, but the geometry
importer does **not** rotate the up-axis — it applies the change-of-basis
`transform.B = diag(1, 1, -1, 1)` (a handedness flip on Z, Y stays Y). So the env
was rotated 90° about X relative to the world the camera and meshes live in. The
prior orientation gate only eyeballed "sky at top" on the env in isolation and
never tested consistency with the imported geometry frame.

Verified at the sssdragon camera: with the Z-flip the reprojected env sampled at
the camera-forward direction reproduces pbrt's value exactly, the rendered sky
hue matches pbrt (blue/green ratio 1.21 vs 1.21), the ground turns blue
(1.06→1.23 vs pbrt 1.22), and the scene brightens ~2× (full render mean
0.039→0.081) as the camera now sees the bright sky.

## What Changes

- **`equiarea._apply_axis` / `_apply_axis_inv` use the geometry basis `B`.** The
  skinny↔pbrt env direction map becomes `(x, y, z) → (x, y, -z)` = `B[:3,:3]`,
  matching `pbrt.transform.B` so `skinny_env(d) == pbrt_env(B·d)` for every
  direction. `B` is an involution, so the forward and inverse maps coincide.
- **Regression test** pinning `_apply_axis == B[:3,:3]` (and its involution), so a
  future edit cannot silently drift back to a rotation that desyncs env and
  geometry.

## Non-Goals

- The residual subsurface-dragon dimness (the walk's ~50% energy loss at high
  optical depth) is unaffected by this change and remains a separate follow-up.
