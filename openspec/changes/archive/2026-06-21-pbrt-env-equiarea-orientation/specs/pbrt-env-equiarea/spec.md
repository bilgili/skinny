## MODIFIED Requirements

### Requirement: Equal-area octahedral env maps are reprojected to equirectangular

The pbrt importer SHALL reproject an `infinite` light's image map from pbrt v4's
equal-area octahedral square parameterization to the equirectangular layout that
skinny's dome-light path and `environment.slang` consume, so that the radiance
the shader samples for a direction matches the radiance pbrt's
`ImageInfiniteLight` returns for that same direction. The reprojection SHALL port
pbrt's `EqualAreaSquareToSphere` / `EqualAreaSphereToSquare` and invert skinny's
`directionToEquirectUV` (`u = atan2(dx,dz)/2π + 0.5`, `v = acos(dy)/π`, `+y` up).

The reprojection's skinny↔pbrt direction map SHALL be the **same change-of-basis
the geometry import uses** — `pbrt.transform.B = diag(1, 1, -1, 1)`, i.e.
`(x, y, z) → (x, y, -z)` — so the baked env is oriented consistently with the
world the camera and meshes occupy (`skinny_env(d) == pbrt_env(B·d)` for every
direction). It SHALL NOT apply an up-axis rotation that the geometry import does
not, which would rotate the env relative to the scene.

#### Scenario: square image map is reprojected

- **WHEN** a `LightSource "infinite"` references a **square** `.exr`/`.pfm` image
  map (equal-area octahedral, as pbrt v4 always authors)
- **THEN** the emitted `.hdr` is the equirectangular reprojection of that map

#### Scenario: env orientation is consistent with the imported geometry

- **WHEN** the reprojection maps a skinny world direction `d` to the pbrt
  light-space direction it samples
- **THEN** it applies `B = diag(1, 1, -1)` (the geometry import's basis), so the
  same direction the camera and meshes use indexes the env pbrt would return —
  the sssdragon sky/ground reflect the HDRI's blue sky, not a neutral band

#### Scenario: orientation map is an involution

- **WHEN** the forward (`_apply_axis`) and inverse (`_apply_axis_inv`) direction
  maps are applied
- **THEN** they coincide and round-trip to identity (`B` is its own inverse)
