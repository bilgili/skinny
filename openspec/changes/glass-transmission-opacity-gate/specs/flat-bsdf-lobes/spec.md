## ADDED Requirements

### Requirement: Transmission opacity refraction gate ignores a default-opaque opacity

The loader bridge that lowers a material's `opacity` to `1 − transmission` (so the flat path's delta-dielectric refraction branch `flat_material.slang: if (m.opacity < 1.0)` fires for a `standard_surface`/OpenPBR `transmission` weight) SHALL NOT be blocked by a **default-opaque** authored `opacity` — a scalar `opacity ≥ 1` or a per-channel `opacity` whose every channel is `≥ 1`.

A `standard_surface` shader always authors `opacity` — its MaterialX default is
the fully-opaque `(1, 1, 1)` — so the bridge MUST treat that default as "no cutout
authored" and let `transmission` win. The bridge SHALL still be skipped when a
**genuine cutout** opacity is authored (any channel `< 1`, e.g. an OpenPBR
`geometry_opacity` alpha), preserving that cutout over transmission. The gate
SHALL remain a no-op when `transmission` is absent or `≤ 0`. This applies on both
material intake paths (the usdMtlx-plugin parse and the `.mtlx` API fallback),
every backend, execution mode and integrator, because it is a host-side
material-loading invariant.

#### Scenario: a standard_surface glass with default opacity becomes transparent

- **WHEN** a `standard_surface` material with `transmission = 1` and the default
  `opacity = (1, 1, 1)` (no authored cutout) is loaded — e.g. the
  `MtlxGlassSphere` in `assets/glass_caustics_test.usda`
- **THEN** the bridge sets `opacity = 0`, so the material reaches the flat
  refraction gate and renders as transparent glass (matching the UsdPreviewSurface
  glass sphere, which authors `opacity = 0` directly)

#### Scenario: a genuine cutout opacity is preserved over transmission

- **WHEN** a material authors a cutout `opacity < 1` (some channel below 1)
  alongside a `transmission` weight
- **THEN** the bridge leaves the authored cutout opacity untouched (the cutout
  wins over transmission), unchanged from before this gate
