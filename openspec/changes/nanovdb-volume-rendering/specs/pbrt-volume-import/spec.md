# pbrt-volume-import

pbrt v4 heterogeneous participating media import: `MakeNamedMedium "nanovdb"` → UsdVol emission,
`Material "interface"` null material, medium transform stack.

## ADDED Requirements

### Requirement: nanovdb named media import as USD volumes

The pbrt importer SHALL convert `MakeNamedMedium` of type `nanovdb` into a `UsdVol.Volume` prim
with an `OpenVDBAsset` field referencing the `.nvdb` file (resolved relative to the `.pbrt`), the
medium's composed CTM, and its coefficients (`sigma_a`, `sigma_s` as RGB via the existing spectrum
resolution, `g`, `scale`) carried as `skinnyOverrides` medium keys (`volume_sigma_a`,
`volume_sigma_s`, `volume_g`, `volume_scale`, `pbrt_medium`). The current "heterogeneous media
unsupported" skip SHALL be removed for `nanovdb`; other heterogeneous types (`uniformgrid`,
`rgbgrid`, …) remain reported skips.

#### Scenario: disney-cloud medium emits a volume prim

- **WHEN** `disney-cloud.pbrt` is imported
- **THEN** the emitted `.usda` contains a `UsdVol.Volume` prim referencing
  `wdas_cloud_quarter.nvdb` with field name `density`, the medium transform from the enclosing
  `Translate`/`Scale` stack, and overrides σ_a=(0,0,0), σ_s=(1,1,1)·scale-handling per the
  renderer's convention, g=0.877, scale=4 — and the import report contains no
  "heterogeneous media unsupported" entry for it

#### Scenario: non-nanovdb heterogeneous media still report as skipped

- **WHEN** a scene declares `MakeNamedMedium` with type `rgbgrid`
- **THEN** the importer skips it with the existing report entry, unchanged

### Requirement: interface material imports as a null boundary

pbrt `Material "interface"` (and the implicit empty-material `MediumInterface` case) SHALL import
as a null boundary material: the bound shape is emitted as geometry whose material carries no BSDF
lobes and whose `skinnyOverrides` reference the shape's `MediumInterface` interior medium, so the
renderer routes it to the free-standing volume path. It SHALL NOT fall back to the default diffuse
or dielectric material.

#### Scenario: interface-bounded sphere binds the named medium

- **WHEN** `bunny-cloud.pbrt` is imported (sphere r=45, `MediumInterface "foo" ""`,
  `Material "interface"`)
- **THEN** the emitted sphere's material resolves with the `foo` medium's `volume_*` overrides and
  no diffuse/dielectric lobes, and `_resolve_medium` returns heterogeneous overrides instead of
  skipping

### Requirement: target cloud scenes import end-to-end

`disney-cloud.pbrt` and `bunny-cloud.pbrt` SHALL import to `.usda` with no skipped entities other
than recorded approximations (e.g. the bunny-cloud film sensor response), preserving each scene's
camera (including disney-cloud's mirrored `Scale -1 1 1` camera), lights (infinite RGB / EXR env,
distant sun), ground geometry, and medium placement.

#### Scenario: both scenes convert cleanly

- **WHEN** each target scene is imported via the pbrt importer CLI/API
- **THEN** the conversion completes, the report lists zero unsupported-entity skips (approx entries
  allowed), and the `.usda` loads in the renderer with the expected prim set (env light, distant
  light where present, ground, volume + interface boundary)
