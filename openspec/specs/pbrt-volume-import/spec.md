# pbrt-volume-import Specification

## Purpose
TBD - created by archiving change nanovdb-volume-rendering. Update Purpose after archive.
## Requirements
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

The importer SHALL treat both pbrt `Material "interface"` and an empty-string pbrt material on a
shape carrying a `MediumInterface` as a null boundary material: the bound shape is emitted as
geometry whose material carries no BSDF lobes and whose `skinnyOverrides` reference the shape's
`MediumInterface` interior medium (with the `volume_interface` marker), so the renderer routes it
to the free-standing volume path. Such a material MUST NOT fall back to the default diffuse or
dielectric material. An empty-string material with no `MediumInterface` SHALL keep the
default-material behavior.

#### Scenario: interface-bounded sphere binds the named medium

- **WHEN** `bunny-cloud.pbrt` is imported (sphere r=45, `MediumInterface "foo" ""`,
  `Material "interface"`)
- **THEN** the emitted sphere's material resolves with the `foo` medium's `volume_*` overrides and
  no diffuse/dielectric lobes, and `_resolve_medium` returns heterogeneous overrides instead of
  skipping

#### Scenario: empty-string material with a medium interface is a null boundary

- **WHEN** `clouds.pbrt` is imported (sphere, `MediumInterface "c" ""`, `Material ""`)
- **THEN** the emitted sphere's material carries the `c` medium overrides + `volume_interface`
  marker with no BSDF lobes (not the grey `UsdPreviewSurface` fallback)

#### Scenario: empty-string material without a medium keeps the default

- **WHEN** a shape has `Material ""` and no `MediumInterface`
- **THEN** it imports with the default material, unchanged

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

### Requirement: procedural cloud media import as volumes

The pbrt importer SHALL convert `MakeNamedMedium` of type `cloud` into a supported procedural
volume: the medium's `sigma_a`/`sigma_s` (RGB via the existing spectrum resolution), `g`,
`density`, `wispiness` (default 1), and `frequency` (default 5) SHALL be carried as `skinnyOverrides`
keys (`volume_cloud`=True, `cloud_density`, `cloud_wispiness`, `cloud_frequency`, `volume_sigma_a`,
`volume_sigma_s`, `volume_g`) on the bound interface material, and the medium's composed CTM SHALL
be preserved (the renderer maps world→medium-local so the altitude falloff reads correctly). The
current "heterogeneous media unsupported" skip SHALL be removed for `cloud`; other unsupported
heterogeneous types (`uniformgrid`, `rgbgrid`, …) remain reported skips.

#### Scenario: clouds.pbrt medium imports as a cloud volume

- **WHEN** `clouds.pbrt` is imported
- **THEN** the import report contains no "heterogeneous media unsupported" entry for the `cloud`
  medium, and the bound sphere's material carries `volume_cloud`=True with `cloud_density`=2 and
  the σ_a/σ_s/g overrides

#### Scenario: non-cloud unsupported media still report as skipped

- **WHEN** a scene declares `MakeNamedMedium` of type `rgbgrid`
- **THEN** the importer skips it with the existing report entry, unchanged

