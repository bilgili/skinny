## MODIFIED Requirements

### Requirement: exported MaterialX round-trips through the loader on both intake paths

A stage exported with `-mtlx` SHALL load in skinny and bind the same materials
whether or not the host has the USD `usdMtlx` file-format plugin: with the plugin
the loader reads `outputs:mtlx:surface` via `_extract_material`; without it the
`_load_mtlx_materials` fallback parses the sidecar `.mtlx`, matching
`surfacematerial` element names to bound USD material leaf names. Both paths
SHALL produce equivalent `Material.parameter_overrides`. The exporter SHALL NOT
author a UsdPreviewSurface material that shadows the MaterialX material for the
same prim (which would cause `ComputeBoundMaterial` to bypass the `.mtlx`).

For materials whose `FlatMaterial`-relevant state cannot be expressed by the
standard_surface shader inputs alone, the `-mtlx` load path SHALL recover the
same `FlatMaterial` parameters the UsdPreviewSurface export produces:

- **`subsurface`** SHALL map to `opacity = 0` (the transmissive-boundary gate the
  flat refraction path requires), and the homogeneous interior coefficients
  authored as `skinnyOverrides` customData on the Material prim SHALL be merged
  into the loaded `Material.parameter_overrides` — both on the
  `_extract_material` path and on the `_load_mtlx_materials` /
  `_resolve_material_binding` fallback path.
- **`coateddiffuse` / `coatedconductor`** coat weight and coat roughness SHALL
  reach the same `FlatMaterial` `coat` / `coat_roughness` slots regardless of
  whether the export authored UsdPreviewSurface `clearcoat` / `clearcoatRoughness`
  or standard_surface `coat` / `coat_roughness`.

#### Scenario: fallback path loads the sidecar when usdMtlx is absent

- **WHEN** the exported `out.usda` + `out.mtlx` are loaded on a host without the
  `usdMtlx` plugin
- **THEN** `_load_mtlx_materials` resolves the sidecar and each bound mesh gets a
  `Material` whose overrides carry the standard_surface inputs (transmission,
  coat, subsurface, anisotropy) authored by the exporter

#### Scenario: texture-bound inputs author resolvable image nodes

- **WHEN** a pbrt material binds an `imagemap` texture to `reflectance` (or a
  `scale`-wrapped imagemap)
- **THEN** the sidecar `.mtlx` authors a MaterialX `<image>` node whose `file`
  resolves via `_find_image_file_in_nodegraph` to the same path the
  UsdPreviewSurface `UsdUVTexture` export would use

#### Scenario: subsurface round-trips opacity and interior on the -mtlx path

- **WHEN** a `Material "subsurface"` is exported with `-mtlx` and loaded through
  the `_load_mtlx_materials` / `_resolve_material_binding` fallback
- **THEN** the bound `Material.parameter_overrides` carry `opacity = 0` and the
  homogeneous interior coefficients (`volume_sigma_a` / `volume_sigma_s` /
  `volume_g` / `ior`) from the prim's `skinnyOverrides`, matching the
  UsdPreviewSurface export's overrides

#### Scenario: coateddiffuse coat lobe is equivalent across export paths

- **WHEN** a `Material "coateddiffuse" "float roughness" r` is exported both as
  UsdPreviewSurface and as `-mtlx`, and both are loaded
- **THEN** the two loaded materials' FlatMaterial-relevant overrides match: coat
  weight is present (> 0) on both, and the coat roughness derived from pbrt
  `roughness` agrees between the two paths
