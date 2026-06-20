## ADDED Requirements

### Requirement: `-mtlx` flag exports a sidecar MaterialX document

The pbrt importer (`skinny-import-pbrt`) SHALL accept a `-mtlx` / `--materialx`
flag. When the flag is set, the importer SHALL, in addition to the `.usda`
stage, write a sidecar MaterialX document at `<output>.mtlx` containing the
scene's materials as `surfacematerial` nodes wrapping `standard_surface`
shaders, and SHALL author a reference to that `.mtlx` from the exported stage so
the document is discoverable by `usd_loader._collect_mtlx_asset_paths`. When the
flag is absent, the importer output SHALL be byte-identical to current
UsdPreviewSurface behavior.

#### Scenario: -mtlx emits a referenced sidecar document

- **WHEN** `skinny-import-pbrt scene.pbrt -o out.usda -mtlx` runs
- **THEN** `out.mtlx` exists, passes MaterialX `doc.validate()`, contains one
  `surfacematerial` per exported material, and `out.usda` references `out.mtlx`
  such that `_collect_mtlx_asset_paths(stage)` returns it

#### Scenario: no -mtlx leaves UsdPreviewSurface output unchanged

- **WHEN** the importer runs without `-mtlx`
- **THEN** no `.mtlx` is written and the `.usda` materials are authored as
  UsdPreviewSurface exactly as before (no regression)

### Requirement: pbrt materials map losslessly onto standard_surface inputs

The `-mtlx` exporter SHALL map each supported pbrt material type onto
`standard_surface` inputs, populating parameters that UsdPreviewSurface cannot
express. Dielectric `eta` SHALL map to `specular_IOR`; `dielectric`/
`thindielectric` SHALL set `transmission` (and `thin_walled` for the thin case);
a tinted transmission SHALL map to `transmission_color`; `coateddiffuse`/
`coatedconductor` SHALL set `coat`/`coat_color`/`coat_IOR` distinct from the base
lobe; `subsurface` SHALL set `subsurface`/`subsurface_color`/`subsurface_radius`;
and anisotropic `uroughness`/`vroughness` SHALL map to `specular_roughness` +
`specular_anisotropy` rather than collapsing to an isotropic geometric mean. The
roughness calibration chain SHALL match `materials.py`
(`alpha = sqrt(roughness)` under `remaproughness`, GGX `alpha = roughness²`).

#### Scenario: glass carries transmission color and IOR

- **WHEN** a `Material "dielectric" "float eta" [1.5]` (optionally tinted) is
  exported with `-mtlx`
- **THEN** the emitted standard_surface sets `transmission` > 0,
  `specular_IOR = 1.5`, and `transmission_color` to the tint (white when
  untinted) — none of which the UsdPreviewSurface export can represent

#### Scenario: anisotropic roughness is preserved, not averaged

- **WHEN** a material declares distinct `uroughness`/`vroughness`
- **THEN** the standard_surface sets `specular_roughness` and a non-zero
  `specular_anisotropy` derived from the two, instead of the single isotropic
  geometric-mean roughness the UsdPreviewSurface path emits

#### Scenario: thindielectric sets thin_walled

- **WHEN** a `Material "thindielectric"` is exported with `-mtlx`
- **THEN** the standard_surface sets `thin_walled = true` and `transmission` > 0

### Requirement: exported MaterialX round-trips through the loader on both intake paths

A stage exported with `-mtlx` SHALL load in skinny and bind the same materials
whether or not the host has the USD `usdMtlx` file-format plugin: with the plugin
the loader reads `outputs:mtlx:surface` via `_extract_material`; without it the
`_load_mtlx_materials` fallback parses the sidecar `.mtlx`, matching
`surfacematerial` element names to bound USD material leaf names. Both paths
SHALL produce equivalent `Material.parameter_overrides`. The exporter SHALL NOT
author a UsdPreviewSurface material that shadows the MaterialX material for the
same prim (which would cause `ComputeBoundMaterial` to bypass the `.mtlx`).

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

### Requirement: parity harness covers the MaterialX export

The pbrt parity harness (`pbrt.parity`) SHALL include the `-mtlx` export as a
scene set evaluated against the same pbrt v4 reference EXRs as the
UsdPreviewSurface export, gated on relMSE + FLIP. Because the production
integrators currently consume only the `FlatMaterial` subset, the `-mtlx` and
UsdPreviewSurface renders SHALL match within tolerance (this gate guards against
the exporter regressing the load path; it is not a fidelity-improvement gate —
that arrives with the Stage-2 lobe changes).

#### Scenario: -mtlx export renders within tolerance of the UsdPreviewSurface export

- **WHEN** a corpus scene is exported both ways and rendered headless on the
  Metal backend
- **THEN** both renders pass the scene's relMSE/FLIP gate against the pbrt v4
  reference, and differ from each other only within tolerance
