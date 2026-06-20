## 1. pbrt → standard_surface mapper

- [x] 1.1 Add `map_material_mtlx(pbrt_material, *, emissive_rgb, textures,
  base_dir)` in `pbrt/materials.py`, sibling of `map_material`, returning
  standard_surface inputs + texture connections + status + notes. Reuse
  `pbrt_roughness_to_alpha` / `alpha_to_usd_roughness` / `_conductor_basecolor`
  for parity with the UsdPreviewSurface path.
- [x] 1.2 Per pbrt type, fill standard_surface slots UsdPreviewSurface drops:
  `dielectric`/`thindielectric` → `transmission`(+`transmission_color`),
  `specular_IOR` from `eta`, `thin_walled` for thin; `coateddiffuse`/
  `coatedconductor` → `coat`/`coat_color`/`coat_IOR`; `subsurface` →
  `subsurface`/`subsurface_color`/`subsurface_radius`; `conductor` → metalness +
  `base_color`/`specular_color`.
- [x] 1.3 Map anisotropic `uroughness`/`vroughness` → `specular_roughness` +
  `specular_anisotropy` (no isotropic collapse); record an APPROX note only when
  truly unrepresentable.

## 2. MaterialX document authoring

- [x] 2.1 Add a `.mtlx` writer (extend `materialx_runtime` or a new
  `pbrt/mtlx_emit.py`) that builds a `mx.Document` of `surfacematerial` +
  `standard_surface` nodes from the mapper output, naming each `surfacematerial`
  to match the bound USD material leaf name. Validate with `doc.validate()`
  before write.
- [x] 2.2 Author texture-bound inputs as MaterialX `<image>` nodes (file +
  colorspace) resolvable by `_find_image_file_in_nodegraph`; reuse
  `resolve_texture` (imagemap + `scale`-unwrap) from `materials.py`.
- [x] 2.3 Author the stage→`.mtlx` reference and make the MaterialX material
  authoritative: do **not** emit a shadowing UsdPreviewSurface for the same prim
  under `-mtlx` (else `ComputeBoundMaterial` bypasses the sidecar). Confirm
  `_collect_mtlx_asset_paths(stage)` returns the reference.

## 3. CLI + orchestration

- [x] 3.1 Add `-mtlx` / `--materialx` to `pbrt/cli.py`; thread a `materialx: bool`
  through `api.import_pbrt` / `translate_scene` and switch `_author_material` to
  the MaterialX path (sidecar write) when set, UsdPreviewSurface when not.
- [x] 3.2 Write `<output>.mtlx` next to the `.usda`; keep `-q/--quiet` and the
  unsupported-material exit code behavior intact.

## 4. Tests

- [x] 4.1 Unit: `map_material_mtlx` for each pbrt type fills the expected
  standard_surface inputs (transmission/coat/subsurface/anisotropy/thin_walled);
  roughness chain matches `map_material`.
- [x] 4.2 Unit: `-mtlx` export writes a `doc.validate()`-clean `.mtlx`, the
  `.usda` references it, surfacematerial names match bound leaf names, and
  no-`-mtlx` output is unchanged.
- [x] 4.3 Round-trip: load the exported `out.usda`+`out.mtlx` and assert each
  bound mesh's `Material.parameter_overrides` carry the rich inputs — exercise
  **both** the `_extract_material` (usdMtlx present) and `_load_mtlx_materials`
  (absent) intake paths; assert equivalent overrides.
- [x] 4.4 Texture round-trip: imagemap (and `scale`-wrapped) `reflectance`
  resolves to the same path via `_find_image_file_in_nodegraph` as the
  UsdUVTexture export.

## 5. Parity gate

- [x] 5.1 Add the `-mtlx` export to `pbrt/parity.py` as a scene set against the
  existing pbrt v4 reference EXRs; gate relMSE/FLIP and assert `-mtlx` vs
  UsdPreviewSurface renders agree within tolerance (Metal backend).
  Non-GPU plumbing unit-tested green; GPU render is `@pytest.mark.gpu` for the
  main thread.

## 6. Docs

- [x] 6.1 `README.md`: document the `-mtlx` flag (new CLI flag).
- [x] 6.2 `docs/PythonAPI.md`: document any new public symbol (`map_material_mtlx`
  / mtlx-emit entry) if exposed.
- [x] 6.3 Note in the importer docs that `-mtlx` is interop + Stage-2-enabling,
  not a current in-skinny fidelity change (point at `design.md` roadmap).
