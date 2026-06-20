# Tasks — pbrt-mtlx-roundtrip-fix

## 1. Regression test (TDD — write first, watch it fail)

- [x] 1.1 Add a subsurface + coateddiffuse round-trip test to
  `tests/pbrt/test_mtlx_roundtrip.py`: export an inline scene both ways
  (UsdPreviewSurface and `-mtlx`), load both, and assert the `-mtlx`
  parameter_overrides match the UsdPreviewSurface ones on the FlatMaterial-
  relevant projection (opacity, coat, coat_roughness, …) plus the SSS interior
  (`volume_*`) survives. Pure test — no renderer/GPU import.

## 2. Loader fixes

- [x] 2.1 `_load_mtlx_materials`: guarded `subsurface > 0 → opacity = 0` bridge.
- [x] 2.2 `_resolve_material_binding` fallback: merge binding-target prim's
  `skinnyOverrides` customData into the returned Material.
- [x] 2.3 Canonicalize `clearcoat`/`clearcoatRoughness` → `coat`/`coat_roughness`
  in `_extract_material` (UsdPreviewSurface load path).

## 3. Exporter fix

- [x] 3.1 `map_material_mtlx` `coateddiffuse`: coat roughness from pbrt
  `roughness` (mirror `map_material`), not the non-existent `interface.roughness`.

## 4. Verify

- [x] 4.1 `tests/pbrt/test_mtlx_roundtrip.py` green (and full `tests/pbrt` no
  regressions — 199 passed; only pre-existing Vulkan-SDK-absent GPU tests fail).
- [x] 4.2 `ruff check src/` (touched files clean).
- [x] 4.3 GPU confirm on Metal: render `dragon_10.pbrt` both ways (wavefront,
  400×300, 128 spp) — `-mtlx`-vs-UsdPreviewSurface relMSE 0.0001 / FLIP 0.0002
  (was 0.88 / 0.46). Both exports now equal vs the pbrt ref (0.7031 each).

## 5. Docs

- [x] 5.1 Update `CHANGELOG.md` limitation note → fixed.
- [x] 5.2 Update `docs/PbrtImport.md` "Known limitation" blockquote → fixed.
- [x] 5.3 `openspec validate pbrt-mtlx-roundtrip-fix --strict`.
