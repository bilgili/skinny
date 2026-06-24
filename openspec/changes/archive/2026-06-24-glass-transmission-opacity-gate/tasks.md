## 1. Reproduce & root-cause

- [x] 1.1 Dump loaded overrides for both glass materials in
  `glass_caustics_test.usda` via `_extract_material`; confirmed the MaterialX
  `standard_surface` glass keeps `opacity = (1, 1, 1)` while `transmission = 1`
  (bridge skipped) and the UsdPreviewSurface glass gets `opacity = 0`.
- [x] 1.2 Identified the guard `if … or "opacity" in overrides` in
  `_derive_opacity_from_transmission` as the early-return that the default-opaque
  authored opacity trips.

## 2. Fix

- [x] 2.1 Added `_opacity_is_fully_opaque(value)` (scalar `≥ 1` or all channels
  `≥ 1`).
- [x] 2.2 Replaced the `"opacity" in overrides` skip in
  `_derive_opacity_from_transmission` with: skip only when an authored opacity is
  present **and** not fully opaque (a real cutout). A default-opaque opacity is
  overwritten by `1 − transmission`.

## 3. Verify

- [x] 3.1 TDD: `test_transmission_opacity_gate_ignores_default_opaque`
  (red → green): default-opaque scalar/`(1,1,1)` opacity + `transmission` → opacity
  lowered; genuine cutout (`opacity < 1`) preserved; no transmission untouched.
- [x] 3.2 Scene test `test_glass_caustics_both_spheres_transparent` (red → green):
  load `assets/glass_caustics_test.usda`, both glass materials reach `opacity < 1`.
- [x] 3.3 No regression: `tests/test_struct_layout.py` (15 passed) + the host-side
  mtlx/material/emit suites (54 passed). The 2 failures
  (`test_glass_overrides_equivalent_via_usdmtlx_plugin`,
  `test_round_trip_rich_overrides_via_load_mtlx_materials`) are **pre-existing**
  on base `2a987ec` (unresolved `.mtlx <defaultPrim>` reference / `transmission_color`
  migration — both unrelated to opacity) and fail identically with and without
  this change.
- [x] 3.4 GPU render `glass_caustics_test.usda` on **Metal + path tracer**
  (megakernel): both spheres transparent. Before/after side-by-side at a shared
  tonemap — the MaterialX sphere went from a solid opaque ball to glass.
- [x] 3.5 `ruff check src/skinny/usd_loader.py` clean.

## 4. Docs

- [x] 4.1 Noted the gate in `docs/PbrtImport.md` (transmission→opacity bridge
  block), the `flat-bsdf-lobes` spec delta, and `CHANGELOG.md`.
