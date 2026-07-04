## 1. Reproduce & lock the failing plugin-path round-trip

- [x] 1.1 Add a `test_mtlx_roundtrip.py` test that exports a `Material "subsurface"`
  scene with `-mtlx`, loads it with `use_usd_mtlx_plugin=True`, and asserts the
  composed material carries `opacity == 0` and `subsurface_sigma_s` /
  `subsurface_eta` equal to the fallback-path values. Confirmed it FAILS on the
  plugin-present interpreter before the fix.

## 2. Author only valid standard_surface inputs in the sidecar

- [x] 2.1 In `_author_material_mtlx` (`api.py`), exclude `_OVERRIDE_ONLY_INPUTS`
  from the `inputs` dict stored into `mtlx_materials` (the `write_mtlx_document`
  package). The medium coefficients stay on the prim's `skinnyOverrides`.
- [x] 2.2 Author `subsurface_radius` as `color3` (empty `_VECTOR3_INPUTS` in
  `mtlx_emit.py`); `vector3` mismatched the node definition and made the plugin
  drop the surface output.

## 3. Make the reference compose and keep the default path stable

- [x] 3.1 `author_mtlx_reference` references
  `/MaterialX/Materials/<surfacematerial_name>` explicitly (the usdMtlx layer
  authors no `defaultPrim`).
- [x] 3.2 `_resolve_material_binding` resolves against the preloaded `.mtlx`
  table first when populated (`_resolve_binding_from_mtlx_table`), so
  `use_usd_mtlx_plugin=False` stays on the fallback even on a plugin-present host.
- [x] 3.3 `_merge_prim_overrides` re-derives `_derive_opacity_from_subsurface`
  after merging the `skinnyOverrides` medium (the medium is absent when
  `_load_mtlx_materials` first derives).

## 4. Verify

- [x] 4.1 New plugin-path subsurface test passes; the fallback-path subsurface
  test (`test_subsurface_roundtrip_equivalent`) still passes.
- [x] 4.2 `pytest tests/pbrt/test_mtlx_roundtrip.py tests/pbrt/test_mtlx_emit.py`
  green except the pre-existing unrelated
  `test_round_trip_rich_overrides_via_load_mtlx_materials` (verified it fails on
  base code too).
- [x] 4.3 `ruff check` clean on changed src/test files; no new `test_parity`
  failures (10 pre-existing asset/env failures unchanged).
- [x] 4.4 `openspec validate mtlx-subsurface-plugin-roundtrip --strict`.
- [x] 4.5 codex pre-merge review (SHIP WITH FIXES). Finding #1 fixed: the table
  preemption could shadow a plain-USD material sharing a leaf name with a sidecar
  entry in a mixed stage. Scoped the preemption with `_prim_has_mtlx_reference`
  (only preempt for targets carrying a `.mtlx` reference) + regression
  `test_mtlx_table_does_not_shadow_plain_material_with_same_leaf`. Findings #2–#6
  confirmed safe, no change needed.
