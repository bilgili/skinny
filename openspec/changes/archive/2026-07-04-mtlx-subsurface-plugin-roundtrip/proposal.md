## Why

The `-mtlx` exporter authors the subsurface **medium coefficients**
(`subsurface_sigma_a`, `subsurface_sigma_s`, `subsurface_g`, `subsurface_eta`)
as `standard_surface` `<input>` elements in the sidecar `.mtlx`. These are
skinny-invented keys, **not** part of the Autodesk `standard_surface` node
definition. When the USD `usdMtlx` file-format plugin composes such a material,
it rejects the unknown inputs and drops the shader's surface output entirely, so
`ComputeBoundMaterial` returns a `Material` with no surface source and
`_extract_material` recovers **none** of the shader inputs (base_color,
subsurface weight, opacity, …). The whole subsurface material is lost on the
plugin-present composed path.

The bug is currently masked: `_resolve_material_binding` keeps the preloaded
`.mtlx` sidecar table authoritative for `use_usd_mtlx_plugin=False`, so the
default loader path still recovers subsurface via the `_load_mtlx_materials`
fallback. But `use_usd_mtlx_plugin=True` (plugin present) loses the interior, and
no test asserts subsurface round-trip equivalence via the plugin path.

The flat / UsdPreviewSurface export already does the right thing: it filters
these `_OVERRIDE_ONLY_INPUTS` out of the shader and carries the interior as
`skinnyOverrides` customData on the Material prim (read by `_extract_material`
and merged via `_merge_prim_overrides`). The `-mtlx` path authors the same
customData already — it just fails to strip the keys from the `.mtlx` shader.

## What Changes

Making a `-mtlx` subsurface material survive the `usdMtlx` plugin required
fixing three ways the exported artifacts were malformed, plus keeping the
default load path stable now that the reference composes:

- **Strip the non-standard_surface keys.** The `-mtlx` exporter SHALL author only
  real `standard_surface` inputs in the sidecar `.mtlx`. The subsurface
  medium-coefficient keys (`subsurface_sigma_a` / `_s` / `_g` / `_eta`) SHALL be
  excluded from the shader (mirroring the UsdPreviewSurface path's
  `_OVERRIDE_ONLY_INPUTS` filter). They continue to ride the Material prim's
  `skinnyOverrides` customData (already authored via `media.subsurface_overrides`).
- **Author `subsurface_radius` as `color3`.** It was authored as `vector3`, which
  mismatches the Autodesk `standard_surface` node definition (`subsurface_radius`
  is a per-channel scattering-radius `color3`); the plugin rejected the node and
  dropped the surface output. `_VECTOR3_INPUTS` becomes empty.
- **Reference the composed material prim explicitly.** `author_mtlx_reference`
  referenced the whole `.mtlx` layer, but the usdMtlx-composed layer has no
  `defaultPrim`, so the reference never resolved (the plugin-present path
  recovered nothing for ANY material). It SHALL reference
  `/MaterialX/Materials/<surfacematerial_name>` directly. Plugin-absent behavior
  is unchanged (the layer can't be read, the typeless `over` still falls back).
- **Keep the preloaded `.mtlx` table authoritative for `use_usd_mtlx_plugin=False`.**
  Now that the reference composes, `_resolve_material_binding` SHALL resolve
  against the preloaded sidecar table (when populated) before
  `ComputeBoundMaterial`, so the default load path stays on the fallback and does
  not silently switch intake by host plugin availability. The opacity gate is
  re-derived after the `skinnyOverrides` medium is merged (`_merge_prim_overrides`),
  since the medium is no longer present when `_load_mtlx_materials` first runs.
- Add a regression test asserting subsurface round-trip equivalence via the
  plugin path (`use_usd_mtlx_plugin=True`): opacity == 0 and
  `subsurface_sigma_s` / `subsurface_eta` equal to the fallback path.

## Impact

- Affected specs: `pbrt-mtlx-export`
- Affected code: `src/skinny/pbrt/api.py` (`_author_material_mtlx`),
  `src/skinny/pbrt/mtlx_emit.py` (`_VECTOR3_INPUTS`, `author_mtlx_reference`),
  `src/skinny/usd_loader.py` (`_resolve_material_binding` +
  `_resolve_binding_from_mtlx_table`, `_merge_prim_overrides`),
  `tests/pbrt/test_mtlx_roundtrip.py`, `tests/pbrt/test_mtlx_emit.py`
- No renderer/GPU change. Sidecar `.mtlx` for subsurface materials drops the four
  bogus `<input>` elements and authors `subsurface_radius` as `color3`; the
  `.usda` reference now targets the composed material prim. `skinnyOverrides`
  customData is unchanged. Non-`-mtlx` (UsdPreviewSurface) loads are unaffected
  (the preloaded table is empty for them, so the new early check is skipped).
