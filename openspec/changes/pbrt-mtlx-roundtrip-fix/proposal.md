# pbrt-mtlx-roundtrip-fix

## Why

The `-mtlx` sidecar exporter (change `pbrt-mtlx-export`) is specified to render
**within tolerance of the UsdPreviewSurface export** — the gate guards the load
path against exporter regressions while the production integrators still consume
only the `FlatMaterial` subset. The corpus scenes that gate it (`glass_arealight`,
`conductor_infinite`, …) only exercise diffuse / conductor / dielectric, all of
which round-trip pixel-identically (`glass_arealight`: 0.000000 between the two
exports).

`subsurface` and `coateddiffuse`/`coatedconductor` are **not** in the corpus, and
they diverge badly. Rendering `sssdragon/dragon_10.pbrt` (subsurface dragon +
coateddiffuse floor) both ways on Metal wavefront against the checked-in pbrt
reference:

- UsdPreviewSurface: relMSE 0.49 / FLIP 0.13 vs pbrt (translucent, roughly tracks pbrt)
- `-mtlx`: relMSE 0.88 / FLIP 0.46 — renders **opaque white**

Three load-side root causes:

1. **`subsurface` → opacity not bridged.** `map_material` sets `opacity = 0` for
   `subsurface` (transmissive boundary). `map_material_mtlx` sets
   standard_surface `subsurface = 1`, but the fallback loader
   `_load_mtlx_materials` only bridges `transmission → opacity` and
   `emission → emissiveColor`, never `subsurface → opacity` — so the `-mtlx`
   surface stays opaque (`opacity` defaults to 1).
2. **SSS interior dropped.** Both export paths author a `skinnyOverrides`
   customData dict (homogeneous interior from `media.subsurface_overrides`) on the
   Material prim. But `_load_mtlx_materials` builds the Material from the `.mtlx`
   document alone, and `_resolve_material_binding`'s fallback branch returns the
   mtlx Material without merging the binding-target prim's `skinnyOverrides`. So
   the SSS interior is lost on the `-mtlx` path (the UsdPreviewSurface path
   recovers it via `_extract_material`).
3. **Coat key/value mismatch.** UsdPreviewSurface `coateddiffuse` authors
   `clearcoat` / `clearcoatRoughness`; `pack_flat_material` reads `coat` /
   `coat_roughness`, so the coat lobe was silently dropped on the
   UsdPreviewSurface path too. `map_material_mtlx`'s `coateddiffuse` read the
   coat roughness from a non-existent `interface.roughness` param (pbrt
   `coateddiffuse` carries its interface roughness in `roughness`), defaulting it
   to 0 — so even after the key is unified the value diverged.

## What Changes

- `_load_mtlx_materials`: add a guarded `subsurface > 0 → opacity = 0` bridge,
  alongside the existing transmission/emission bridges.
- `_resolve_material_binding` fallback branch: merge the binding-target prim's
  `skinnyOverrides` customData into the returned Material's `parameter_overrides`,
  so the `-mtlx` SSS interior is recovered (matching `_extract_material`).
- Loader: canonicalize UsdPreviewSurface `clearcoat` / `clearcoatRoughness` →
  `coat` / `coat_roughness` so both export paths land coat in the same
  `FlatMaterial` slot (fixes the silent coat drop on the UsdPreviewSurface path).
- `map_material_mtlx` `coateddiffuse`: read the coat roughness from pbrt
  `roughness` (the interface roughness), mirroring `map_material`, so the value
  matches the UsdPreviewSurface export.

**Behavior change (intended):** `coateddiffuse`/`coatedconductor` renders change
on **both** export paths — the coat lobe now actually contributes. No
subsurface/coated scene is in the corpus, so the parity gate is not perturbed;
the dragon scene is the verification.

## Impact

- Affected specs: `pbrt-mtlx-export` (round-trip equivalence requirement).
- Affected code: `src/skinny/usd_loader.py`, `src/skinny/pbrt/materials.py`.
- Affected tests: `tests/pbrt/test_mtlx_roundtrip.py` (new subsurface/coated
  round-trip regression).
- Docs: `CHANGELOG.md` + `docs/PbrtImport.md` "Known limitation" blockquote.
