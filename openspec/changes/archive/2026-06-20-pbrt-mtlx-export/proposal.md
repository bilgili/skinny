## Why

The pbrt importer maps every material to **UsdPreviewSurface**
(`materials.map_material` → `api._author_material`). UsdPreviewSurface holds only
`{diffuseColor, metallic, roughness, opacity, ior, clearcoat,
clearcoatRoughness}`, so 7 of ~9 pbrt material types are flagged APPROX: pbrt's
`transmission_color`, dielectric `eta`, separate coat IOR, subsurface radius,
anisotropic `uroughness`/`vroughness`, and `thin_walled` are squashed or dropped
at export time. The exported `.usda` is therefore **lossy** and is not a
MaterialX-native asset other MaterialX-aware engines can consume.

Two facts make a MaterialX export cheap on the consumption side:

- The renderer **already** packs a full `StdSurfaceParams` record (binding 19,
  `renderer.pack_std_surface_params`) for every material and already loads
  MaterialX back three ways (`usd_loader._extract_material`'s
  `outputs:mtlx:surface` path, the `.mtlx` reference fallback
  `_load_mtlx_materials`, and the `skinnyMaterialX` customData hint). The rich
  standard_surface slots exist on the GPU and in every pipeline's descriptor
  set — they arrive **empty** today only because the exporter never fills them.
- `materialx_runtime.add_standard_surface_material` + `MaterialLibrary` already
  author and validate standard_surface MaterialX documents in-process.

So a `.mtlx` exporter is mostly **assembling existing parts**: a pbrt →
standard_surface input mapper (lossless where standard_surface can express it)
plus a `.mtlx` writer and a stage reference the loader already resolves.

This change delivers the **interop** half (a portable MaterialX asset) and feeds
the richer parameters into `StdSurfaceParams` so they are **ready** the day the
production lobe model grows to read them. It does **not** change rendered pixels
in skinny yet — see Non-Goals.

## What Changes

- The importer SHALL accept a `-mtlx` / `--materialx` flag
  (`skinny-import-pbrt`). When set, in addition to the `.usda` it SHALL write a
  sidecar MaterialX document (`<output>.mtlx`) whose `surfacematerial` /
  `standard_surface` nodes carry the scene's materials, and SHALL reference that
  `.mtlx` from the exported stage so both skinny's loader
  (`_load_mtlx_materials` / `_extract_material`) and external MaterialX engines
  consume it.
- A new pbrt → **standard_surface** input mapper (`map_material_mtlx`, sibling
  of `map_material`) SHALL map each pbrt material type to standard_surface
  inputs, populating slots UsdPreviewSurface cannot express:
  `transmission`/`transmission_color`, `specular_IOR` from dielectric `eta`,
  `coat`/`coat_color`/`coat_IOR`, `subsurface`/`subsurface_color`/
  `subsurface_radius`, `specular_anisotropy` from `uroughness`/`vroughness`, and
  `thin_walled` for `thindielectric`. The roughness calibration chain
  (`alpha = sqrt(roughness)` remap, GGX `alpha = roughness²`) SHALL match
  `materials.py` exactly.
- Texture-bound inputs (`reflectance`, `roughness`, …) SHALL be authored as
  MaterialX `<image>` nodes in the sidecar document, resolvable by the loader's
  `_find_image_file_in_nodegraph`.
- The emitted `.mtlx` SHALL pass MaterialX document validation
  (`doc.validate()`), and its `surfacematerial` element names SHALL match the
  bound USD material leaf names so `_load_mtlx_materials` matches by name.
- Without `-mtlx`, the importer behavior SHALL be byte-unchanged
  (UsdPreviewSurface as today). The parity harness (`pbrt.parity`) SHALL cover
  the `-mtlx` export as a separate scene set so it is validated against the same
  pbrt v4 reference EXRs as the UsdPreviewSurface export (expected: equal within
  tolerance today, since the production integrators consume the FlatMaterial
  subset of both — see Non-Goals).

## Non-Goals (and why)

- **No rendered-fidelity change in skinny in this change.** The production
  integrators (`path.slang`, `bdpt.slang`, `wavefront_sppm.slang`, megakernel
  `main_pass.slang`) shade flat materials through the unified `FlatMaterial`
  lobe model (`materials/flat/flat_lobes.slang`), **not** `evalStdSurfaceBSDF`
  (which `flat-bsdf-lobes` confines to the raster preview pass by design). So the
  extra standard_surface inputs this exporter writes are not yet read by the
  estimator — both UsdPreviewSurface and a full standard_surface collapse to the
  same ~10 `FlatMaterial` fields at a hit. Reading them requires **growing the
  unified lobe set**, which is a separate, parity-gated rendering effort tracked
  as Changes 2–5 in `design.md`. This exporter is the prerequisite that feeds
  those lobes real pbrt values.
- **No spectral conductor / dispersion fidelity.** standard_surface uses
  artistic complex-IOR; pbrt's spectral `eta`/`k` and `transmission_dispersion`
  cannot be represented in the RGB pipeline and stay APPROX either way.
