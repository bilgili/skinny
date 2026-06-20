## 1. Plumb rich inputs into the flat path

- [x] 1.1 Add `transmissionColor` (float3), `specularColor` (float3),
  `diffuseRoughness` (float) to `FlatHitMat` (flat_shading.slang) and to the
  `FlatMaterialParams` struct (binding 14, common.slang / flat layout).
- [x] 1.2 Extend `pack_flat_material` (renderer.py): pack `transmission_color`
  (fallback `diffuseColor`), `specular_color` (fallback white), `diffuse_roughness`
  (fallback 0). Keep the byte layout documented; update any MSL repack
  (`pack_*_msl`) + re-verify Metal stride/offsets.
- [x] 1.3 Populate the new fields in `fetchFlatHitData` (flat_shading.slang).

## 2. Consume in the unified lobe model

- [x] 2.1 Smooth colored glass: in `FlatMaterial.sample()` delta-transmission
  branch (flat_material.slang) tint `weight` by `transmissionColor` instead of
  `albedo`. Stays delta (pdf = 0) — no MIS/pdf change.
- [x] 2.2 `specular_color`: multiply the spec-lobe response by `specularColor` in
  BOTH `FlatMaterial.sample()` and `flatBsdfResponse` (flat_lobes.slang) — pdf
  untouched, weight stays bounded.
- [x] 2.3 Oren-Nayar diffuse: replace the Lambert diffuse response with Oren-Nayar
  (driven by `diffuseRoughness`) in BOTH `sample()` and `flatBsdfResponse`, keeping
  cosine sampling so the diffuse pdf is unchanged. `diffuseRoughness = 0` ⇒ exact
  Lambert.
- [x] 2.4 Recompile SPIR-V (`slangc` main_pass) and confirm the Metal in-process
  compile; default-input renders must be byte/behaviour-unchanged.

## 3. Invariant + parity verification

- [x] 3.1 Unit: `sample().pdf == evaluate().pdf` for a tinted-spec / Oren-Nayar
  material (pdf-symmetry preserved); bounded `response/pdf`.
- [x] 3.2 GPU four-way convergence PT-BSDF / PT-(BSDF+Env) / PT-Env / BDPT on
  `three_materials`, megakernel + wavefront — values for default inputs unchanged.
- [x] 3.3 Metal ↔ Vulkan shaded parity.
- [x] 3.4 Back-compat: the pbrt parity corpus (untinted scenes) is unchanged
  (relMSE delta ≈ 0 vs pre-change).

## 4. New fidelity gates

- [x] 4.1 Add a colored-glass parity scene (a tinted `dielectric`, exported via
  `-mtlx` so `transmission_color` is carried) + pbrt v4 reference EXR; gate
  relMSE/FLIP and assert the tint reaches pixels (differs from the white-glass
  render).
- [x] 4.2 Add a `specular_color` / `diffuse_roughness` check (render differs from
  the untinted/Lambert baseline in the expected direction).

## 5. Docs

- [x] 5.1 `docs/SkinRendering.md` / the flat-BSDF doc: document the lobe model now
  consumes `transmission_color` / `specular_color` / `diffuse_roughness`.
- [x] 5.2 `README.md` Compatibility/feature note + `CHANGELOG.md` entry.
- [x] 5.3 Update the `-mtlx` docs (PbrtImport.md): colored glass / tinted spec now
  render (the export's richer slots are no longer inert for these inputs).
