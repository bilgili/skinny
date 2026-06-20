## 1. Promoting texture accessor (mirror pbrt `GetFloatTexture`)

- [x] 1.1 Add `get_float_texture(params, name, default, textures, base_dir)` and
  `get_spectrum_texture(...)` returning a tagged `Const(value)` | `Tex(path,
  color_space)`. Promote like pbrt: absent → `Const(default)`; `float`/`spectrum`/
  `rgb` → `Const(value)`; `texture` → resolve named tex (existing imagemap +
  `scale`-unwrap) → `Tex(...)`. Unresolved/unsupported class → `Const(default)` +
  APPROX note (skinny's best-effort deviation from pbrt's `ErrorExit`).
- [x] 1.2 Replace `_resolve_roughness` and the scalar `eta` / `interface.roughness`
  reads with the accessor; `map_material` holds tagged values, not raw floats — no
  `float(values[0])` path left to crash on a texture name.
- [x] 1.3 Collapse `map_material`'s two-pass (scalar-then-texture) flow into one
  uniform pass over textureable params, consuming the accessor's tag.

## 2. Generalize texture → parameter mapping (single source of truth)

- [x] 2.1 Make `_TEXTURABLE` carry `pbrt_param → (usd_input, value_type)` where
  `value_type ∈ {color3f, float}`; remove the dead `_scalar` flag duplication and
  the separate hardcoded `_SCALAR_TEX_INPUTS` set in `api.py`.
- [x] 2.2 Drive `_author_texture` from the map: pick the `UsdUVTexture` output
  channel (`.rgb`/`.r`) and the `UsdPreviewSurface` input type from `value_type`,
  generic over `usd_input` — no name-based special-casing, never assume diffuse.
- [x] 2.3 Map texture-valued `roughness`/`uroughness`/`vroughness` to the USD
  `roughness` input (collapse anisotropic to the single connection with the
  existing isotropic-collapse APPROX note).
- [x] 2.4 Authoring honors the accessor tag: `Const` → constant input, `Tex` →
  connection; unresolved/unsupported (already `Const(default)` + APPROX from 1.1)
  authors no connection. No raise on any input.

## 3. Tests

- [x] 3.1 Unit: conductor with `"texture roughness"` → no raise; `tex_inputs`
  carries the `roughness` connection when the texture resolves to an imagemap.
- [x] 3.2 Unit: nested `scale`→`imagemap` roughness resolves to the inner image
  path; APPROX note recorded for the dropped scale factor.
- [x] 3.3 Unit: unsupported FloatTexture class (e.g. `mix`/`constant`) →
  scalar default + APPROX note, no raise.
- [x] 3.4 Unit: constant `"float roughness"` unchanged (regression guard).
- [x] 3.5 Unit: texture-bound `eta` / `interface.roughness` → scalar default, no
  raise.
- [x] 3.6 Smoke: `skinny-import-pbrt` on `crown.pbrt` completes without raising
  and emits the expected shape count. **Verified end-to-end manually**:
  `import_pbrt('crown.pbrt')` → 793 meshes / 793 materials, 0 raise, 2 roughness
  texture connections authored. Not committed as a test (external scene under
  `../pbrt-v4-scenes/`); the self-contained `test_uv.py` cases (3.7) cover the
  same path in CI.
- [x] 3.7 USD-structure assert: a roughness-textured UV-less shape gets
  `primvars:st` (via `references_texture`); the material's **`roughness`** input
  (not `diffuseColor`) is connected to a `UsdUVTexture.r`, and a reflectance-textured
  material connects `diffuseColor` to `.rgb` — proving per-parameter mapping.
- [x] 3.8 Verify GPU *consumption*. **Settled by deterministic code trace** (a blind
  Metal render was unsafe: 5.8 GB free < the 10 GB megakernel-compile floor in
  `guarded_metal.sh`). The roughness-texture connection is sampled end-to-end:
  loader `_build_material` captures any connected input → `texture_bindings["roughness"]`
  (usd_loader.py:443-449); renderer `_upload_flat_materials` slots it via
  `texture_paths["roughness"]` → `pack_flat_material(roughness_texture_idx=…)`
  (renderer.py:5483-5528, `.r` channel); flat shader samples `p.roughnessTextureIdx`
  (flat_shading.slang:163-165). Integrator-agnostic (shared flat shading, mega+wave).
  No GPU follow-up needed. Empirical A/B render deferrable to a ≥10 GB-free window for
  visual confirmation only.

## 4. Docs & validation

- [x] 4.1 Update `docs/PythonAPI.md` / pbrt importer docs if the texturable-param
  behavior or report notes are user-visible; note crown support in `CHANGELOG.md`.
- [x] 4.2 `ruff check src/` clean; `openspec validate pbrt-float-texture-params
  --strict` passes.
