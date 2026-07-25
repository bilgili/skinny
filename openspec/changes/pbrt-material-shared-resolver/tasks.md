# Tasks — pbrt-material-shared-resolver

## 1. Baseline snapshot (before any code change)

- [x] 1.1 Import the confirming-suite scenes + parity corpus `.pbrt` sources
      (plain and `-mtlx`) into a scratch dir; record file hashes of every
      produced `.usda` / `.mtlx` and the import reports (notes/statuses are
      gated output).
- [x] 1.2 Add a synthetic hostless `.pbrt` exercising all 11 material-type
      branches (incl. `coatedconductor`, `diffusetransmission` — absent from
      suite+corpus — plus textured and named-spectra variants) and include it
      in the 1.1 snapshot sweep.
- [x] 1.3 Run the hostless `tests/pbrt/` sweep (`-m "not gpu"`) green as the
      pre-change baseline.

## 2. Resolver extraction

- [x] 2.1 Define `ResolvedMaterial` (lobes as `ParamValue`, unreduced per-axis
      roughness, overrides, status, notes) in `src/skinny/pbrt/materials.py`.
- [x] 2.2 Implement `resolve_material(pbrt_material, *, emissive_rgb, textures,
      base_dir, flavor)` covering all 11 material-type branches; fold
      `_resolve_roughness` / `_resolve_roughness_mtlx` into one roughness
      resolver returning texture / iso / per-axis form.
- [x] 2.3 Flavor-gate every divergent or one-sided read per the design D2
      inventory (`coatedconductor` base roughness spelling;
      `diffusetransmission` `transmittance`; subsurface `reflectance`/`radius`;
      `interface.eta` on both coated types), each gate commented with its
      follow-up; preserve per-flavor note wording and note ORDER (accessor
      notes interleave with branch notes in read order).

## 3. Emit adapters

- [x] 3.1 Rewrite `map_material` as a thin UsdPreviewSurface adapter over
      `resolve_material` — same signature, same `(inputs, tex_inputs, status,
      notes)` return, byte-identical note strings.
- [x] 3.2 Rewrite `map_material_mtlx` as a thin standard_surface adapter —
      anisotropy as mean + `specular_anisotropy`, transmission/coat/subsurface
      slots, `emission = 1.0` weight preserved.
- [x] 3.3 Delete the superseded twin helpers and duplicated closures/postludes.

## 4. Tests

- [x] 4.1 Add hostless `tests/pbrt/test_material_resolve.py`: resolved form per
      material type, named-spectra (metals/glasses d-line), texture bindings,
      anisotropy, subsurface precedence, per-flavor note/status text and order,
      and usd-flavor non-reads of mtlx-only params (no note, no escalation).
- [x] 4.2 Add the grep gate: a hostless test asserting no `ParamSet` access
      (`params.get`/`.string`/`.rgb`/`.floats`/`.bool`, `get_float_texture`,
      `get_spectrum_texture`) occurs outside `resolve_material`.
- [x] 4.3 Existing output-level tests pass unmodified: `test_materials.py`,
      `test_materials_mtlx.py`, `test_mtlx_emit.py`, `test_mtlx_roundtrip.py`,
      `test_named_spectra.py`, `test_subsurface_coeffs.py`,
      `test_subsurface_routing.py`.

## 5. Byte-identity + equivalence gates

- [x] 5.1 Re-run the 1.1+1.2 import sweep (suite + corpus + synthetic
      all-mtypes scene); diff hashes against the snapshot —
      empty diff required (importer output byte-identical, no `.usda` fixture
      regen).
- [x] 5.2 Full hostless `tests/pbrt/` sweep green.
- [ ] 5.3 GPU: confirming-suite authoring-equivalence gate
      (`tests/pbrt/test_suite.py`, plain-USD ≡ MaterialX) green on Metal, no
      baseline/tolerance changes (headless env rules from CLAUDE.md).

## 6. Docs + wrap-up

- [x] 6.1 Update `docs/Architecture.md` (pbrt import material-mapping section)
      and any materials.py docstrings referencing the twin pipelines.
- [x] 6.2 File follow-up notes for the frozen drifts (coatedconductor base
      roughness spelling; diffusetransmission transmittance on the USD path).
- [ ] 6.3 `openspec validate pbrt-material-shared-resolver` clean; codex
      pre-merge review; archive after merge.
