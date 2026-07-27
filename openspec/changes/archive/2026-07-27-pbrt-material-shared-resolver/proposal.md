# pbrt-material-shared-resolver

## Why

`src/skinny/pbrt/materials.py` (677 lines) maps pbrt materials to two authoring
targets via twin copy-paste pipelines: `map_material` (:553 → UsdPreviewSurface)
and `map_material_mtlx` (:375 → MaterialX standard_surface), plus
`_resolve_roughness` (:193) vs `_resolve_roughness_mtlx` (:295). The low-level
accessors (`get_float_texture`/`get_spectrum_texture`, `_conductor_basecolor`,
`_subsurface_overrides`, `material_spectral_overrides`, the roughness
calibration chain) are already shared — but the *pbrt-param interpretation per
material type* (11 `mtype` branches: which params are read, with what defaults,
which lobes result) is duplicated verbatim across both mappers, with the emit
vocabulary (`diffuseColor` vs `base_color`, `opacity=0` vs `transmission=1`,
`clearcoat` vs `coat`) interleaved into the interpretation. Consequences:

- Every new pbrt material param must be wired **twice**; miss one and the two
  authoring paths silently diverge.
- Authoring-equivalence failures (the suite gate plain-USD ≡ MaterialX) land in
  two implementations that must be debugged and fixed in tandem.
- Copy drift has already happened once (the `coateddiffuse` coat-roughness
  source — since fixed and now consistent, see the comment at materials.py:480)
  and one **live** drift remains: the two mappers read *different* params for
  the `coatedconductor` base metal roughness (`roughness` vs
  `conductor.roughness`). Beyond that, several pbrt-param reads are one-sided
  (mtlx-only): `transmittance`, subsurface `reflectance`/`radius`, and
  `interface.eta` on the coated types.

## What Changes

- Extract pbrt-param interpretation into **one resolver** in
  `src/skinny/pbrt/materials.py` (or a sibling module): pbrt material →
  `ResolvedMaterial`, a target-agnostic intermediate (lobes, constant values,
  texture refs, per-target-representable extras like anisotropy, plus
  status/notes).
- `map_material` and `map_material_mtlx` become **thin emit adapters** that
  consume `ResolvedMaterial` and only translate to their target's input
  vocabulary. Genuinely target-specific behavior (aniso collapse vs
  `specular_anisotropy`, `opacity=0` gate vs `transmission=1`, emission
  weight+color vs `emissiveColor`) lives in the adapters; nothing else does.
- Collapse `_resolve_roughness` / `_resolve_roughness_mtlx` into one roughness
  resolver producing `(iso_alpha_chain_value, per_axis_values, texture)`; the
  adapters pick their representation.
- Hostless unit tests for the resolver (param → resolved form), keeping the
  existing `test_materials.py` / `test_materials_mtlx.py` output-level tests as
  the behavioral lock.
- **No behavior change.** Importer output is byte-identical; the committed
  `.usda` suite fixtures need no regeneration, and the authoring-equivalence
  gate stays green as the behavioral proof. Known copy drifts (e.g.
  `coatedconductor` base roughness param) are *preserved as-is* and recorded as
  follow-ups — fixing them here would change output.

## Capabilities

### New Capabilities

- `pbrt-material-resolution` — the shared pbrt-param → resolved-intermediate
  resolver both authoring adapters consume; single wiring point for new pbrt
  material params; hostless-testable.

### Modified Capabilities

None. `pbrt-mtlx-export` requirements ("materials map losslessly onto
standard_surface inputs") and all import-side requirements are unchanged in
behavior; this is an internal refactor whose gate is byte-identical output.

## Impact

- **Code:** `src/skinny/pbrt/materials.py` (restructure), possibly a new
  `src/skinny/pbrt/material_resolve.py`. Callers unchanged: only
  `api.py:_author_material` / `_author_material_mtlx` call the mappers, and
  `mtlx_emit.py` consumes the mtlx mapper's return shape — both signatures and
  return shapes are preserved.
- **Tests:** new hostless resolver tests; existing `tests/pbrt/test_materials.py`,
  `test_materials_mtlx.py`, `test_mtlx_emit.py`, `test_named_spectra.py`,
  `test_subsurface_coeffs.py` must pass unmodified. GPU: the confirming-suite
  authoring-equivalence gate (`tests/pbrt/test_suite.py`) must stay green.
- **Fixtures:** committed `.usda` under `tests/` are importer output — target
  is byte-identical import so no regen; any diff fails the change.
- **Risk:** subtle recorded past bugs concentrated in this file (subsurface
  radius vec3≠color3, transmission→opacity bridge, skinnyOverrides SSS merge,
  achromatic constant spectra, named conductor/glass spectra) — see design.md
  Risks.
