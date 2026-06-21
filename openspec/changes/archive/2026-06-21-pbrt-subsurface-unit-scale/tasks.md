## 1. Unit-scale fix (import-side)

- [x] 1.1 Add `PBRT_STAGE_METERS_PER_UNIT = 1.0` constant in `emit.py` and use it
  in the `SetStageMetersPerUnit` call (single source of truth for the stage unit).
- [x] 1.2 In `media.subsurface_overrides`, divide the derived `subsurface_sigma_a`
  / `subsurface_sigma_s` by `PBRT_STAGE_METERS_PER_UNIT * 1000` (= the loader's
  resulting `mm_per_unit`) before packing onto `skinnyOverrides`. Document why
  (cancels the generic unit factor pbrt does not apply to media). Leave `g`,
  `eta`, `ior` untouched.

## 2. Tests (brightness-independent)

- [x] 2.1 Unit test: `subsurface_overrides` for `Skin1 scale 10` emits σ equal to
  the pre-fix σ / 1000 (exact), and a custom explicit-`sigma_a`/`sigma_s` material
  is divided by the same factor.
- [x] 2.2 Analytic optical-depth test: for the imported coefficients, the per-mm
  `σ_packed · mm_per_unit` round-trips to the original pbrt mm⁻¹ coefficients.
- [x] 2.3 Confirm the existing subsurface furnace / forward-compat tests still
  pass (no regression in energy or routing).

## 3. Render verification (manual A/B, not a CI gate)

- [x] 3.1 Re-render `sss_dragon_small.pbrt` (wavefront, materialx) through the
  production path; confirm the dragon is translucent at its geometric optical
  depth (not opaque gold/brown), and capture a labelled pbrt | before | after
  side-by-side at shared exposure.

## 4. Docs

- [x] 4.1 Update `docs/Subsurface.md` (or `docs/PbrtImport.md`) noting pbrt
  subsurface coefficients are stored per-world-unit (divided by `mm_per_unit`),
  and that visual parity additionally needs the deferred env/walk brightness work.
- [x] 4.2 Validate `openspec validate pbrt-subsurface-unit-scale --strict`.
