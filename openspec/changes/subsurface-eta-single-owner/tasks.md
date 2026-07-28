# Tasks: subsurface-eta-single-owner

## 1. Record the defect

- [x] 1.1 Capture the pre-change baseline: import every parity-corpus and
      confirming-suite scene on both flavors, hash the `.usda` / `.mtlx` output.
      76 imports (37 resolvable manifest scenes + the `all_mtypes` fixture ×
      2 flavors), 0 errors.
- [x] 1.2 Write the failing regression test: a `subsurface` material with
      `"spectrum eta" "glass-LASF9"` imports on both flavors, and
      `ior == subsurface_eta == 1.85004` in the authored `skinnyOverrides`.
      Red first: both flavors, plus a texture-bound `eta`, raised
      `ValueError: could not convert string to float`.

## 2. One eta resolution in the resolver

- [x] 2.1 `materials._subsurface_overrides` takes the resolved `eta` as an
      argument; the hand-rolled named-spectrum branch is removed.
- [x] 2.2 `resolve_material`'s subsurface branch passes `lobes["ior"]` into it,
      and spells the default `subsurface.ETA_DEFAULT`.
- [x] 2.3 Promote the builder to a public `subsurface_medium_overrides(params,
      eta=None)`; when `eta` is None it resolves through `get_float_texture`
      with `notes=None`.

## 3. media.py delegates

- [x] 3.1 `media.subsurface_overrides` calls the resolver's builder and applies
      only the mm-per-unit division and the `ior` carry.
- [x] 3.2 Its own `ParamSet` reads are removed; the docstring records the split.

## 4. Gates

- [x] 4.1 Regression test from 1.2 passes on both flavors.
- [x] 4.2 Texture-bound `eta` imports and reports exactly one note.
- [x] 4.3 Hostless source gate: no `ParamSet` read remains in `media.py`'s
      subsurface path. Two AST gates in `tests/pbrt/test_material_resolve.py` —
      the syntactic one reuses the existing `_reads_in` detector, the structural
      one refuses any mention of `params` outside the delegation call, so the
      first cannot be walked around by aliasing.
- [x] 4.4 Byte-identical corpus check against the 1.1 baseline: 74 of 76
      identical. The two that differ are `fixture_all_mtypes` on both flavors,
      where task 5.1 deliberately added a material.
- [x] 4.5 `tests/pbrt` hostless sweep: 806 passed, 16 failed — the same 16 fail
      on unmodified `main` in this worktree (5 corpus scenes whose assets live
      only in the primary checkout, and an absolute-vs-relative `.mtlx` path
      assertion). No new failure; 6 new tests pass.
      `ruff check src/skinny` clean.

## 5. Documentation

- [x] 5.1 `KnownBugs.md` item 4 removed; item 3 rewritten to name the whole
      remaining surface (`reflectance`, `sigma_a`, `sigma_s`, `mfp` via
      `ParamSet.rgb`; `g`, `scale` via `ParamSet.floats`) and to record that one
      fix now covers both call paths. `tests/pbrt/fixtures/all_mtypes.pbrt` gains
      the named-spectrum `eta` material and its comment names only the case that
      still crashes.
- [x] 5.2 `docs/PbrtImport.md` records the resolver/emission split and the single
      `eta` resolution.
- [ ] 5.3 `slang_layout.INTAKE_ONLY_KEYS` comment for `subsurface_eta` — NOT
      APPLICABLE on this branch. `INTAKE_ONLY_KEYS` does not exist in `src/` yet;
      it arrives with the unmerged `flat-material-field-table` change. That
      change must state the `ior`/`subsurface_eta` agreement is structural, not
      empirical.
- [x] 5.4 `openspec validate --strict`; codex pre-merge review.
