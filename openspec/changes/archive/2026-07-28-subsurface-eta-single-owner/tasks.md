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
      eta)`. `eta` is REQUIRED — the `eta=None` self-resolving fallback was a
      second contract, and a pre-merge review measured it re-resolving `eta` on
      the emission path (see 6.1).

## 3. media.py consumes the resolved intermediate

- [x] 3.1 `media.subsurface_overrides` takes the mapper's `inputs` dict and
      applies only the mm-per-unit division and the `ior` carry. `api`
      hands it `inputs` at both authoring sites.
- [x] 3.2 It receives no `ParamSet` at all; the docstring records the split, and
      the five emitted keys are enumerated rather than spread, so a key added to
      the resolver cannot land on `skinnyOverrides` unrouted.

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
- [x] 5.3 `slang_layout.INTAKE_ONLY_KEYS` comment for `subsurface_eta` updated:
      the `ior` agreement is STRUCTURAL, not empirical-across-the-corpus. Was
      not applicable while this branch was cut from `2ac8aba` — `INTAKE_ONLY_KEYS`
      arrived with `flat-material-field-table` (`main` at `a652bee`), which was
      merged in before this branch landed.
- [x] 5.4 `openspec validate --strict`; pre-merge review (group 6).

## 6. Pre-merge review fold

The codex runtime stalled (8h, no log output, cancelled), so the review ran
through the fallback review subagent, per the standing rule.

- [x] 6.1 BLOCKER, folded — the "one eta resolution" claim was FALSE as first
      implemented. `api._author_material` calls `map_material` AND
      `media.subsurface_overrides` separately, so `eta` was resolved twice per
      material (measured: 2 calls to `get_float_texture` for `eta` per import),
      and the emitter's copy is the one that reaches the stage. Fixed by having
      emission consume the resolved intermediate. A test now counts the calls and
      fails at two, and asserts the shader `ior` equals the `skinnyOverrides`
      `ior` — the pair that could genuinely diverge.
- [x] 6.2 `eta=None` sentinel removed (it existed only for 6.1's second caller).
- [x] 6.3 `{**coeffs, …}` spread replaced with the five explicit keys.
- [x] 6.4 `_PARAMSET_METHODS` gained `int`/`ints`; the phantom `spectrum`
      (not a `ParamSet` method) removed. The structural gate now pins the
      signature, which no rename or missing method name can evade.
- [x] 6.5 Test-quality fixes: the note assertion matched a string that only
      appears for an UNRESOLVABLE texture (an imagemap binding would have made it
      a false failure) — now matches the bare param name. The `slang_layout.
      INTAKE_ONLY_KEYS` citation, which does not exist in `src/` on this branch,
      now cites `material_pack.pack_flat_material` directly.
- [x] 6.6 Verified-fine, no action: KnownBugs item 3 (checked empirically — all
      six params raise on both a texture and a named-spectrum binding, in both
      call paths); `1.33` → `ETA_DEFAULT` equivalence; no `ior` conflict between
      the shader input and the override (the loader applies overrides last and
      both descend from one `eta`); no import cycle.
- [x] 6.7 Re-gated after the fold: 836 passed / 16 pre-existing failures,
      `ruff check src/skinny` clean, `openspec validate --strict` valid, and the
      byte-identity check still shows 74 of 74 real scenes unchanged.
