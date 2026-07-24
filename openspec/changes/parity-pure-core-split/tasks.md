# Tasks: parity-pure-core-split

## 1. Guard tests first (green against today's code)

- [x] 1.1 Add hostless import-surface compatibility test: import the full grepped surface from `skinny.pbrt.parity` — `SceneSpec`, `RenderCombo`, `ParityResult`, `ANCHOR`, `SPECTRAL_ANCHOR`, `INTEGRATORS`, `EXECUTION_MODES`, `PROPOSAL_AXES`, `REUSE_AXES`, `all_combos`, `combo_is_valid`, `combo_axis_class`, `enumerate_combos`, `spectral_envelope`, `spectral_selfconsistency_assertable`, `self_consistency_anchor`, `self_consistency_tol`, `load_manifest`, `materialx_specs`, `reference_exists`, `pbrt_truth_result`, `absolute_radiance_result`, `self_consistency_result`, `authoring_equivalence_result`, `render_log_path`, `render_linear`, `render_combo`, `evaluate`, `scene_has_environment`, `_DEFAULT_SELF_CONSISTENCY`, `_DEFAULT_SPECTRAL_SELF_CONSISTENCY`, `_scene_source`, `_render_log` — and exercise `combo_is_valid` + `self_consistency_tol` through the `parity` facade.
- [x] 1.2 Add hostless spectral-table equality test: pin the current `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` as a full-precision literal ({mode: 0.03/0.03, integrator: 0.09/0.06, sppm: 0.15/0.12, mlt: 0.15/0.12, unbiased: 0.05/0.05}) and assert dict equality; also pin the RGB table literal (mode 0.02/0.03, integrator 0.06/0.06, sppm 0.15/0.12, mlt 0.15/0.12, unbiased 0.05/0.05). Both tests pass before any move.

## 2. Module split (verbatim move, no behavior change)

- [x] 2.1 Create `src/skinny/pbrt/parity_core.py`: move the pure half verbatim (schema + `ParityResult` + combo/anchor/validity code + tolerance tables + `self_consistency_tol` + `load_manifest` + result builders + `materialx_specs` + `reference_exists` + `render_log_path`/`_render_log`); imports limited to stdlib, numpy, `skinny.{mlt,spectral}_capability`, `.metrics` (plus `skinny.render_envelope` once/if `unified-render-envelope-predicate` has landed — still hostless, see design D5).
- [x] 2.2 Shrink `src/skinny/pbrt/parity.py` to the GPU adapter (`render_linear`, `render_combo`, `evaluate`, `scene_has_environment`, `_repo_root`, `_scene_source`, `_usd_has_dome`, `_env_off_for`) plus one explicit `from .parity_core import …` re-export block naming every moved symbol including the underscore names (no `import *`).
- [x] 2.3 Move `from .api import import_pbrt` from module level into `render_linear`'s body; verify `import skinny.pbrt.parity` succeeds with `pxr` absent (e.g. monkeypatched out of `sys.modules`) and that `parity_core` never imports `.api`.

## 3. Overlay derivation

- [x] 3.1 Replace the `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` literal in `parity_core.py` with the overlay derivation (`_SPECTRAL_TOL_OVERLAY = {"mode": {"relmse": 0.03}, "integrator": {"relmse": 0.09}}` merged over `_DEFAULT_SELF_CONSISTENCY`); collapse the duplicated MLT-row comment to one site. Test 1.2 must stay green unmodified.

## 4. Verification and docs

- [x] 4.1 Run the hostless suites unchanged: `PYTHONPATH=src .venv/bin/python -m pytest tests/pbrt/test_matrix.py tests/pbrt/test_metrics.py tests/pbrt/test_parity.py tests/pbrt/test_suite.py -m "not gpu"` — zero edits to any existing test file allowed.
- [x] 4.2 `ruff check src/` clean (re-export block via `__all__` or `# noqa: F401`).
- [x] 4.3 Update `docs/Architecture.md` → Parity Matrix Harness with the `parity_core`/`parity` split; no CLAUDE.md/README behavior wording changes needed (surface unchanged).
- [x] 4.4 Sequencing check before merge: if `unified-render-envelope-predicate` has landed, rebase so its `combo_is_valid`/`spectral_envelope` delegation sits in `parity_core.py`; if not landed, note in that change's proposal that its parity edits target `parity_core.py`.
- [ ] 4.5 `openspec validate parity-pure-core-split` clean; archive after merge.
