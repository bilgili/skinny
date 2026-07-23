# Proposal: parity-pure-core-split

## Why

`src/skinny/pbrt/parity.py` (796 lines) bundles two different kinds of code in one module: **pure, hostless matrix logic** — the `SceneSpec` manifest schema + `load_manifest` (:51–141, :492–496), the combo validity oracle `combo_is_valid`/`spectral_envelope`/`all_combos`/`enumerate_combos` (:270–412), anchor/axis-class bookkeeping (:226–267, :415–431), the self-consistency tolerance tables + lookup (:434–489), and the pure result builders (:668–759) — and the **GPU render adapter** `render_linear`/`render_combo`/`evaluate` (:508–665, :762–771). The GPU imports are already lazy (function-body imports of `HeadlessRenderer`), but the module-level `from .api import import_pbrt` (parity.py:25) drags `pxr`/USD into every import of the pure logic, and `import_pbrt` is used by exactly one function (`render_linear`, :616).

Within the pure half, `_DEFAULT_SELF_CONSISTENCY` (:436–445) and `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` (:457–466) are near-identical twins — five rows each, identical except `mode.relmse` 0.02→0.03 and `integrator.relmse` 0.06→0.09, with the four-line MLT-row comment copied verbatim (:440–443 ≡ :461–464). A future tolerance-class addition (a new integrator) must be made twice, and a divergence between the copies would be silent.

**Honesty note:** the architecture review rated this *Speculative* — the pain today is low (imports work, the tables agree, tests pass). The proposal is deliberately minimal: relocate + derive, change no behavior, no value, no gate semantics, and no test wording.

## What Changes

- **New module `src/skinny/pbrt/parity_core.py`** holding the pure, hostless half verbatim: `SceneSpec`, `ParityResult`, `RenderCombo`, `INTEGRATORS`/`EXECUTION_MODES`/`PROPOSAL_AXES`/`REUSE_AXES`, `ANCHOR`/`SPECTRAL_ANCHOR`/`self_consistency_anchor`, `spectral_selfconsistency_assertable`, `spectral_envelope`, `combo_is_valid`, `all_combos`, `enumerate_combos`, `combo_axis_class`, the tolerance tables + `self_consistency_tol`, `load_manifest`, the result builders (`pbrt_truth_result`, `absolute_radiance_result`, `self_consistency_result`, `authoring_equivalence_result`), `materialx_specs`, `reference_exists`, and the log-path helpers `render_log_path`/`_render_log` (stdlib-only). It imports only stdlib + numpy + `skinny.{mlt,spectral}_capability` + `.metrics` — no `pxr`, no renderer.
- **`parity.py` becomes the GPU render adapter + compatibility facade**: it keeps `render_linear`, `render_combo`, `evaluate`, `scene_has_environment`, `_scene_source`/`_repo_root`/`_usd_has_dome`/`_env_off_for`, and re-exports every moved name (including the test-consumed private names `_DEFAULT_SELF_CONSISTENCY`, `_DEFAULT_SPECTRAL_SELF_CONSISTENCY`) so `from skinny.pbrt.parity import …` and `parity.<name>` resolve exactly as today. Zero test churn; `tests/pbrt/test_matrix.py`, `test_parity.py`, `test_suite.py`, `test_emissive_nee.py`, `test_convergence.py`, `test_bdpt_energy.py`, `test_radiometric_parity.py`, `tests/test_sppm_gpu.py`, and `src/skinny/pbrt/furnace.py` are untouched.
- **The module-level `from .api import import_pbrt` moves into `render_linear`** (its sole consumer), so importing `skinny.pbrt.parity` itself no longer requires `pxr` — matching the module docstring's stated intent ("imports without a GPU").
- **The spectral tolerance table becomes an overlay diff over the RGB table**: `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` is derived at module load from `_DEFAULT_SELF_CONSISTENCY` plus an overlay containing only the two rows that genuinely widen (`mode.relmse` 0.03, `integrator.relmse` 0.09). A hostless test asserts the derived table equals the exact current literal — tolerance VALUES do not change, and the RGB table is untouched.
- **No behavior change anywhere**: same validity verdicts, same skip reasons, same tolerances, same gate semantics, same `SKINNY_RENDER_LOG` behavior.

## Capabilities

### New Capabilities

_None — no new user- or harness-facing capability; this is an internal module split under the existing parity-matrix capability._

### Modified Capabilities

- `render-parity-matrix`: adds a requirement that the pure matrix logic (schema, validity oracle, tolerance lookup, result builders) is importable without GPU/USD dependencies, that the historical `skinny.pbrt.parity` surface remains intact via re-export, and that the spectral tolerance table is derived from the RGB table by an overlay that reproduces the current values exactly. Existing requirements (metric battery, validity table, gates) are unchanged.

## Impact

- `src/skinny/pbrt/parity_core.py` — new (moved code, no rewrites).
- `src/skinny/pbrt/parity.py` — shrinks to the render adapter + explicit re-exports; `import_pbrt` import made lazy.
- `tests/pbrt/test_parity.py` (or a small new hostless test file) — two added tests: (1) spectral-table overlay equality against the pinned current literal; (2) import-surface compatibility (every consumed name importable from `skinny.pbrt.parity`; `parity_core` importable with `skinny.pbrt.api`/`pxr` absent).
- Docs: `docs/Architecture.md` → Parity Matrix Harness (module split noted); CLAUDE.md parity-harness pointer unchanged in substance.
- **Sequencing:** sibling change `unified-render-envelope-predicate` rewrites `combo_is_valid`/`spectral_envelope` to delegate to a shared predicate. This change lands **after** it (or rebases on it); `parity_core.py` is then exactly where the parity-side delegation to that predicate lives. Landing order the other way is also mechanically fine (the predicate change would edit `parity_core.py` instead of `parity.py`), but after-is-simpler and is the recorded plan.
