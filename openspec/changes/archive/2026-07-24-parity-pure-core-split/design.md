# Design: parity-pure-core-split

## Context

`skinny.pbrt.parity` is the parity-matrix harness: manifest schema, combo validity oracle, self-consistency tolerance tables, result builders, and the GPU render orchestration, all in one 796-line module. The heavy renderer imports are already lazy (inside `render_linear`), so the real coupling today is only:

1. the module-level `from .api import import_pbrt` (parity.py:25) — `api.py` imports `pxr` at module top, so *any* import of the pure matrix logic requires a working USD install;
2. the twin tolerance tables `_DEFAULT_SELF_CONSISTENCY` (:436) / `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` (:457) — five rows each, differing only in `mode.relmse` (0.02→0.03) and `integrator.relmse` (0.06→0.09), with a four-line comment duplicated verbatim.

The de-facto public surface, measured by grepping `tests/` + `src/` (not assumed):

- **Pure names consumed:** `SceneSpec`, `RenderCombo`, `ParityResult` (via builders), `ANCHOR`, `SPECTRAL_ANCHOR`, `INTEGRATORS`, `EXECUTION_MODES`, `all_combos`, `combo_is_valid`, `combo_axis_class`, `enumerate_combos`, `spectral_envelope`, `spectral_selfconsistency_assertable`, `self_consistency_anchor`, `self_consistency_tol`, `load_manifest`, `materialx_specs`, `reference_exists`, `pbrt_truth_result`, `absolute_radiance_result`, `self_consistency_result`, `authoring_equivalence_result`, `render_log_path` (+ `SKINNY_RENDER_LOG` env behavior, asserted by `test_parity.py:269–291`).
- **GPU/adapter names consumed:** `render_linear` (test_convergence, test_bdpt_energy, furnace.py), `render_combo`, `evaluate`, `scene_has_environment`.
- **Private names consumed (must survive):** `_DEFAULT_SELF_CONSISTENCY` + `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` (test_matrix.py), `_scene_source` (furnace.py:118, test_parity.py:52/94), `_render_log` (called directly by test_parity.py:274–275).
- Consumers: `tests/pbrt/test_matrix.py`, `test_parity.py`, `test_suite.py`, `test_emissive_nee.py`, `test_convergence.py`, `test_bdpt_energy.py`, `test_radiometric_parity.py`, `tests/test_sppm_gpu.py`, `src/skinny/pbrt/furnace.py`. CLAUDE.md/README document `parity.render_log_path()`.

The architecture review rated this split **Speculative** — low pain today. Scope is therefore deliberately minimal: move + re-export + derive one table; nothing else.

## Goals / Non-Goals

**Goals**
- The pure matrix logic (schema, validity, tolerances, result builders) importable with no `pxr`, no renderer, no GPU.
- `from skinny.pbrt import parity` and every name above (public *and* the three consumed private names) resolve exactly as today — zero churn in tests, furnace.py, docs.
- Single tolerance source: spectral table = RGB table + minimal overlay, provably equal to the current values.

**Non-Goals**
- Changing any tolerance value, baseline, gate semantics, skip reason, or `SKINNY_RENDER_LOG` behavior. The overlay must reproduce the current table bit-for-bit; a mismatch is a bug in this change, never a reason to touch a value.
- Restructuring the validity oracle's logic (that is `unified-render-envelope-predicate`'s job).
- Splitting further (metrics, furnace, manifest-as-its-own-module, a `parity/` package). Opportunistic-only; the two-file split is the whole change.
- Any behavior change visible to the matrix sweep or the CLI.

## Decisions

### D1 — Two files, not a package
`src/skinny/pbrt/parity_core.py` (pure) + the existing `parity.py` (GPU adapter + facade). A `parity/` package with `__init__` re-exports was considered and rejected: it renames the module object (`skinny.pbrt.parity` becomes a package), risks subtle `__file__`/pickling/monkeypatch differences, and buys nothing over a sibling module. Code moves verbatim — no rewrites, so the diff is reviewable as a pure move.

**What lands in `parity_core.py`:** everything listed under "Pure names consumed" above, plus `ParityResult`, `PROPOSAL_AXES`/`REUSE_AXES`, `_render_log`, and both tolerance tables. Its imports: stdlib (`json`, `os`, `tempfile`, `time`, `dataclasses`), `numpy`, `skinny.mlt_capability`, `skinny.spectral_capability`, `from . import metrics` (metrics is already hostless — the `__init__` docstring lists it among the lightweight submodules).

**What stays in `parity.py`:** `render_linear`, `render_combo`, `evaluate`, `scene_has_environment` (lazy parser/state import), `_repo_root`, `_scene_source`, `_usd_has_dome`, `_env_off_for` — the scene-source/env helpers are orchestration policy consumed only by the render path and furnace.py, and furnace.py already imports them from `parity`, so leaving them put is the no-churn choice.

### D2 — Re-export by explicit `from .parity_core import` list, not `*`
`parity.py` opens with one explicit import block naming every moved symbol, including `_DEFAULT_SELF_CONSISTENCY`, `_DEFAULT_SPECTRAL_SELF_CONSISTENCY`, and `_render_log` (underscore names are skipped by `*`, and tests consume all three via the facade — test_matrix.py the tables, test_parity.py:274–275 `_render_log` — so `*` would silently break them). Explicit also keeps `ruff` able to see what is intentionally re-exported (`# noqa: F401` on the block or `__all__`). A compatibility test asserts the full grepped surface resolves from `skinny.pbrt.parity`.

### D3 — Make the `import_pbrt` import lazy in `render_linear`
`import_pbrt` is used once (:616). Moving `from .api import import_pbrt` into `render_linear`'s body makes `parity.py` itself pxr-free at import time, exactly like the renderer imports already are. This is the one line of the change that alters `parity.py`'s import-time side effects; it is what the module docstring already claims ("imports without a GPU"). Callers see no difference — `render_linear` behaves identically.

### D4 — Spectral table as overlay, guarded by an equality assert
```python
_SPECTRAL_TOL_OVERLAY = {"mode": {"relmse": 0.03}, "integrator": {"relmse": 0.09}}
_DEFAULT_SPECTRAL_SELF_CONSISTENCY = {
    cls: {**tol, **_SPECTRAL_TOL_OVERLAY.get(cls, {})}
    for cls, tol in _DEFAULT_SELF_CONSISTENCY.items()
}
```
The name and shape of `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` are unchanged (test_matrix.py reads it; `self_consistency_tol` copies it). The overlay carries **only** the rows that genuinely widen; the sppm/mlt/unbiased rows and every `flip` value flow from the RGB table, so a future class addition is written once. A hostless test pins the derived dict against a literal copy of today's table (full precision), so any drift — in either table or the overlay — fails loudly. The duplicated MLT comment collapses to one site.

Alternative considered: a `self_consistency_tol`-internal widening rule ("spectral ⇒ relmse×1.5"). Rejected — the widths are measured floors, not a ratio, and test_matrix.py reads the table as data.

### D5 — Sequencing vs `unified-render-envelope-predicate`
That sibling change rewrites `combo_is_valid`/`spectral_envelope` to delegate to a shared `render_envelope` predicate. Recorded order: **this change lands after it** (or rebases onto it), so the delegation lands once and `parity_core.py` is the file that imports the predicate (`render_envelope` sits above `pbrt/`, so no cycle). If this change happens to land first, the predicate change edits `parity_core.py` instead of `parity.py` — mechanically equivalent, but the after-ordering avoids a move-then-rewrite double diff on the same functions.

## Risks / Trade-offs

- **[Risk] A consumed name is missed in the re-export list** → Mitigation: the compatibility test imports the exact grepped surface (public + the three private names) from `skinny.pbrt.parity` and is written before the move (fails red on omission); `ruff` + the existing hostless suites (`test_matrix.py`/`test_parity.py`/`test_suite.py` collection) run unchanged as the backstop.
- **[Risk] Overlay derivation silently changes a tolerance** → Mitigation: the pinned-literal equality test (D4), full precision, added in the same commit as the derivation; the recorded rule "never loosen self-consistency to hide divergence" makes any future edit to the overlay reviewable as a value change.
- **[Risk] Monkeypatching in tests targets `parity.<name>` and the moved function reads its module-global** → Mitigation: audit before moving — today's tests monkeypatch env vars and `spectral_capability`/`mlt_capability` flags (read via module reference, unaffected), not parity globals; the compatibility test also exercises `combo_is_valid` through the `parity` facade to catch any indirection break.
- **[Risk] Merge conflict churn with `unified-render-envelope-predicate`** → Mitigation: D5 records the sequencing here (land after, or rebase); whichever change merges second rebases its parity edits onto the other's file layout — the functions are the same either way.
- **Trade-off:** one more module in `pbrt/` for a pain rated low today. Accepted because the change is a verbatim move with two genuinely deduplicating deltas (lazy pxr import, overlay table) and zero consumer churn — the minimum that makes the pure logic hostless.

## Migration Plan

1. Add the two hostless tests first (import-surface compatibility; spectral-table literal equality against the *current* twin table) — green against today's code.
2. Create `parity_core.py` (verbatim move), shrink `parity.py` to adapter + explicit re-export block, make `import_pbrt` lazy.
3. Replace the spectral literal with the overlay derivation; the equality test keeps the pinned literal.
4. Run the hostless suites (`tests/pbrt/test_matrix.py`, `test_metrics.py`, `test_parity.py -m "not gpu"`, `test_suite.py` collection) + `ruff check src/`. No GPU sweep required — no rendered pixel can change (no shader, no dispatch, no tolerance edit); if any parity-adjacent doubt arises, one suite scene through `render_combo` on Metal is the spot check.
5. Docs: `docs/Architecture.md` Parity Matrix Harness section notes the split. No consumer edits anywhere.

Rollback: revert the commit — no data, no persisted format, no manifest change.

## Open Questions

- Should `ParityResult`'s forward-ref annotations (`"RenderCombo | None"`) switch to real types once co-located in `parity_core.py`? (Cosmetic; decide at implementation.)
- If `unified-render-envelope-predicate` stalls, does this change still land first? Default per D5: yes, it may — the predicate change then targets `parity_core.py`.
