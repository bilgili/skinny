# Tasks: unified-render-envelope-predicate

Sequencing note: the sibling change `parity-pure-core-split` lands **after**
this change and moves parity's delegation into `parity_core.py`; group 3 here
targets `pbrt/parity.py` as it exists today.

## 1. Baseline snapshot — the before/after equivalence gate (D7)

- [ ] 1.1 Add `tests/test_render_envelope.py` with a snapshot generator over the **full cartesian query space** — integrator × execution mode × proposal subsets (∅/{env}/{neural}/{env,neural}) × reuse × spectral × material class (`flat`, `subsurface`, `skin`, `volume`) × `online_training` × `megakernel_ok` — through the **current** `parity.combo_is_valid` (axes it lacks, e.g. `online_training`, recorded from current CLI guard behavior); commit the `(query) → (valid, reason)` map as a JSON fixture under `tests/assets/`. `parity.all_combos()` is NOT the generator (it never enumerates spectral×reuse, proposals+reuse, or online-training — it stays the rendered-set enumerator only); its subset enumeration is snapshotted additionally
- [ ] 1.2 Snapshot the capability-flag-off variants too (monkeypatch `SPECTRAL_IMPLEMENTED=False`, `MLT_IMPLEMENTED=False`) into the same fixture
- [ ] 1.3 Add the equivalence test asserting the live `combo_is_valid` (and the `all_combos()` rendered set) reproduces the fixture exactly — green by construction before any refactor lands

## 2. The predicate module (D1–D3)

- [ ] 2.1 New `src/skinny/render_envelope.py`: frozen query dataclass (`integrator`, `execution_mode`, `proposals`, `reuse`, `spectral`, `material_class="flat"`, `online_training=False`); `evaluate(query)` returns a verdict listing **all** violated rules as `(code, reason)` pairs in canonical rule order (never just the first — D3 counterexamples); the ordered rule set; and the per-guard code-ownership data (`CLI_GUARD_CODES`) recording which codes each CLI guard enforces and which codes no guard owns — importing only `spectral_capability` / `mlt_capability`, referenced live
- [ ] 2.2 Reason strings verbatim from today's `combo_is_valid` / `spectral_envelope` (e.g. "SPPM is wavefront-only", "spectral is incompatible with the neural proposal (v1)", "… not yet wired …") so recorded skips keep their wording — clause-per-reason granularity, expect ~18 codes (one per distinct reason string)
- [ ] 2.3 Unit tests: one accept + one refuse case per reason code; a multi-violation case asserting ordering (e.g. bdpt+neural under megakernel lists both the neural-wavefront-only and bdpt-no-neural codes); ownership-table coverage (every code in `CLI_GUARD_CODES` exists; the deliberately unowned codes are listed); capability-flag monkeypatch flips the "not yet wired" verdicts; isolated-import test (`skinny.render_envelope` imports with no renderer/parity/cli modules loaded)

## 3. Port the parity matrix (D6)

- [ ] 3.1 `parity.combo_is_valid` and `parity.spectral_envelope` delegate to the predicate (`spectral_envelope` kept as a thin public wrapper); delete the duplicated RGB-branch vs spectral-branch neural/env/reuse rules and the "Mirrors `cli_common.reject_spectral_unsupported`" docstrings
- [ ] 3.2 Gate: `tests/pbrt/test_matrix.py` passes with **zero** edits; the group-1 snapshot equivalence test passes against the fixture
- [ ] 3.3 Confirm the coverage meta-tests (`test_coverage_meta_app_integrators_covered`, `test_coverage_meta_execution_modes_pinned`, `test_coverage_meta_spectral_axis_covered`) still fail on an unregistered integrator (temporarily extend `renderer.integrator_modes` in-test to prove it)

## 4. Port the CLI guards (D5)

- [ ] 4.1 `reject_sppm_without_wavefront`, `reject_mlt_unsupported`, `reject_spectral_unsupported`, and the envelope branches of `validate_render_flags` in `src/skinny/cli_common.py` become predicate adapters: build query (resolved execution mode, effective startup integrator semantics unchanged; tolerate a `Namespace` missing suppressed axes via `getattr` defaults — `test_cli_common.py:309–312` passes one without `execution_mode`) → `evaluate` → scan the verdict for the guard's **owned** codes (`CLI_GUARD_CODES`) in the guard's own precedence (MLT guard: not-yet-wired before megakernel, opposite of parity) → map each owned code to the **existing** `SystemExit` prose, byte-identical; unowned codes never refuse (path|megakernel|+neural stays accepted)
- [ ] 4.2 Non-envelope validation (`--width`/`--height` positivity) stays in `validate_render_flags` untouched; guard signatures and all four front-end call sites (`app.py`, `ui/qt/app.py`, `headless.py`, `web_app.py`) unchanged
- [ ] 4.3 Gate: `tests/test_cli_common.py` and `tests/test_mlt_selection.py` pass with **zero** edits (wording preserved); any semantic mismatch discovered between the old mirrored copies is resolved in favor of current observable behavior and noted here
- [ ] 4.4 Update `spectral_capability.py` / `mlt_capability.py` docstrings to name the predicate as the consumer (flags themselves unchanged)

## 5. Port the renderer scene gate (D4)

- [ ] 5.1 The spectral non-flat refusal in `renderer.py` (~:7229) builds a query from the packed material types via a **pinned** int-code→material_class mapping (skin=0→"skin", subsurface=4→"subsurface", volume=5→"volume"; debug=2/python=3 are not parity classes — they map to a non-flat class solely for this gate, staying refused as today per renderer.py:7234–7241) and scans the verdict for **only the spectral-nonflat code** — it must not newly refuse any other rule (MLT/neural on non-flat, etc.) at runtime; the existing message naming the offending material type codes is kept as the detail text
- [ ] 5.2 Hostless tests: predicate refuses `spectral` × each non-flat material class with the spectral-nonflat code; a non-spectral query over the same classes triggers no renderer-gate refusal (only-that-code scan proven)

## 6. Docs + doc-sync check (D8) — the recorded-decision supersession

- [ ] 6.1 CLAUDE.md: reword the Compatibility matrix intro and the Parity-matrix-harness bullet ("mirror this file's Compatibility matrix") to state `render_envelope.py` is the source of truth and the tables document it; keep the tables
- [ ] 6.2 README Compatibility matrix: same authority rewording; keep the tables
- [ ] 6.3 `docs/Architecture.md` → Parity Matrix Harness: describe the shared predicate and the consumer set (parity, CLI guards, renderer scene gate)
- [ ] 6.4 Add the hostless doc-sync test: derive the wavefront-only integrator set and the spectral-excluded axes from the predicate and assert the README + CLAUDE.md matrix sections state them (substring check)

## 7. Validation

- [ ] 7.1 Full hostless sweep: `.venv/bin/pytest tests/test_render_envelope.py tests/test_cli_common.py tests/test_mlt_selection.py tests/pbrt/test_matrix.py -m "not gpu"` green; `ruff check src/` clean
- [ ] 7.2 Final before/after equivalence check against the group-1 fixture (same combos accepted/refused, same reasons); decide whether the fixture stays as a permanent golden (default: keep)
- [ ] 7.3 `openspec validate unified-render-envelope-predicate` clean; archive per workflow after merge
