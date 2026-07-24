# Tasks: unified-render-envelope-predicate

Sequencing note: the sibling change `parity-pure-core-split` lands **after**
this change and moves parity's delegation into `parity_core.py`; group 3 here
targets `pbrt/parity.py` as it exists today.

## 1. Baseline snapshot — the before/after equivalence gate (D7)

- [x] 1.1 Add `tests/test_render_envelope.py` with a snapshot generator over the **full cartesian query space** — integrator × execution mode × proposal subsets (∅/{env}/{neural}/{env,neural}) × reuse × spectral × material class (`flat`, `subsurface`, `skin`, `volume`) × `online_training` × `megakernel_ok` — through the **current** `parity.combo_is_valid` (axes it lacks, e.g. `online_training`, recorded from current CLI guard behavior); commit the `(query) → (valid, reason)` map as a JSON fixture under `tests/assets/`. `parity.all_combos()` is NOT the generator (it never enumerates spectral×reuse, proposals+reuse, or online-training — it stays the rendered-set enumerator only); its subset enumeration is snapshotted additionally
- [x] 1.2 Snapshot the capability-flag-off variants too (monkeypatch `SPECTRAL_IMPLEMENTED=False`, `MLT_IMPLEMENTED=False`) into the same fixture
- [x] 1.3 Add the equivalence test asserting the live `combo_is_valid` (and the `all_combos()` rendered set) reproduces the fixture exactly — green by construction before any refactor lands

## 2. The predicate module (D1–D3)

- [x] 2.1 New `src/skinny/render_envelope.py`: frozen query dataclass (`integrator`, `execution_mode`, `proposals`, `reuse`, `spectral`, `material_class="flat"`, `online_training=False`); `evaluate(query)` returns a verdict listing **all** violated rules as `(code, reason)` pairs in canonical rule order (never just the first — D3 counterexamples); the ordered rule set; and the per-guard code-ownership data (`CLI_GUARD_CODES`) recording which codes each CLI guard enforces and which codes no guard owns — importing only `spectral_capability` / `mlt_capability`, referenced live
- [x] 2.2 Reason strings verbatim from today's `combo_is_valid` / `spectral_envelope` (e.g. "SPPM is wavefront-only", "spectral is incompatible with the neural proposal (v1)", "… not yet wired …") so recorded skips keep their wording — clause-per-reason granularity, expect ~18 codes (one per distinct reason string)
- [x] 2.3 Unit tests: one accept + one refuse case per reason code; a multi-violation case asserting ordering (e.g. bdpt+neural under megakernel lists both the neural-wavefront-only and bdpt-no-neural codes); ownership-table coverage (every code in `CLI_GUARD_CODES` exists; the deliberately unowned codes are listed); capability-flag monkeypatch flips the "not yet wired" verdicts; isolated-import test (`skinny.render_envelope` imports with no renderer/parity/cli modules loaded)

## 3. Port the parity matrix (D6)

- [x] 3.1 `parity.combo_is_valid` and `parity.spectral_envelope` delegate to the predicate (`spectral_envelope` kept as a thin public wrapper); delete the duplicated RGB-branch vs spectral-branch neural/env/reuse rules and the "Mirrors `cli_common.reject_spectral_unsupported`" docstrings
- [x] 3.2 Gate: `tests/pbrt/test_matrix.py` passes with **zero** edits; the group-1 snapshot equivalence test passes against the fixture
- [x] 3.3 Confirm the coverage meta-tests (`test_coverage_meta_app_integrators_covered`, `test_coverage_meta_execution_modes_pinned`, `test_coverage_meta_spectral_axis_covered`) still fail on an unregistered integrator (temporarily extend `renderer.integrator_modes` in-test to prove it)

## 4. Port the CLI guards (D5)

- [x] 4.1 `reject_sppm_without_wavefront`, `reject_mlt_unsupported`, `reject_spectral_unsupported`, and the envelope branches of `validate_render_flags` in `src/skinny/cli_common.py` become predicate adapters: build query (resolved execution mode, effective startup integrator semantics unchanged; tolerate a `Namespace` missing suppressed axes via `getattr` defaults — `test_cli_common.py:309–312` passes one without `execution_mode`) → `evaluate` → scan the verdict for the guard's **owned** codes (`CLI_GUARD_CODES`) in the guard's own precedence (MLT guard: not-yet-wired before megakernel, opposite of parity) → map each owned code to the **existing** `SystemExit` prose, byte-identical; unowned codes never refuse (path|megakernel|+neural stays accepted)
- [x] 4.2 Non-envelope validation (`--width`/`--height` positivity) stays in `validate_render_flags` untouched; guard signatures and all four front-end call sites (`app.py`, `ui/qt/app.py`, `headless.py`, `web_app.py`) unchanged
- [x] 4.3 Gate: `tests/test_cli_common.py` and `tests/test_mlt_selection.py` pass with **zero** edits (wording preserved); any semantic mismatch discovered between the old mirrored copies is resolved in favor of current observable behavior and noted here
- [x] 4.4 Update `spectral_capability.py` / `mlt_capability.py` docstrings to name the predicate as the consumer (flags themselves unchanged)

## 5. Port the renderer scene gate (D4)

- [x] 5.1 The spectral non-flat refusal in `renderer.py` (~:7229) builds a query from the packed material types via a **pinned** int-code→material_class mapping (skin=0→"skin", subsurface=4→"subsurface", volume=5→"volume"; debug=2/python=3 are not parity classes — they map to a non-flat class solely for this gate, staying refused as today per renderer.py:7234–7241) and scans the verdict for **only the spectral-nonflat code** — it must not newly refuse any other rule (MLT/neural on non-flat, etc.) at runtime; the existing message naming the offending material type codes is kept as the detail text
- [x] 5.2 Hostless tests: predicate refuses `spectral` × each non-flat material class with the spectral-nonflat code; a non-spectral query over the same classes triggers no renderer-gate refusal (only-that-code scan proven)

## 6. Docs + doc-sync check (D8) — the recorded-decision supersession

- [x] 6.1 CLAUDE.md: reword the Compatibility matrix intro and the Parity-matrix-harness bullet ("mirror this file's Compatibility matrix") to state `render_envelope.py` is the source of truth and the tables document it; keep the tables
- [x] 6.2 README Compatibility matrix: same authority rewording; keep the tables
- [x] 6.3 `docs/Architecture.md` → Parity Matrix Harness: describe the shared predicate and the consumer set (parity, CLI guards, renderer scene gate)
- [x] 6.4 Add the hostless doc-sync test: derive the wavefront-only integrator set and the spectral-excluded axes from the predicate and assert the README + CLAUDE.md matrix sections state them (substring check)

## 7. Validation

- [x] 7.1 Full hostless sweep: `.venv/bin/pytest tests/test_render_envelope.py tests/test_cli_common.py tests/test_mlt_selection.py tests/pbrt/test_matrix.py -m "not gpu"` green; `ruff check src/` clean
- [x] 7.2 Final before/after equivalence check against the group-1 fixture (same combos accepted/refused, same reasons); decide whether the fixture stays as a permanent golden (default: keep)
- [x] 7.3 `openspec validate unified-render-envelope-predicate` clean; archive per workflow after merge

## Notes (recorded during implementation)

**4.3 — semantic mismatches found between the mirrored copies.** All resolved in
favour of current observable behaviour; the group-1 fixture reproduces byte-identical
guard prose for all 768 CLI query points × 3 capability variants.

1. **`--execution-mode auto` reaches the guards.** `validate_render_flags` runs
   *before* `resolve_execution_mode` on the interactive front-ends, so `auto` is a
   real input. Today's guards test `(mode or "megakernel") == "megakernel"`, so
   `auto` never refuses — a naive `mode != "wavefront"` port would have started
   refusing `--integrator sppm` at that call site. Preserved by `cli_common._envelope_mode`.
2. **`reject_spectral_unsupported` refused *any* proposal token outside `{bsdf, env}`**,
   not just `neural` — reachable through `SKINNY_PROPOSALS`, which argparse does not
   validate against `choices`. Parity's rule was the narrower `has_neural`. Unified as
   "any proposal layer other than `env`" under the verbatim `SPECTRAL_NO_NEURAL` reason
   string, which preserves both (parity never enumerates a third token).
3. **`reject_spectral_unsupported` does not own the wavefront-only codes** (a spectral
   `sppm`+megakernel passes it and is refused by `reject_sppm_without_wavefront`),
   whereas `spectral_envelope` refuses it itself. Not reconciled — recorded as code
   ownership, which is exactly what `CLI_GUARD_CODES` exists to express.
4. **Parity fuses MLT proposals+reuse into one reason; the CLI prints three messages**
   (proposals / reuse / online-training). Split into `MLT_NO_PROPOSALS` /
   `MLT_NO_REUSE` / `ONLINE_TRAINING_PATH_ONLY`, the first two sharing parity's verbatim
   reason string, so `combo_is_valid`'s first-violation text is unchanged.
5. **`spectral_envelope`'s precedence is spectral-axis-first**, unlike `combo_is_valid`'s
   canonical order (which reports SPPM/MLT wavefront-only first). Preserved via the
   `SPECTRAL_ENVELOPE_CODES` scan order rather than flattened.
6. `ONLINE_TRAINING_PATH_ONLY` is deliberately **shared** by two guards (`reject_mlt_unsupported`
   and the bdpt branch of `validate_render_flags`), each printing its own prose — one
   rule, two presentations. The early `if integrator != …: return` structure is kept in
   both guards, because it is what stops `sppm`+`--online-training` (accepted today) from
   newly refusing.

**5.1 deviation.** The pinned int-code→material-class mapping and the gate helper live in
`render_envelope.py` (`MATERIAL_CLASS_FOR_TYPE_CODE`, `spectral_refuses_material_types`)
rather than in `renderer.py`, which the design's letter suggested. Reason: `renderer.py`
imports `vulkan` at module load, so a helper defined there is not testable in the hostless
sweep. `renderer.py` calls the helper; the scan is still `RENDERER_SCENE_CODES`-only.

**7.2 — fixture disposition: KEPT** as a permanent golden
(`tests/assets/render_envelope_snapshot.json`, ~390 kB, string-interned). It is the
"enumerate the query space before/after" gate made durable, and it costs one test file to
regenerate deliberately when an envelope change is intended.

**Test-suite baseline.** `tests/pbrt/test_parity.py` shows 6 pre-existing
`test_corpus_scene_imports_cleanly_mtlx[*]` failures — identical on clean `main` under the
same interpreter (the 3.12 `.venv` lacks `PyMaterialXGenSlang`). Not a regression.
