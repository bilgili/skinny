# Design: unified-render-envelope-predicate

## Context

The render envelope — which (integrator × execution-mode × spectral × proposals × reuse × material-class) combinations actually run — is enforced in three independent layers that restate the same rules:

- **Parity matrix** (`src/skinny/pbrt/parity.py:270–383`): `spectral_envelope` + `combo_is_valid` return `(valid, reason)` per combo; docstrings say "Mirrors `cli_common.reject_spectral_unsupported`". Within the file the neural/env/reuse constraints appear twice — once in the spectral branch (:287–298), once in the RGB branch (:335–356).
- **CLI guards** (`src/skinny/cli_common.py`): `validate_render_flags` (:107–178), `reject_sppm_without_wavefront` (:224–239), `reject_mlt_unsupported` (:242–304), `reject_spectral_unsupported` (:346–411) — each restates its slice of the envelope as refusal prose and raises `SystemExit`. Consumed by all four front-ends (`app.py:533–545`, `ui/qt/app.py:698–710`, `headless.py:387–393`, `web_app.py:725–731`).
- **Renderer scene gate** (`renderer.py:~7227–7245`): refuses `--spectral` on a scene with non-flat material types — legitimately separate in *timing* (the CLI cannot see the material set), but the rule ("spectral is flat-only") is the same envelope fact.

Two capability flags — `spectral_capability.SPECTRAL_IMPLEMENTED`, `mlt_capability.MLT_IMPLEMENTED` — are already single-source and referenced live by both layers. They stay.

CLAUDE.md currently *prescribes* the duplication ("mirror this file's Compatibility matrix above", "keep the wording in sync" with README) and the `render-parity-matrix` spec says "The table SHALL mirror the documented compatibility matrix". This change supersedes that convention: the predicate becomes the source of truth; the doc tables become documentation of it.

Refusal wording is load-bearing: `tests/test_mlt_selection.py` matches `"no megakernel path"`; `tests/test_cli_common.py` asserts `"sppm"`/`"wavefront"` substrings and no-raise cases; `tests/pbrt/test_matrix.py` asserts skip-reason substrings ("wavefront-only", "not yet wired", etc.).

## Goals / Non-Goals

**Goals**

- One module answering "does combo X run, and if not, why" with a machine-readable reason; parity, all CLI guards, and the renderer scene gate consume it.
- Behavior-preserving: the exact same combo set is accepted/refused, gated by a before/after snapshot over `parity.all_combos()` × material classes.
- User-facing refusal text equivalent to today's; existing tests pass without wording edits.
- Fold the intra-parity spectral/RGB axis-constraint duplication into single per-axis rules.
- Coverage meta-tests keep failing the build on unregistered integrators; unification makes one registration cover parity *and* CLI.
- Supersede the CLAUDE.md/README mirror-in-lockstep workflow; keep the human-readable tables; add a doc-sync check.

**Non-Goals**

- No envelope changes — no combo becomes valid or invalid.
- No changes to the capability-flag mechanism (`SPECTRAL_IMPLEMENTED` / `MLT_IMPLEMENTED` stay the kill-switches).
- No GPU tests; everything here is hostless.
- Not unifying non-envelope validation (`--width`/`--height` positivity, MCP flag checks, bdpt-walk alias resolution) — those are argument hygiene, not envelope facts, and stay where they are.
- **The single-statement mandate is scoped to exactly three consumers** — the parity validity table, the four CLI guards, and the renderer spectral scene gate. Other pre-existing envelope echoes are explicitly out of scope and unchanged: `renderer.can_online_train` (renderer.py:9832–9854, the online-training prerequisite check), `parity.render_linear`'s sppm/mlt→wavefront execution-mode force (parity.py:558–559), and `DEFAULT_EXECUTION_FOR_INTEGRATOR`/`resolve_execution_mode`, which **stay in `cli_common`** (`test_mlt_selection.py:18` imports `DEFAULT_EXECUTION_FOR_INTEGRATOR` from `skinny.cli_common`; execution-mode *resolution* is a default-derivation concern, not a validity rule). Routing these through the predicate is a follow-up candidate, not this change.
- No change to runtime integrator-cycling behavior (e.g. the recorded "megakernel-fixed session cycling to MLT shows the path tracer" wart).

## Decisions

### D1 — Code predicate with small data tables, not a pure data table

The predicate is a function `evaluate(query) -> Verdict` over a frozen dataclass query (`integrator`, `execution_mode`, `proposals: tuple`, `reuse`, `spectral: bool`, `material_class: str = "flat"`, `online_training: bool = False`). Inside it, genuinely tabular facts live as module-level data (`WAVEFRONT_ONLY_INTEGRATORS = {"sppm", "mlt"}`, `LAYER_FREE_INTEGRATORS = {"mlt"}`, `DEFAULT_EXECUTION_FOR_INTEGRATOR` — already tabular in `cli_common`); conditional interactions (env proposal is path-only; volume scenes are path-only; spectral admits env on path only; MLT layer-free) stay as ordered `if` rules, one per rule, stated once.

*Alternative considered*: a fully data-driven validity table (rows of axis constraints). Rejected — the rules are conditional across axes (e.g. "env proposal requires path *when spectral*" vs "env proposal requires path + flat *in RGB*" differ only in material clause), so a table needs an expression language, which is more machinery than the ~15 `if` statements it replaces. The parity matrix's own docstring already calls `combo_is_valid` "one data-driven validity table"; the predicate keeps that spirit at the same complexity.

### D2 — Module location: `src/skinny/render_envelope.py`

A new top-level leaf module. Its only skinny imports are `spectral_capability` and `mlt_capability` (themselves leaves). `skinny.cli_common` already imports both capability modules; `skinny.pbrt.parity` already imports them too — so both consumers add one import with zero cycle risk. `renderer.py` imports it for the scene gate (renderer already imports half the package; no cycle since `render_envelope` never imports renderer/parity/cli).

*Alternative considered*: putting it in `cli_common` (parity imports cli_common). Rejected — drags argparse and every CLI helper into the parity harness's import graph and inverts the "mirrors" relationship instead of removing it.

### D3 — Verdict = ordered violation codes; guards own code-selection + prose

`evaluate(query)` returns **every** violated rule, in the predicate's canonical rule order — `Verdict(ok, violations=((code, reason), …))` — never just the first. A single-code verdict is unimplementable; three counterexamples, all pinned by existing tests:

- **bdpt + neural under megakernel**: parity's first-violated rule is neural-wavefront-only (`parity.py:335–339`), but the CLI must print the bdpt-incompatibility prose (`cli_common.py:173–178`, `test_cli_common.py:321–324`) — while **accepting** `path` | megakernel | +neural on that same neural-wavefront-only code (`test_cli_common.py:333–335`; the renderer strips the neural bit at runtime). Same first violation, opposite CLI outcomes depending on a *different* code's presence.
- **spectral + neural under megakernel**: the neural-wavefront-only violation fires first, but `reject_spectral_unsupported` must print its "BSDF/environment proposals" message (`test_reject_spectral_neural_raises`).
- **mlt × megakernel with the gate off**: the CLI reports `MLT_IMPLEMENTED` before the megakernel refusal (`cli_common.py:267` vs `:275`); parity refuses megakernel first (`parity.py:324` vs `:332`) — opposite precedence over the same two codes.

Consumption model: **parity** takes the first violation in predicate order (preserving today's skip reasons and precedence). **Each CLI guard** owns a declared subset of codes, recorded as data in the predicate module (e.g. `CLI_GUARD_CODES = {"reject_sppm_without_wavefront": (…), "reject_mlt_unsupported": (…), "reject_spectral_unsupported": (…), "validate_render_flags": (…)}`), scans the violation list for its owned codes **in its own precedence** (so `reject_mlt_unsupported` keeps not-yet-wired before megakernel), and maps each owned code to its existing `SystemExit` prose. Codes owned by **no** guard (neural-wavefront-only when the integrator is `path` — the runtime bit-strip case; likewise the env-proposal matrix-only rules) are deliberately not CLI-refused; that one-sided acceptance is part of the recorded ownership data, not an accident. The invariant is therefore not "no envelope logic in guards" but: **no envelope RULES in guards** — a guard owns only (a) which codes it enforces and (b) code→prose.

Reason strings stay **verbatim today's `combo_is_valid`/`spectral_envelope` strings** (e.g. `"SPPM is wavefront-only"`, `"spectral is incompatible with the neural proposal (v1)"`), so parity's recorded skips and `test_matrix.py` substring assertions are untouched. Verbatim preservation forces clause-per-reason granularity: expect ~18 codes (one per distinct reason string), not a coarse ~10 — e.g. neural-wavefront-only and spectral-no-neural are distinct codes even though both concern the neural axis.

Adapters must tolerate a parsed `Namespace` missing an axis a front-end suppresses — `test_cli_common.py:309–312` passes one without `execution_mode` — exactly as today's guards do via `getattr` defaults.

*Alternative considered*: one canonical message per code used by both layers. Rejected — parity wants terse skip labels, the CLI wants actionable multi-sentence prose naming flags; forcing one string degrades one or the other and forces test churn for zero de-duplication of *rules* (prose is not a rule).

### D4 — Scene-vs-flags split: one predicate, two call times

The predicate takes `material_class` with default `"flat"`. CLI guards call it with the default (they cannot know the scene) — flag-level refusals only. The renderer scene gate calls it again at material-pack time and scans the verdict for **only the spectral-nonflat code** — it must never newly refuse other rules (MLT/neural on non-flat, etc.) at runtime, where today's behavior is the recorded parity skip / runtime fallback, not a `SystemExit`. The renderer maps its packed int material-type codes to a query `material_class` via a **pinned mapping**: skin=0→`"skin"`, subsurface=4→`"subsurface"`, volume=5→`"volume"`; debug=2 and python=3 are *not* parity material classes and map to a non-flat class solely for this spectral gate (renderer.py:7234–7241 refuses them today, and must keep doing so). On the spectral-nonflat code it raises the existing message — the detail naming the offending type codes stays at the call site, appended to the verdict reason. Same rule, stated once, checked at the two times it is checkable. `parity.combo_is_valid` passes the scene's declared `material_class` as today.

### D5 — Guards stay as named functions; call sites untouched

`reject_sppm_without_wavefront`, `reject_mlt_unsupported`, `reject_spectral_unsupported`, and `validate_render_flags` keep their names and signatures (they encode real call-time contracts: effective startup integrator, resolved execution mode, persisted-integrator re-checks on interactive front-ends). Their bodies shrink to: build query → `evaluate` → map code to existing prose → raise. The four front-end call sites do not change. This keeps the diff minimal and the persisted-integrator subtleties (documented in the docstrings) exactly where they are exercised.

### D6 — Fold the intra-parity duplication

`combo_is_valid`'s RGB-branch neural/env/reuse rules (:335–356) and `spectral_envelope`'s restatement of the same axes (:287–298) collapse into the predicate — each **rule** stated once. "Fold" is deliberately modest: verbatim reason preservation (D3) forces clause-per-reason granularity, so the spectral variant of an axis rule keeps its own code and reason string (e.g. `"neural proposal is wavefront-only"` vs `"spectral is incompatible with the neural proposal (v1)"` are two clauses of one neural rule, both emitted when both apply). What is deleted is the *two-branch* structure — the same axis constraint evaluated in two places with drift potential — not the per-reason clauses. `spectral_envelope` remains as a thin public wrapper (its name is used by `test_matrix.py` and docs) delegating to the predicate with `spectral=True` forced; its independent rule copy is deleted.

### D7 — Behavior-preservation gate: snapshot the full query space, not `all_combos()`

`parity.all_combos()` (:386–405) is the **rendered-set enumerator** and deliberately sparse: it never enumerates spectral×reuse, never combines proposals with reuse, and has no online-training axis — so a snapshot taken from it cannot gate rules the predicate query exposes. The fixture is therefore generated from the **full cartesian query space**: integrator × execution mode × proposal subsets (∅/{env}/{neural}/{env,neural}) × reuse × spectral × material class {flat, subsurface, skin, volume} × online_training × megakernel_ok, each also under the capability-flag-off variants (`SPECTRAL_IMPLEMENTED=False`, `MLT_IMPLEMENTED=False`). For every point, the pre-port `(valid, reason)` comes from the **current** `combo_is_valid` (axes it lacks, like online_training, are recorded from the current CLI guard behavior); after the port the predicate must reproduce the map exactly. `all_combos()` keeps its rendered-set role unchanged, and its (subset) enumeration is *additionally* asserted identical before/after. The fixture is a committed JSON under `tests/assets/`; while the change is in flight it is the equivalence gate the proposal promises.

### D8 — Docs: tables stay, authority moves, drift gets a check

CLAUDE.md and README keep their human-readable compatibility-matrix tables (they are good docs), reworded to state they document `render_envelope.py`, which is authoritative. The CLAUDE.md parity-harness instruction "mirror this file's Compatibility matrix" and the `render-parity-matrix` spec's "SHALL mirror the documented compatibility matrix" are superseded (spec delta in this change). Doc-sync check: a hostless test derives a handful of hard envelope facts from the predicate (the wavefront-only integrator set; that neural/reuse are refused under spectral; that volume scenes are path-only) and asserts the README + CLAUDE.md matrix sections still state them (substring checks on the table text). Deliberately shallow — it catches "someone edited the envelope and forgot the docs", not full prose equivalence. <!-- ponytail: substring doc-check, upgrade to a generated-table diff only if drift actually recurs -->

## Risks / Trade-offs

- **[Risk] Hidden semantic mismatch between the mirrored copies surfaces as a behavior change during the port** (e.g. `reject_spectral_unsupported` deliberately delegates execution-mode refusals to the integrator-specific guards, while `spectral_envelope` refuses them itself). → Mitigation: D7's snapshot is taken from the *parity* predicate (the superset view); the CLI adapters are additionally covered by the existing `test_cli_common.py`/`test_mlt_selection.py` no-raise and raise cases, which encode the CLI-visible contract including guard ordering. Any mismatch found is resolved in favor of current observable behavior and noted in the task.
- **[Risk] Refusal-wording drift breaks substring-matching tests.** → Mitigation: D3 keeps parity reason strings verbatim and CLI prose byte-identical; the task list runs the three test files before and after with zero edits allowed.
- **[Risk] Import cycle via renderer.** → Mitigation: `render_envelope` imports only the two capability leaf modules; a task adds a trivial import-order test (`python -c "import skinny.render_envelope"` in isolation).
- **[Risk] Coverage meta-test weakens if it starts trusting the predicate blindly.** → Mitigation: the meta-test keeps comparing `renderer.integrator_modes` (what the app *exposes*) against the validity table (now the predicate) — that comparison is between two independent sources and stays a build-failer; a new integrator added to the renderer without a predicate rule still fails.
- **[Trade-off] Two prose layers remain (short parity reasons vs long CLI messages).** Accepted: prose duplication without logic duplication is cheap and each layer's audience genuinely differs; the reason *code* is the shared truth.
- **[Trade-off] The doc-sync check is substring-shallow.** Accepted: a generated-table pipeline is more machinery than the drift it prevents; revisit only if doc drift recurs.

## Migration Plan

1. **Snapshot first** (no production code touched): commit the D7 fixture generated from current `combo_is_valid`, plus the test that asserts against it (asserting current code — green by construction).
2. **Add `render_envelope.py`** with query/verdict/codes and unit tests; nothing imports it yet.
3. **Port parity**: `combo_is_valid`/`spectral_envelope` delegate; delete the duplicated branches and "Mirrors …" docstrings; `test_matrix.py` + snapshot gate must be green unchanged.
4. **Port CLI guards** to predicate adapters; `test_cli_common.py` + `test_mlt_selection.py` green unchanged; front-ends untouched.
5. **Port renderer scene gate** to consult the predicate; existing spectral non-flat refusal message preserved.
6. **Docs + doc-sync check**: CLAUDE.md (matrix section + parity-harness workflow supersession), README matrix wording, `docs/Architecture.md` Parity Matrix Harness section; add the doc-sync test.
7. Full hostless sweep + `ruff check src/` + `openspec validate`.

Rollback at any step is a plain revert; steps 3–5 are independently revertible because the guards keep their signatures.

**Sequencing with sibling change `parity-pure-core-split`:** that change lands **after** this one and moves parity's delegation from `pbrt/parity.py` into `parity_core.py`. This change targets `pbrt/parity.py` as it exists today; the later split relocates the (already thin) delegation call, not the predicate.

## Open Questions

- Should the D7 snapshot fixture be kept permanently as a golden envelope regression (cheap, catches accidental envelope changes forever) or deleted at archive time once the port is proven? Default: keep it — it is exactly the "enumerate via parity.all_combos before/after" gate, made durable.
- `spectral_envelope`: keep as a public alias forever, or deprecate after `test_matrix.py` migrates to the predicate? Default: keep the alias; deleting it buys nothing.
- Should `renderer.integrator_modes` itself eventually derive from the predicate's integrator table? Out of scope here (UI ordering concerns); noted for a follow-up.
