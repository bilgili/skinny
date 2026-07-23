# Proposal: unified-render-envelope-predicate

## Why

"Does combo X run?" — the envelope over integrator × execution-mode × spectral × proposals × reuse × material-class — is restated in roughly ten places that must be edited in lockstep: `parity.spectral_envelope` + `parity.combo_is_valid` (whose docstrings literally say "Mirrors `cli_common.reject_spectral_unsupported`", and which state the neural/env/reuse axis constraints twice — once in the spectral branch, once in the RGB branch), the four CLI guards in `cli_common.py` (`validate_render_flags`, `reject_sppm_without_wavefront`, `reject_mlt_unsupported`, `reject_spectral_unsupported`), the renderer's scene-level spectral flat-material gate (renderer.py ~7229), and the CLAUDE.md / README compatibility-matrix tables. Every envelope change (spectral-wavefront, spectral-mlt, mlt-integrator each touched all of them) risks silent drift: a combo the parity matrix renders but the CLI refuses, or vice versa. The mirroring is currently *prescribed* by CLAUDE.md ("The table SHALL mirror the documented compatibility matrix" / "mirror this file's Compatibility matrix"), so the duplication is load-bearing convention, not accident — it needs a recorded supersession, not a quiet refactor.

## What Changes

- **New module `src/skinny/render_envelope.py`** — one predicate: a combo query (integrator, execution mode, proposals, reuse, spectral, material class, online-training) → a verdict listing **every** violated rule as `(machine-readable code, human-readable reason)` pairs in canonical order, plus recorded per-guard code-ownership data (which codes each CLI guard enforces; which codes no guard owns). It references `spectral_capability.SPECTRAL_IMPLEMENTED` and `mlt_capability.MLT_IMPLEMENTED` **live** (the two flags stay the single-source kill-switches, unchanged) and imports nothing above the capability modules, so both `skinny.pbrt.parity` and `skinny.cli_common` can import it without cycles.
- **`parity.combo_is_valid` / `parity.spectral_envelope` delegate to the predicate.** The intra-parity duplication (neural/env/reuse constraints stated separately in the spectral and RGB branches) folds into single per-axis rules evaluated for both. Skip-reason strings are preserved verbatim so recorded skips and matrix tests keep their wording.
- **CLI guards become thin adapters.** Each `reject_*` builds a query from the parsed args and the resolved execution mode, scans the predicate's verdict for the codes that guard **owns** (recorded ownership data; unowned codes — e.g. the runtime-stripped neural-under-megakernel case — never refuse at the CLI), in the guard's own precedence, and maps each owned code to its existing user-facing `SystemExit` prose — refusal text seen by users stays equivalent (tests assert on substrings like "no megakernel path", "sppm"/"wavefront", "incompatible with --integrator bdpt"; the prose stays in `cli_common` keyed by reason code, so `tests/test_cli_common.py` and `tests/test_mlt_selection.py` pass unchanged). No envelope *rules* remain in the guards — a guard owns only its code selection and code→prose mapping. Front-end call sites (`app.py`, `ui/qt/app.py`, `headless.py`, `web_app.py`) are unchanged in shape.
- **Renderer scene gate shares the predicate.** The runtime spectral flat-material refusal (the CLI legitimately cannot see the material set) builds a query with the actual scene material class and consults the same predicate; its message (naming the offending material type codes) is retained as the detail text.
- **Behavior-preserving, gated.** A hostless equivalence gate snapshots `(valid, reason)` for every `parity.all_combos()` combo × material class **before** the port and asserts the post-port predicate reproduces it exactly — same combos accepted, same combos refused.
- **Coverage meta-tests strengthened, not weakened.** `tests/pbrt/test_matrix.py::test_coverage_meta_app_integrators_covered` keeps failing the build on an integrator with no validity entry; because the CLI now consumes the same table, an unregistered integrator is uncovered *everywhere* at once.
- **Recorded-decision supersession: CLAUDE.md + README.** The CLAUDE.md workflow text that prescribes mirror-in-lockstep editing ("mirror this file's Compatibility matrix above"; "keep the wording in sync") is superseded: the human-readable compatibility-matrix tables in CLAUDE.md and README **stay**, but are restated as *documentation of the predicate* — `render_envelope.py` is the source of truth, and a lightweight hostless doc-sync check asserts the key envelope facts (wavefront-only integrators, spectral/neural/reuse exclusions) still appear in the doc tables.

## Capabilities

### New Capabilities
- `render-envelope`: the single combo-envelope predicate — one module answering "does combo X run, and if not, why", consumed by the parity matrix, every CLI refusal guard, and the renderer scene-level gate; capability flags referenced live; documented matrix kept in sync by a hostless check.

### Modified Capabilities
- `render-parity-matrix`: the "Parity matrix is derived from a validity table" requirement now derives the table from the shared `render-envelope` predicate instead of mirroring the documented matrix by hand; skip reasons come from the predicate's verdicts.
- `render-cli`: the "Reject impossible render-flag combinations at startup" requirement now derives each refusal from the shared predicate's reason codes, with user-facing refusal text equivalent to today's.

## Impact

- `src/skinny/render_envelope.py` — new (the predicate + reason codes).
- `src/skinny/pbrt/parity.py` — `spectral_envelope`/`combo_is_valid` (:270–383) delegate; "Mirrors cli_common…" docstrings deleted; duplicated axis constraints folded.
- `src/skinny/cli_common.py` — `validate_render_flags` (:107–178), `reject_sppm_without_wavefront` / `reject_mlt_unsupported` (:224–304), `reject_spectral_unsupported` (:346–411) become predicate adapters; refusal prose unchanged.
- `src/skinny/renderer.py` — scene-level spectral gate (~:7229) consults the predicate.
- `src/skinny/spectral_capability.py` / `src/skinny/mlt_capability.py` — unchanged flags; docstrings note the predicate as the consumer.
- Front-ends `app.py` / `ui/qt/app.py` / `headless.py` / `web_app.py` — call sites unchanged (same guard functions).
- Tests: new `tests/test_render_envelope.py` (predicate unit tests + before/after combo-set equivalence gate + doc-sync check); `tests/pbrt/test_matrix.py`, `tests/test_cli_common.py`, `tests/test_mlt_selection.py` expected green without wording edits. All hostless.
- Docs: CLAUDE.md (compatibility-matrix section wording + the parity-harness "mirror" instruction superseded), README Compatibility matrix wording, `docs/Architecture.md` (Parity Matrix Harness section).
- Out of scope (unchanged, recorded in design Non-Goals): `renderer.can_online_train`, `parity.render_linear`'s execution-mode forcing, `DEFAULT_EXECUTION_FOR_INTEGRATOR` (stays importable from `skinny.cli_common`).
- Sequencing: the sibling change `parity-pure-core-split` lands **after** this change and relocates parity's delegation into `parity_core.py`.
