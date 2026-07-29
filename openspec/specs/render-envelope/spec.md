# render-envelope Specification

## Purpose
TBD - created by archiving change unified-render-envelope-predicate. Update Purpose after archive.
## Requirements
### Requirement: Single render-envelope predicate
The renderer SHALL provide one predicate module (`skinny.render_envelope`) that answers, for any combination of integrator, execution mode, directional proposals, reuse mode, spectral flag, material class, and online-training flag, whether the combination runs — returning a verdict listing **every** violated rule as `(machine-readable reason code, human-readable reason)` pairs in the predicate's canonical rule order (not only the first violation). For its three consumers — the parity validity table, the four CLI refusal guards, and the renderer's spectral scene-level gate — the predicate SHALL be the single statement of the render envelope: none of these SHALL restate an envelope **rule** (which integrator is wavefront-only, which axes are refused under spectral, which material classes an integrator supports, and so on) as independent logic. Consumers own only **code selection and presentation**: which verdict codes they enforce (recorded as code-ownership data in the predicate module), in what precedence, and the code→prose mapping. Pre-existing envelope echoes outside these three consumers (the online-training prerequisite check `renderer.can_online_train`, the parity harness's sppm/mlt→wavefront execution-mode forcing in `render_linear`, and the execution-mode default-derivation table `DEFAULT_EXECUTION_FOR_INTEGRATOR` in `cli_common`) are outside this requirement's scope and unchanged. The module SHALL import nothing above the capability-flag modules, so the parity harness, the CLI layer, and the renderer can all import it without cycles. The capability flags `spectral_capability.SPECTRAL_IMPLEMENTED` and `mlt_capability.MLT_IMPLEMENTED` SHALL remain the live kill-switches, referenced by the predicate at evaluation time (not captured at import) so a test monkeypatch takes effect.

#### Scenario: refusals cannot contradict the rendered set
- **WHEN** the same combination is evaluated by the parity matrix's validity check, a CLI refusal guard, and the renderer scene-level gate
- **THEN** all three derive their decision from the same predicate verdict: a combination the CLI refuses is never in the matrix's rendered set, and a combination the matrix renders is never refused by the CLI — while combinations the CLI deliberately accepts despite a matrix skip (violation codes owned by no guard, e.g. the neural proposal under the megakernel with the path integrator, which the renderer strips at runtime) are recorded in the code-ownership data, not accidental

#### Scenario: capability flags gate live
- **WHEN** `SPECTRAL_IMPLEMENTED` (or `MLT_IMPLEMENTED`) is monkeypatched off
- **THEN** the predicate reports the envelope-eligible spectral (or MLT) combinations invalid with a distinct "not yet wired" reason code, without any consumer-side special-casing

#### Scenario: verdicts list all violations, machine-readably
- **WHEN** the predicate refuses a combination violating multiple rules (e.g. `bdpt` with the neural proposal under the megakernel)
- **THEN** the verdict lists every violated rule as a stable `(code, reason)` pair in canonical order, with codes distinct per rule clause (e.g. wavefront-only, layer-free, spectral-excluded axis, non-flat material, not-yet-wired), so each consumer can select the code it owns rather than being forced to act on the first violation

### Requirement: Envelope unification is behavior-preserving
Adopting the shared predicate SHALL NOT change the render envelope: the exact set of combinations accepted and refused SHALL be identical before and after the unification, and user-facing refusal text SHALL remain equivalent (existing CLI and matrix tests that assert on refusal wording SHALL pass without wording edits). The equivalence SHALL be gated over the **full cartesian query space** — integrator × execution mode × proposal subsets × reuse × spectral × material class × online-training × megakernel-budget, including the capability-flag-off variants — not merely over `parity.all_combos()`, which is the rendered-set enumerator and deliberately omits axes and axis combinations (spectral×reuse, proposals+reuse, online-training) that the predicate covers. `parity.all_combos()` SHALL keep its rendered-set enumeration role unchanged.

#### Scenario: combo-set equivalence gate
- **WHEN** the full cartesian query space is enumerated through the predicate-backed validity check and compared against a committed snapshot of the pre-unification `(valid, reason)` results (with the `parity.all_combos()` rendered-set enumeration additionally compared)
- **THEN** every combination has the identical validity and reason

#### Scenario: refusal wording preserved
- **WHEN** a CLI front-end refuses an out-of-envelope combination after the unification
- **THEN** the error message is equivalent to the pre-unification message for that combination (same incompatibility named, same suggested fix), and the pre-existing wording-assertion tests pass unchanged

### Requirement: Documented compatibility matrix documents the predicate
The human-readable compatibility-matrix tables SHALL be retained as
documentation **of** the predicate, with the predicate module as the stated
source of truth — superseding the prior convention that code mirrors the
documented matrix. The documented tables SHALL live in `CLAUDE.md` and in
`docs/RenderingModes.md`; `README.md` SHALL link `docs/RenderingModes.md`
instead of holding a second copy of the matrix. A hostless doc-sync check SHALL
assert that key envelope facts derived from the predicate (at minimum: the
wavefront-only integrator set, and the axes refused under spectral) are stated
in the documented tables, so envelope edits that skip the docs fail a test. The
checked file set SHALL name the documents that hold the tables, so moving a
table to another document means updating the checked set in the same change.

#### Scenario: doc drift fails the check
- **WHEN** an envelope rule covered by the doc-sync check changes in the
  predicate but the documented compatibility tables are not updated
- **THEN** the hostless doc-sync check fails, naming the stale fact

#### Scenario: the documented table moves to another document
- **WHEN** a change moves a documented compatibility table to a different
  Markdown file
- **THEN** the same change updates the doc-sync check's file set to the new
  path, and the check passes against the new location

