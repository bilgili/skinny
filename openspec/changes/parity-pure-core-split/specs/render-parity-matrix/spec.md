# render-parity-matrix — delta for parity-pure-core-split

## ADDED Requirements

### Requirement: Pure matrix core is hostless and the parity surface is preserved

The parity harness's pure matrix logic — the scene manifest schema and loader, the combo validity oracle, the anchor/axis-class bookkeeping, the self-consistency tolerance tables and lookup, and the pure result builders — SHALL live in a module (`skinny.pbrt.parity_core`) importable with no GPU, no renderer, and no USD (`pxr`) dependency. The GPU render adapter (`render_linear`, `render_combo`, `evaluate`) SHALL remain in `skinny.pbrt.parity`, which MUST continue to expose the full historical surface — including the consumed private names: `_DEFAULT_SELF_CONSISTENCY`, `_DEFAULT_SPECTRAL_SELF_CONSISTENCY`, and `_render_log` re-exported from the core, and `_scene_source` remaining in the adapter — so every existing `from skinny.pbrt.parity import …` and `parity.<name>` reference resolves unchanged, with unchanged behavior (including `render_log_path` / `SKINNY_RENDER_LOG`). The split SHALL NOT change any validity verdict, skip reason, tolerance value, baseline, or gate semantics.

#### Scenario: pure core imports without pxr or a GPU

- **WHEN** `skinny.pbrt.parity_core` is imported in an environment where
  `skinny.pbrt.api` / `pxr` and the renderer/GPU stack are unavailable
- **THEN** the import succeeds and `combo_is_valid`, `load_manifest`,
  `self_consistency_tol`, and the result builders are usable

#### Scenario: historical parity surface intact

- **WHEN** the names consumed by the existing tests and `furnace.py` (`SceneSpec`,
  `RenderCombo`, `ANCHOR`, `SPECTRAL_ANCHOR`, `all_combos`, `combo_is_valid`,
  `combo_axis_class`, `enumerate_combos`, `spectral_envelope`,
  `spectral_selfconsistency_assertable`, `self_consistency_anchor`,
  `self_consistency_tol`, `load_manifest`, `materialx_specs`, `reference_exists`,
  `pbrt_truth_result`, `absolute_radiance_result`, `self_consistency_result`,
  `authoring_equivalence_result`, `render_log_path`, `render_linear`,
  `render_combo`, `evaluate`, `scene_has_environment`, `INTEGRATORS`,
  `EXECUTION_MODES`, `_DEFAULT_SELF_CONSISTENCY`,
  `_DEFAULT_SPECTRAL_SELF_CONSISTENCY`, `_scene_source`, `_render_log`) are
  imported from `skinny.pbrt.parity`
- **THEN** every name resolves and behaves as before the split, with no edit to
  any existing test or consumer module

### Requirement: Spectral tolerance table is an overlay over the RGB table

The spectral self-consistency tolerance table SHALL be derived from the RGB default table plus an overlay containing only the rows that genuinely widen (`mode.relmse`, `integrator.relmse`), instead of a hand-maintained near-duplicate literal. The derived table MUST reproduce the pre-split table exactly, at full precision, and a hostless test SHALL assert that equality against a pinned literal so any drift in either table or the overlay fails the build. The overlay SHALL NOT loosen any RGB tolerance, and per-scene `self_consistency` / `spectral_self_consistency` overrides SHALL behave unchanged.

#### Scenario: derived spectral table equals the recorded values

- **WHEN** the hostless equality test compares the derived
  `_DEFAULT_SPECTRAL_SELF_CONSISTENCY` against the pinned pre-split literal
  ({mode: 0.03/0.03, integrator: 0.09/0.06, sppm: 0.15/0.12, mlt: 0.15/0.12,
  unbiased: 0.05/0.05})
- **THEN** the tables are equal at full precision and the test fails on any
  divergence

#### Scenario: RGB table untouched by the overlay

- **WHEN** `self_consistency_tol` is evaluated for any non-spectral combo after
  the split
- **THEN** it returns the same values as before the split (mode 0.02/0.03,
  integrator 0.06/0.06, sppm 0.15/0.12, mlt 0.15/0.12, unbiased 0.05/0.05),
  honouring per-scene overrides exactly as before
