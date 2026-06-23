## Context

Parity is enforced today by `tests/pbrt/test_parity.py::test_corpus_scene_parity_gate`,
which calls `src/skinny/pbrt/parity.py::evaluate()`. `render_linear()` already
*accepts* `integrator` and `execution_mode` arguments, but `evaluate()` hardcodes
the default (`path`, `megakernel`) and the corpus is six 128² synthetic scenes.
So:

- Only one of {Path, BDPT, SPPM} × {megakernel, wavefront} is ever gated.
- ReSTIR DI and the neural directional proposal are never gated for parity.
- The real heavy scenes — `assets/bathroom.usda` (imported from pbrt
  `contemporary-bathroom`, 1200²) and the dragon — are not in the corpus at all.
- `assets/bathroom.usda` currently mismatches pbrt and nothing catches it.

The valid renderer combinations are already documented in CLAUDE.md →
**Compatibility matrix** and README.md. The local pbrt v4 binary
(`~/projects/pbrt-v4/build/pbrt`) and the source scene
(`~/projects/pbrt-v4-scenes/contemporary-bathroom/`) are available for offline
reference generation.

## Goals / Non-Goals

**Goals:**
- A **data-driven parity matrix**: combos are generated from one validity table
  that mirrors the CLAUDE.md compatibility matrix; pytest sweeps every valid
  (scene × combo) and skips invalid ones with an explicit reason.
- **Dual gating** per (scene, combo): pbrt-truth (relMSE/FLIP vs the checked-in
  pbrt v4 EXR) **and** self-consistency (equality to a designated golden combo
  within a per-axis tolerance).
- Add **bathroom** and **dragon** as first-class heavy corpus scenes with
  checked-in pbrt v4 reference EXRs.
- **Recorded-baseline** handling: a combo with a known pbrt-truth mismatch
  records its measured metrics and gates against *regression past the baseline*
  (not against the strict tolerance), so the suite is green and the deltas are
  pinned and visible. The actual bathroom/dragon corrections are follow-ups.
- An **extensibility contract**: a meta-test that fails if a renderer combo the
  app exposes (new integrator, new mode) has no matrix coverage — so "new feature
  ⇒ tested against all renderers" is enforced, not aspirational.
- A `not gpu` import-only tier that constructs the matrix and imports/loads every
  scene with no GPU, so "tests always work for all integrators" holds on any host.

**Non-Goals:**
- **Fixing** the bathroom mismatch or the dragon brightness residual — explicit
  follow-up changes, unblocked by this harness.
- Adding new integrators, BSDF lobes, or samplers.
- Changing any `integrator-convergence` requirement (those convergence rules are
  unchanged; this harness enforces them across more combos).
- Online training as a matrix axis (training is orthogonal to render parity; only
  the neural proposal's *inference-time* effect on the image is an axis).
- GPU-skinned / animation parity (static frames only).

## Decisions

### D1 — Single validity table drives the combo set
Define a `RenderCombo(integrator, execution_mode, proposals, reuse)` and a
`COMBO_VALIDITY` predicate table keyed by axis interactions, mirroring the
CLAUDE.md matrix:
- SPPM ⇒ `execution_mode == wavefront` (no megakernel SPPM).
- neural proposal ⇒ `wavefront` **and** scene material-class is `flat` (skin/SSS
  scenes skip neural by design); BDPT ignores the neural proposal (skip).
- ReSTIR DI ⇒ direct-light reuse layer (wavefront; megakernel skip).
- A scene declares a `max_geometry` / `material_class` so a too-heavy or
  wrong-material combo is pruned (e.g. dragon's 28.8M-tri mesh ⇒ megakernel
  invalid — it OOMs — so dragon is wavefront-only).

`combo_is_valid(combo, scene)` returns `(bool, reason)`. Invalid combos are
`pytest.skip(reason)`, never silently dropped.

*Alternative considered:* hand-listing each valid combo per scene. Rejected — a
new integrator/mode would need edits in N places and could silently miss a combo.
The table is the single source of truth.

### D2 — Self-consistency anchored on `(Path, wavefront)` with per-axis tolerances
The golden anchor per scene is `(Path, wavefront, no-proposal, no-reuse)`: it is
the unbiased baseline and the only integrator that supports every axis (neural,
ReSTIR, SPPM are all wavefront). Every other valid combo asserts exposure-aligned
equality to the anchor within a tolerance chosen by *which axis differs*:
- mode-only diff (`Path-megakernel` vs anchor) → **tight** (same estimator, must
  match to MC noise).
- integrator diff (`BDPT`, `SPPM`) → **looser** (SPPM caustics, BDPT connection
  noise).
- proposal/reuse diff (`+neural`, `+ReSTIR`) → **unbiasedness** tolerance: the
  proposal/reuse changes variance, not expectation, so the converged image must
  match the anchor.

This reuses the spirit of existing gates (`test_wavefront_path_ab`,
`test_metal_wavefront_*_ab`, neural-unbiased gates) but unifies them under one
matrix. Self-consistency is independent of the bathroom-vs-pbrt bug: a shared
material/brightness error makes megakernel and wavefront *both* wrong identically,
so self-consistency stays green while the pbrt-truth gate goes to baseline.

*Alternative considered:* anchor on `(Path, megakernel)` (the historically-gated
combo). Rejected — megakernel cannot run neural/ReSTIR, so wavefront-only axes
would have no anchor to compare against.

### D3 — Dual gate, with recorded baselines for known mismatches
`evaluate(spec, combo)` returns both the pbrt-truth metrics and the
self-consistency metrics. The manifest carries, per (scene, combo) where needed:
- `relmse_tol` / `flip_tol` (strict pbrt-truth tolerance), and
- optional `baseline: {relmse, flip}` — when present, the pbrt-truth assertion is
  `metric <= max(tol, baseline * (1 + margin))` and the test logs
  `delta = metric - baseline`. Self-consistency always uses the strict per-axis
  tolerance (no baseline escape — invariants must hold).

This keeps the suite green today (bathroom/dragon record their current pbrt
delta) while pinning against further drift; the follow-up fixes will tighten the
baseline toward `tol`.

*Alternative considered:* `xfail(strict=True)`. Rejected — xfail hides the
measured number and flips to a failure the moment a fix *partially* improves the
metric; a numeric baseline tracks progress and guards regression simultaneously.

### D4 — Heavy scenes load the existing `.usda` asset; refs generated offline
Extend `SceneSpec` so a scene source is either a corpus `.pbrt` (imported at gate
time, as today) **or** a `usd` asset path (`assets/bathroom.usda`,
`assets/dragon_sss.usda`) loaded directly — `render_linear` branches on which is
set. The pbrt reference EXRs are produced **offline** by a documented,
non-test-time helper that overrides the pbrt film `xresolution/yresolution` down
to the corpus resolution (e.g. 256²) and runs the pinned pbrt binary; the gate
consumes only the checked-in EXR (no pbrt at test time, matching today's design).

*Alternative considered:* copy the full `contemporary-bathroom.pbrt` (plus all its
geometry/textures) into the corpus and import it in-gate. Rejected — multi-hundred-MB
assets and slow in-gate import; the `.usda` is already the skinny-side scene of
record.

### D5 — Cost tiers
Small resolution (128–256²) and per-combo spp budgets keep wall-clock bounded.
Tiers:
- `not gpu`: matrix construction + every scene imports/loads (no render). Always.
- `gpu` (default): the full matrix at corpus resolution.
- `slow` (opt-in): higher spp / higher res confirmation runs.
Apple-Silicon thermal rule from prior Metal work applies: one guarded Metal
megakernel compile at a time; heavy sweeps prefer wavefront / `not gpu`.

### D6 — One canonical `ImageMetrics` battery
All reported numbers flow through a single `compute_metrics(img, ref=None) ->
ImageMetrics` in `metrics.py`, extending the existing `relmse`/`flip`/
`align_exposure`/`read_exr` (kept). Battery:
- **error vs reference** (exposure-aligned): `mse`, `rmse`, `mae`, `relmse`,
  `psnr` (peak = aligned reference max, documented), `flip`.
- **single-image quality**: `variance` (global luminance variance), `noise_sigma`
  (Immerkær fast Laplacian estimator — dependency-free), `firefly_fraction`
  (fraction of pixels whose luminance exceeds a 3×3-neighborhood median by a large
  factor — an outlier/spike detector).
All implemented in pure numpy (no scipy/skimage dep — matches the repo's
no-network constraint). Gates assert on the relevant field(s); the harness logs
the whole struct so every run prints fireflies/variance/PSNR/error uniformly. This
makes "standardize the numbers" structural: a new call-site cannot invent its own
error formula.

*Alternative considered:* pull in `scikit-image`/`OpenImageIO` metrics. Rejected —
new heavy deps on an offline box, and the existing relMSE/FLIP already define the
house convention the battery must stay consistent with.

## Risks / Trade-offs

- **Combinatorial blow-up of (scene × combo)** → D1 validity table prunes
  aggressively; cost tiers (D5) keep the default sweep small; `not gpu` tier
  carries coverage on hostless CI.
- **GPU memory: dragon 28.8M tris OOMs the megakernel; Metal megakernel
  cold-compile is a RAM bomb** → dragon is wavefront-only by validity (D1); heavy
  combos run wavefront; sweeps obey the one-compile-at-a-time thermal rule.
- **Monte-Carlo noise causes false self-consistency failures** → exposure-align
  before compare; per-axis tolerances sized to the spp budget; deterministic seeds
  where the renderer allows.
- **pbrt reference drift across pbrt versions** → pin the pbrt commit in the
  manifest (as today) and document the offline regen step; refs are regenerated
  deliberately, never in-gate.
- **Recorded baselines can mask a true regression if set too loose** → baseline
  margin is small and explicit; self-consistency invariants never use a baseline;
  the follow-up fix must *lower* the baseline, never raise it.
- **dragon material class**: `dragon_sss.usda` is subsurface (neural axis invalid,
  pbrt residual ~0.097 vs 0.211 per prior work) — recorded as baseline; a flat
  `dragon.usda` variant may also be added for the neural/flat axes. Final scene
  selection is an open question below.

## Migration Plan

Purely additive: new `parity.py` matrix code, new `test_parity.py`
parametrization, new corpus manifest entries + EXRs, optional new corpus `.usda`
copies. No code or requirement removed; existing single-combo behavior becomes the
`(Path, *)` rows of the matrix. Rollback = revert the change; refs regenerated via
the documented offline helper. Implementation is done in a dedicated git worktree
off `main` per project convention.

## Open Questions

- **Corpus resolution / spp budget** for bathroom & dragon that balances CI cost
  vs. a meaningful gate — start 256² and tune.
- **Which dragon**: `dragon_sss.usda` (subsurface, the user's likely target, known
  residual) and/or `dragon.usda` (flat, exercises the neural/ReSTIR axes). Lean
  toward including the SSS dragon as the heavy baseline scene and optionally the
  flat dragon for the flat-only axes.
- **ReSTIR DI / neural availability in the headless harness**: confirm
  `HeadlessRenderer`/`RenderOptions` can arm ReSTIR reuse and the neural proposal
  for a parity render (they are runtime-selectable on the GUI; verify the headless
  path exposes the same switches).
