## ADDED Requirements

### Requirement: Standardized image-quality metric battery
The harness SHALL compute every reported image number through one canonical metric
module. The battery SHALL include error-vs-reference metrics (`MSE`, `RMSE`, `MAE`,
`relMSE`, `PSNR`, `FLIP`) and single-image quality statistics (`variance`, a
noise-σ estimate, and a `firefly` outlier fraction). A single `compute_metrics`
entry point SHALL return all of them in one `ImageMetrics` struct; no gate or
report SHALL compute "error" or "noise" by an ad-hoc inline formula. Error metrics
SHALL be computed after exposure alignment, consistently with the existing parity
comparison.

#### Scenario: one entry point yields the full battery
- **WHEN** a rendered image and a reference are passed to `compute_metrics`
- **THEN** the returned struct exposes MSE, RMSE, MAE, relMSE, PSNR, FLIP,
  variance, noise-σ, and firefly fraction, all computed on the exposure-aligned
  pair

#### Scenario: single-image stats need no reference
- **WHEN** only a rendered image is passed (no reference)
- **THEN** the variance, noise-σ, and firefly fraction are still computed and the
  error-vs-reference fields are reported as absent

#### Scenario: every gate uses the canonical battery
- **WHEN** the pbrt-truth gate or the self-consistency gate reports a number
- **THEN** that number comes from `compute_metrics` / `ImageMetrics`, not a
  call-site-local formula

### Requirement: Parity matrix is derived from a validity table
The parity harness SHALL derive the set of rendered combinations from a single
validity table over the axes `integrator ∈ {Path, BDPT, SPPM}`, `execution_mode ∈
{megakernel, wavefront}`, `proposals ⊇ {neural}`, and `reuse ⊇ {ReSTIR DI}`. The
table SHALL mirror the documented compatibility matrix. Every (scene × combo)
SHALL be either exercised or skipped with an explicit, machine-readable reason; no
valid combo SHALL be silently dropped.

#### Scenario: SPPM is wavefront-only
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(SPPM, megakernel)` is skipped with reason "SPPM is wavefront-only"
- **AND** `(SPPM, wavefront)` is present in the rendered set

#### Scenario: neural proposal requires wavefront and a flat material scene
- **WHEN** the matrix is enumerated for a subsurface/skin scene (e.g. the SSS dragon)
- **THEN** every combo carrying the neural proposal is skipped with reason
  "neural proposal is flat-material + wavefront only"
- **AND** for a flat-material scene, `(Path, wavefront, +neural)` is present

#### Scenario: BDPT ignores the neural proposal
- **WHEN** the matrix is enumerated
- **THEN** no `BDPT` combo carries the neural proposal (skipped by design)

### Requirement: megakernel and wavefront produce the same image
For any integrator that runs in both execution modes, the harness SHALL assert
that the megakernel and wavefront renders of the same scene agree. The
exposure-aligned relMSE/FLIP between the two modes SHALL be within the tight
mode-equivalence tolerance for that scene.

#### Scenario: Path megakernel matches Path wavefront
- **WHEN** the same scene is rendered with `(Path, megakernel)` and `(Path, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

#### Scenario: BDPT megakernel matches BDPT wavefront
- **WHEN** the same scene is rendered with `(BDPT, megakernel)` and `(BDPT, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

### Requirement: Integrators converge to a common golden image
The harness SHALL designate `(Path, wavefront, no-proposal, no-reuse)` as the
self-consistency anchor per scene and assert that every other valid integrator
combo agrees with the anchor within a per-integrator tolerance (looser for SPPM
caustics and BDPT connection noise than for a pure mode change).

#### Scenario: BDPT matches the Path anchor
- **WHEN** a scene is rendered with BDPT and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the
  integrator-equivalence tolerance

#### Scenario: SPPM matches the Path anchor within its caustic tolerance
- **WHEN** a scene is rendered with `(SPPM, wavefront)` and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the (looser)
  SPPM-equivalence tolerance

### Requirement: Proposal and reuse layers are unbiased
Adding the neural directional proposal or ReSTIR DI direct-light reuse SHALL change
variance, not the converged expectation. The harness SHALL assert that a combo
carrying a proposal/reuse layer agrees with the anchor (which has neither) within
the unbiasedness tolerance.

#### Scenario: neural proposal does not bias the converged image
- **WHEN** a flat-material scene is rendered with `(Path, wavefront, +neural)` and
  with the anchor `(Path, wavefront)`
- **THEN** the exposure-aligned relMSE/FLIP between them is within the unbiasedness
  tolerance

#### Scenario: ReSTIR DI does not bias the converged image
- **WHEN** a scene with direct lighting is rendered with `(Path, wavefront, +ReSTIR DI)`
  and with the anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the unbiasedness
  tolerance

### Requirement: Every valid combo is gated against the pbrt reference
For each (scene × valid combo) the harness SHALL compute exposure-aligned
relMSE/FLIP versus the checked-in pbrt v4 reference EXR and SHALL pass when the
metrics are within the scene's pbrt-truth tolerance, or — when the combo declares a
recorded baseline — within that baseline (per the recorded-baseline requirement).

#### Scenario: a small synthetic scene matches pbrt across the matrix
- **WHEN** a corpus synthetic scene is rendered with every valid combo and a pbrt
  reference EXR exists
- **THEN** each combo's exposure-aligned relMSE/FLIP is within the scene's
  pbrt-truth tolerance

#### Scenario: the gate needs no pbrt binary at test time
- **WHEN** the parity gate runs on a host with no pbrt binary
- **THEN** it relies solely on the checked-in reference EXRs and still evaluates
  every combo

### Requirement: Heavy reference scenes are covered
The corpus SHALL include the bathroom scene (`assets/bathroom.usda`, imported from
pbrt `contemporary-bathroom`) and a dragon scene, each with a checked-in pbrt v4
reference EXR. A scene source SHALL be either a corpus `.pbrt` (imported at gate
time) or an existing `.usda` asset loaded directly.

#### Scenario: bathroom is swept by the matrix
- **WHEN** the matrix runs
- **THEN** `assets/bathroom.usda` is rendered for every valid combo and compared to
  its pbrt reference and to the self-consistency anchor

#### Scenario: dragon is wavefront-only due to geometry size
- **WHEN** the matrix is enumerated for the dragon scene
- **THEN** megakernel combos are skipped with reason "geometry exceeds megakernel
  budget" and wavefront combos are rendered

### Requirement: Known mismatches are recorded as regression baselines
The manifest SHALL carry a numeric `baseline` for any scene-and-combo pair that does
not yet meet its strict pbrt-truth tolerance. The pbrt-truth assertion SHALL then
require the measured metric to be no worse than the baseline plus a small explicit
margin, and SHALL log the delta to the baseline. For a normal (non-heavy) scene,
self-consistency assertions SHALL NOT use a per-combo baseline escape — those
invariants always use the strict tolerance.

A heavy scene MAY additionally be flagged `known_divergent` for a known,
not-yet-fixed defect (e.g. bathroom mismatching pbrt, or BDPT diverging from the
path anchor on it). When so flagged, the matrix gate SHALL record the measured
battery and `xfail` (visible, non-blocking) instead of hard-failing, so the suite
stays green while the deltas are pinned; the follow-up fix SHALL clear the flag.
A scene that is NOT `known_divergent` SHALL hard-fail on any gate violation.

#### Scenario: bathroom records its current pbrt delta and guards against regression
- **WHEN** bathroom is rendered and its pbrt-truth metric exceeds the strict
  tolerance but is at or below the recorded baseline
- **THEN** the test passes and logs the delta to the baseline

#### Scenario: a known-divergent heavy scene xfails rather than blocking
- **WHEN** a `known_divergent` heavy scene has a combo that fails either gate
- **THEN** the matrix gate records the measured battery and `xfail`s (the suite
  stays green and the divergence stays visible), not a hard failure

#### Scenario: a regression past the baseline fails
- **WHEN** a combo's measured pbrt-truth metric exceeds its recorded baseline by
  more than the margin
- **THEN** the test fails

#### Scenario: a mode/integrator divergence fails regardless of baseline
- **WHEN** two modes of the same integrator diverge beyond the mode-equivalence
  tolerance
- **THEN** the self-consistency assertion fails even though a pbrt-truth baseline
  exists for those combos

### Requirement: New renderer features are covered automatically
The harness SHALL include a coverage meta-test that fails if a renderer combination
the application exposes (any integrator in `renderer.integrator_modes` crossed with
each supported execution mode, and each declared proposal/reuse axis) has no entry
in the validity table. Adding a new integrator, mode, or proposal/reuse axis without
extending the matrix SHALL fail this meta-test.

#### Scenario: a new integrator without a matrix row fails the meta-test
- **WHEN** a new integrator is added to `renderer.integrator_modes` but no validity
  rows reference it
- **THEN** the coverage meta-test fails, naming the uncovered combos

#### Scenario: the meta-test runs without a GPU
- **WHEN** the coverage meta-test runs on a host with no GPU
- **THEN** it compares the validity table against the exposed combos without
  rendering

### Requirement: Hostless import tier validates the full matrix
The harness SHALL provide a `not gpu` tier that constructs the entire matrix and
imports/loads every corpus scene (including bathroom and dragon) without rendering,
so the matrix wiring and scene intake are exercised on any host.

#### Scenario: every scene imports cleanly with no GPU
- **WHEN** the `not gpu` tier runs
- **THEN** every corpus scene imports/loads with no unsupported feature and the
  matrix enumerates without error
