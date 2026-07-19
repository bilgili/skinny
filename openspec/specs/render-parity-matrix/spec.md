# render-parity-matrix Specification

## Purpose
TBD - created by archiving change integrator-parity-matrix. Update Purpose after archive.
## Requirements
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
validity table over the axes `integrator ∈ {Path, BDPT, SPPM, MLT}`, `execution_mode ∈
{megakernel, wavefront}`, `proposals ⊇ {neural}`, `reuse ⊇ {ReSTIR DI}`, and
`spectral ∈ {off, on}`. The table SHALL mirror the documented compatibility matrix. Every
(scene × combo) SHALL be either exercised or skipped with an explicit, machine-readable
reason; no valid combo SHALL be silently dropped. The spectral **envelope** SHALL admit
`path`/`bdpt` under either execution mode, `sppm` under the wavefront mode, and `mlt` under
the wavefront mode — all without proposal or reuse layers, on flat-material scenes without
subsurface/skin or heterogeneous-volume transport. An envelope-eligible spectral combo SHALL
enter the rendered set only once its transport is wired (the live capability gates —
`SPECTRAL_IMPLEMENTED`, and `MLT_IMPLEMENTED` for the `mlt` combo); while unwired it SHALL be
a recorded "not yet wired" skip and SHALL be absent from the rendered set, so the matrix never
renders a spectral combo as an ordinary RGB frame and gates it as if it were spectral.

#### Scenario: SPPM is wavefront-only
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(SPPM, megakernel)` is skipped with reason "SPPM is wavefront-only"
- **AND** `(SPPM, wavefront)` is present in the rendered set

#### Scenario: MLT is wavefront-only and layer-free
- **WHEN** the matrix is enumerated for any scene
- **THEN** `(MLT, megakernel)` is skipped with reason "MLT is wavefront-only"
- **AND** every MLT+neural / MLT+ReSTIR combo is skipped with a recorded reason
- **AND** `(MLT, wavefront)` and `(MLT, wavefront, spectral)` are present in
  the rendered set for flat-material scenes, while skin/subsurface/volume-
  dominated scenes record an out-of-envelope skip

#### Scenario: neural proposal requires wavefront and a flat material scene
- **WHEN** the matrix is enumerated for a subsurface/skin scene (e.g. the SSS dragon)
- **THEN** every combo carrying the neural proposal is skipped with reason
  "neural proposal is flat-material + wavefront only"
- **AND** for a flat-material scene, `(Path, wavefront, +neural)` is present

#### Scenario: BDPT ignores the neural proposal
- **WHEN** the matrix is enumerated
- **THEN** no `BDPT` combo carries the neural proposal (skipped by design)

#### Scenario: spectral envelope admits path/bdpt/sppm/mlt without layers
- **WHEN** the spectral envelope is evaluated for a flat-material scene
- **THEN** spectral+proposal and spectral+reuse combos are rejected with a
  recorded reason, `(SPPM, megakernel, spectral)` and `(MLT, megakernel,
  spectral)` are rejected as wavefront-only, and the envelope-eligible spectral
  combos are `path`/`bdpt` under either execution mode plus `sppm` and `mlt`
  under the wavefront mode

#### Scenario: spectral combos are gated until the transport is wired
- **WHEN** the matrix is enumerated while a spectral transport capability gate is off
- **THEN** the envelope-eligible spectral combos it guards are skipped with a
  "not yet wired" reason and are absent from the rendered set
- **AND** once the gate is on, those combos are present in the rendered set for a
  flat-material scene

#### Scenario: spectral skips volume and skin scenes
- **WHEN** the matrix is enumerated for a scene with heterogeneous media or skin/subsurface
  materials
- **THEN** every spectral combo is skipped with a recorded reason

### Requirement: megakernel and wavefront produce the same image
For any integrator that runs in both execution modes, the harness SHALL assert
that the megakernel and wavefront renders of the same scene agree. The
exposure-aligned relMSE/FLIP between the two modes SHALL be within the tight
mode-equivalence tolerance for that scene — **except** where the two modes are
specified to run different algorithms: the subsurface interior walk (change
`pbrt-subsurface-3d-walk`) is a true 3D per-segment walk under the wavefront
mode and a watchdog-safe 1D slab under the megakernel, so a
subsurface-dominated scene MAY record a per-scene `self_consistency` `mode`
override, measured and tighten-only, with the measured delta noted in the
manifest entry. The override SHALL NOT be used to mask an unexplained
divergence on flat or volume scenes.

#### Scenario: Path megakernel matches Path wavefront
- **WHEN** the same scene is rendered with `(Path, megakernel)` and `(Path, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

#### Scenario: BDPT megakernel matches BDPT wavefront
- **WHEN** the same scene is rendered with `(BDPT, megakernel)` and `(BDPT, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

#### Scenario: Recorded subsurface mode divergence

- **WHEN** `subsurface_infinite` renders `path|megakernel` against the
  `path|wavefront` anchor at the manifest spp
- **THEN** the self-consistency gate asserts against the scene's recorded
  `mode` override (relMSE ≤ 0.05, FLIP ≤ 0.07; measured 0.0362/0.0554 at
  512 spp), and the gate fails if the divergence grows beyond it

### Requirement: Integrators converge to a common golden image
The harness SHALL designate `(Path, wavefront, no-proposal, no-reuse)` as the
self-consistency anchor per scene and assert that every other valid integrator
combo agrees with the anchor within a per-integrator tolerance (looser for SPPM
caustics and BDPT connection noise than for a pure mode change). MLT SHALL gate
against the same anchor with its own recorded MLT-equivalence tolerance,
measured harness-first: Markov-chain samples are correlated, so at equal spp the
per-pixel noise structure differs from independent sampling even though the
means agree; the tolerance SHALL only ever be tightened, never loosened to hide
a real divergence.

#### Scenario: BDPT matches the Path anchor
- **WHEN** a scene is rendered with BDPT and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the
  integrator-equivalence tolerance

#### Scenario: SPPM matches the Path anchor within its caustic tolerance
- **WHEN** a scene is rendered with `(SPPM, wavefront)` and with the Path anchor
- **THEN** the exposure-aligned relMSE/FLIP between them is within the (looser)
  SPPM-equivalence tolerance

#### Scenario: MLT matches the Path anchor within its recorded tolerance
- **WHEN** a scene is rendered with `(MLT, wavefront)` and with the Path anchor
  at the manifest spp
- **THEN** the exposure-aligned relMSE/FLIP between them is within the recorded
  MLT-equivalence tolerance for that scene

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

### Requirement: Suite scenes under tests/assets are corpus scenes
The corpus manifest SHALL accept scene entries whose sources live under
`tests/assets/suite/` (USD assets and pbrt counterparts), registered in the single existing
manifest (`tests/pbrt/corpus/manifest.json`) with references in the single existing refs
directory. Suite scenes SHALL be swept by the same gates and combo enumeration as existing
corpus scenes — no parallel harness.

#### Scenario: a suite scene is swept like a corpus scene
- **WHEN** the parity matrix runs with a manifest entry pointing at
  `tests/assets/suite/<scene>/<scene>.usda`
- **THEN** the scene is rendered for every valid combo and evaluated by the pbrt-truth
  (when a ref exists) and self-consistency gates identically to a `tests/pbrt/corpus/` scene

### Requirement: Authoring-equivalence gate class
The harness SHALL support a third gate class beside pbrt-truth and self-consistency:
authoring equivalence, which renders a scene's plain-USD and MaterialX variants with the
anchor combo and asserts the standard metric battery between them is within the scene's
recorded equivalence tolerance. Scenes with only one authoring variant SHALL carry a
recorded skip reason for this gate.

#### Scenario: equivalent authorings agree
- **WHEN** a dual-authored scene's variants are both rendered with the anchor combo
- **THEN** the metric battery between the two images passes the recorded equivalence
  tolerance

#### Scenario: single-authoring scenes are recorded skips
- **WHEN** the equivalence gate enumerates a scene with a recorded single-authoring reason
- **THEN** the gate reports a skip with that reason instead of failing or passing silently

### Requirement: Coverage meta-test spans suite gate classes
The coverage meta-test SHALL fail the build when a suite scene lacks a disposition for any
applicable gate class: pbrt-truth (ref EXR or recorded skip reason), authoring equivalence
(tolerance or recorded skip reason), and — for scenes participating in furnace closure — a
furnace disposition (tolerance or recorded baseline).

#### Scenario: a scene missing a gate disposition fails the build
- **WHEN** a new suite scene is added with no pbrt ref and no recorded pbrt skip reason
- **THEN** the coverage meta-test fails on the hostless tier

### Requirement: Spectral combos gate against pbrt truth

Spectral combos SHALL be gated by pbrt-truth versus the checked-in (spectrally rendered)
pbrt v4 reference EXRs. On scenes whose pbrt divergence is attributable to RGB reduction, the
spectral combo's recorded pbrt-truth measurement SHALL be at or below the RGB combo's; no
tolerance or baseline SHALL be loosened to admit a spectral combo. With spectral transport
available in both execution modes, the mode-equivalence (mega≡wave) assertion SHALL apply to
spectral `path`/`bdpt` combos exactly as it does to their RGB counterparts — wavefront spectral
anchoring to the **megakernel spectral path** image within the recorded self-consistency
tolerance — and SHALL no longer be a blanket "spectral is megakernel-only" skip. The one
retained skip SHALL be spectral `bdpt` on an out-of-gamut dispersion (light-tracer splat)
scene, whose per-splat gamut clamp is nonlinear and differs by splat granularity between the
fused and staged pipelines; that skip SHALL record its reason. The spectral-versus-RGB anchor
delta SHALL still be reported, not asserted tight.

#### Scenario: spectral tightens a spectrum-authored scene

- **WHEN** a corpus scene authored with spectra (blackbody or sampled illuminant) is rendered
  with `(Path, megakernel, spectral)` and `(Path, megakernel, RGB)`
- **THEN** the spectral combo's exposure-aligned relMSE/FLIP against the pbrt reference is at
  or below the RGB combo's recorded measurement

#### Scenario: spectral mode-equivalence is asserted

- **WHEN** the self-consistency gate enumerates a spectral `path` or `bdpt` combo in the
  wavefront execution mode on an in-gamut (non-dispersion-splat) scene
- **THEN** its accumulated image is asserted equivalent to the megakernel spectral path image
  within the recorded tolerance, not skipped

#### Scenario: spectral BDPT dispersion splat stays a recorded skip

- **WHEN** the self-consistency gate enumerates spectral `bdpt` on an out-of-gamut dispersion
  (light-tracer splat) scene
- **THEN** the mega≡wave assertion records a skip naming the per-splat clamp granularity, rather
  than failing or asserting tight

### Requirement: Coverage meta-test spans the spectral axis

The coverage meta-test SHALL fail when any exposed integrator × spectral combination lacks a
validity entry, and the confirming suite SHALL include at least one spectral-discriminating
scene (dispersive dielectric and/or blackbody-lit) with dispositions for the pbrt-truth gate
in both spectral and RGB modes.

#### Scenario: missing spectral validity entry fails the build

- **WHEN** an integrator exposed by `renderer.integrator_modes` has no validity entry for
  `spectral = on`
- **THEN** the hostless coverage meta-test fails, naming the uncovered combo

#### Scenario: spectral discriminating scene is registered

- **WHEN** the suite coverage meta-test runs
- **THEN** at least one suite scene declares a spectral-discriminating disposition with pbrt
  references gated in both modes

### Requirement: Corpus scene data is integrity-checked hostlessly

The hostless matrix tier SHALL verify, without a GPU, that the corpus scene
data the GPU gates depend on is self-consistent: for every manifest scene whose
`usd` asset exists on disk, every `texture:file` reference authored in that
asset SHALL resolve to an existing file (relative references resolve against
the asset's directory); and every manifest scene whose `.pbrt` source authors a
non-flat material or a named medium (`Material "subsurface"`,
`MakeNamedMedium`) SHALL declare a non-flat `material_class`, and every
manifest scene whose on-disk `.usda` asset authors a volume field
(`OpenVDBAsset`) SHALL declare `material_class: "volume"`. A scene whose
`usd` asset or `.pbrt` source is absent on the current checkout (e.g. a
worktree without the untracked asset tree, or a `usd:`-sourced scene whose
`file` name is informational) SHALL be skipped for that check, not failed.

#### Scenario: Deleted baked-env side-file is caught hostlessly

- **WHEN** a manifest scene's `.usda` asset references a baked
  `light_infinite_*_const.hdr` that has been deleted from `assets/`
- **THEN** the hostless integrity meta-test fails naming the scene and the
  dangling reference, before any GPU gate renders the scene against the wrong
  (fallback) environment

#### Scenario: Non-flat scene missing material_class is caught hostlessly

- **WHEN** a corpus `.pbrt` scene source authors `Material "subsurface"` but its
  manifest entry declares no `material_class`
- **THEN** the hostless integrity meta-test fails naming the scene, before the
  spectral envelope admits a spectral combo that the renderer refuses at
  scene-build time

### Requirement: Missing dome-light textures fail loudly at load

The USD loader SHALL emit a visible warning (stderr) when a `DomeLight` authors
a `texture:file` that does not resolve to an existing file, identifying the
prim and the dangling path, while retaining the existing fallback behavior
(no DomeLight upload; the built-in environment library remains active).

#### Scenario: Dangling DomeLight texture warns

- **WHEN** a USD scene whose DomeLight references a missing `.hdr` is loaded
- **THEN** a warning naming the DomeLight prim and the unresolved path is
  printed, and the scene still renders under the built-in environment

### Requirement: Spectral wavefront combos are valid rendered combos

The validity table SHALL admit `(path, wavefront, spectral)`, `(bdpt, wavefront, spectral)`,
and `(sppm, wavefront, spectral)` into the rendered set (flat materials, BSDF proposal, no
reuse). Spectral SPPM SHALL be gated for self-consistency against the **spectral** path anchor
(the megakernel spectral path image), NOT the RGB golden — on a spectrum-authored scene
spectral and RGB differ by construction, so an RGB anchor would be invalid; if
spectral-path-anchoring is infeasible, spectral SPPM SHALL instead be gated by pbrt-truth only,
with no RGB-golden self-consistency claim. The self-consistency gate SHALL select the anchor
image per the spectral axis, so a spectral combo is never compared against an RGB anchor. The
remaining spectral rejections — the neural directional proposal, ReSTIR reuse, and
skin/subsurface/heterogeneous-volume scenes — SHALL stay recorded exclusions.

#### Scenario: spectral wavefront combos render and gate

- **WHEN** the parity matrix enumerates the spectral axis under the wavefront execution mode
- **THEN** the `path`, `bdpt`, and `sppm` integrators are admitted as rendered combos and gated
  against pbrt truth and the appropriate self-consistency anchor

#### Scenario: spectral SPPM validity entry exists

- **WHEN** the coverage meta-test enumerates `sppm` on the spectral axis
- **THEN** a validity entry exists (rendered under wavefront), so the build does not fail for a
  missing spectral SPPM combo

### Requirement: Spectral MLT combo is dual-gated against spectral anchors

The `(mlt, wavefront, spectral)` combo SHALL be gated by pbrt-truth against
spectral reference EXRs regenerated by the pinned pbrt v4 binary running
`Integrator "mlt"` (pbrt's MLT is spectral natively), and by self-consistency
against the **spectral** `(path, wavefront)` anchor — never an RGB anchor —
with its own recorded tolerance measured harness-first, allowing a per-scene
baseline where dispersion demands it (the BK7 prism scene). No RGB MLT
tolerance and no existing spectral tolerance SHALL be loosened to admit the
combo. Scenes authoring pbrt `maxcomponentvalue` SHALL NOT enter the
spectral-MLT gated set without a recorded note of the clamp asymmetry (the
path anchor clamps per sample; the MLT splat path never clamps, matching
pbrt's unclamped `AddSplat`).

#### Scenario: spectral MLT renders and gates in the suite sweep

- **WHEN** the GPU parity sweep enumerates suite scenes on the spectral axis
- **THEN** `(mlt, wavefront, spectral)` is rendered for flat-material scenes
  and asserted against both the spectral pbrt-mlt truth EXR and the spectral
  Path-wavefront anchor within recorded tolerances

#### Scenario: spectral MLT anchor is never RGB

- **WHEN** the self-consistency gate selects the anchor for
  `(mlt, wavefront, spectral)`
- **THEN** the anchor is the spectral `(path, wavefront)` image, selected per
  the spectral axis

