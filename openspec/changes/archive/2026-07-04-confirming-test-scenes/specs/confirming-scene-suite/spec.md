# confirming-scene-suite

## ADDED Requirements

### Requirement: Per-axis discriminating scene inventory
The repository SHALL provide a suite of minimal scenes under `tests/assets/suite/`, each
designed to discriminate one renderer axis: material lobe families (diffuse, conductor
smooth+rough, dielectric, plastic, emissive, textured, subsurface), integrator behavior
(indirect-dominant box, specular caustic, multi-bounce color bleed), and sampling modes
(many-small-lights for ReSTIR DI, env-dominant glossy for proposal mixtures and the neural
proposal). Every suite scene SHALL render at 128×128 within its manifest-recorded spp and
SHALL contain at most ~1k triangles with no external model-file dependencies.

#### Scenario: a material lobe regression is localized
- **WHEN** a shader change breaks one BSDF lobe (e.g. dielectric transmission)
- **THEN** the corresponding single-lobe scene's gate fails while unrelated lobe scenes
  still pass, localizing the regression to that lobe

#### Scenario: scenes stay inside the speed budget
- **WHEN** any suite scene is rendered headless at its manifest spp on the host backend
- **THEN** the render completes within seconds and under the metal-dispatch-hygiene
  per-dispatch bounds

### Requirement: Dual USD and MaterialX authoring per scene
Every suite scene SHALL be authored twice: a plain-USD variant (`<scene>.usda`,
`UsdPreviewSurface` materials) and a MaterialX variant (`<scene>_mtlx.usda` referencing
`<scene>.mtlx`). When a material is not expressible as `UsdPreviewSurface` (OpenPBR or
standard_surface-only materials), the plain variant SHALL be a recorded skip with a reason
string, never a silent absence.

#### Scenario: both variants exist and load
- **WHEN** the hostless suite tier runs
- **THEN** for every scene both variants (or one variant plus a recorded skip) open via the
  USD loader and expose the expected material prims

#### Scenario: authoring variants agree
- **WHEN** both variants of a scene are rendered with the anchor combo
- **THEN** the standard metric battery between the two images is within the scene's recorded
  equivalence tolerance

### Requirement: Scenes using existing PBR materials from assets
The suite SHALL include mini-shaderball scenes (sphere on plane) for a selected subset of at
least six physically-based OpenPBR materials originating from
`assets/materialxusd/tests/physically_based/` spanning conductor, dielectric, coated,
subsurface, and radiometric-extreme behavior (e.g. Gold, Copper, Glass, Plastic PC, Skin I,
Gray Card, Musou Black). Each scene SHALL carry the material as a single extracted,
self-contained prim (no embedded material library, no shaderball/HDR payloads) with its
source file recorded, so the suite has no test-time dependency on the `assets/` directory.

#### Scenario: PBR material scenes run in a worktree
- **WHEN** the suite runs in a fresh git worktree that has no `assets/` directory
- **THEN** the PBR material scenes load and render from `tests/assets/` alone

#### Scenario: PBR materials are swept
- **WHEN** the parity matrix runs over the suite
- **THEN** each PBR material scene is rendered by every valid combo and gated by
  self-consistency against the anchor

#### Scenario: original shaderball authoring is smoke-tested
- **WHEN** the opt-in gpu smoke test runs on a checkout where
  `assets/materialxusd/tests/physically_based/` exists
- **THEN** at least one original `*_OPBR_MAT_PBM.usda` shaderball scene renders as-is, and
  on checkouts without the directory the test skips with a recorded reason

### Requirement: pbrt counterpart per expressible scene
Every pbrt-expressible suite scene SHALL ship a `.pbrt` counterpart in its scene directory —
expressible meaning its materials and lights map onto pbrt v4 primitives (`diffuse`,
`conductor`, `dielectric`, `coateddiffuse`, `subsurface`, `imagemap`, area/infinite
lights) — that renders unmodified with the pinned pbrt v4 build, plus a checked-in
reference EXR generated from it. Scenes not
expressible in pbrt SHALL record a skip reason and be covered by the self-consistency and
equivalence gates only.

#### Scenario: user runs the pbrt counterpart directly
- **WHEN** the user invokes the pinned pbrt binary on `<scene>.pbrt`
- **THEN** pbrt renders it without edits and the output corresponds to the checked-in
  reference EXR

#### Scenario: pbrt-truth gates the skinny render
- **WHEN** a suite scene with a reference EXR is rendered by a valid combo
- **THEN** the exposure-aligned metric battery versus the reference is within the scene's
  pbrt-truth tolerance or its recorded baseline

### Requirement: Hostless validity tier for the suite
The suite SHALL include hostless (no-GPU) tests verifying, for every scene: the USD variants
open with expected prims and material bindings, the `.mtlx` parses, the `.pbrt` counterpart
imports through the pbrt importer, and the manifest entry is schema-valid.

#### Scenario: broken scene file fails without a GPU
- **WHEN** a scene file is malformed or a material binding is dropped
- **THEN** the hostless tier fails on plain `pytest` with no GPU or reference EXR needed

### Requirement: Staged regression checkpoints with user-gated GPU runs
Each implementation stage of suite work SHALL end with the hostless tier run automatically
and a GPU regression sweep that is proposed to the user — with exact commands and estimated
time — and executed only on explicit approval. Sweep results SHALL be persisted per stage and
reported as before/after metric deltas against the previous stage.

#### Scenario: stage ends without auto-running the GPU
- **WHEN** an implementation stage completes its hostless tier
- **THEN** the GPU sweep is offered but not executed until the user approves

#### Scenario: regressions are visible per stage
- **WHEN** an approved sweep completes
- **THEN** the report shows per-scene metric deltas versus the prior stage, flagging any
  regression beyond tolerance
