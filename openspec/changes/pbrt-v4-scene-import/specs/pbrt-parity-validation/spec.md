## ADDED Requirements

### Requirement: Reference scene corpus

The change SHALL ship a curated corpus of small, low-resolution (≤256 px) pbrt v4
scenes under `tests/pbrt/`, each isolating one parity axis (e.g. diffuse under an
area light, conductor under an environment map, glass under an area light, a
coated material, a homogeneous-medium/subsurface case, an instanced grid, a
thick-lens depth-of-field case). Each scene SHALL have a committed pbrt v4
reference image (linear EXR) and a manifest entry pinning the pbrt version, the
per-scene error tolerance, and a content hash. Corpus assets SHALL be
force-added because the repo `.gitignore` is `*`.

#### Scenario: Every corpus scene has a pinned reference
- **WHEN** the corpus is enumerated
- **THEN** each `.pbrt` scene has a matching reference EXR and a manifest entry with pbrt version, tolerance, and hash

### Requirement: Headless comparison harness

A headless harness SHALL import each corpus scene to USD, render it in skinny via
the offscreen path, read skinny's **linear-HDR accumulation** (not the tonemapped
sRGB display output), and compare it to the reference EXR at matching resolution
and converged sample count.

#### Scenario: Imported scene renders and compares
- **WHEN** the harness runs a corpus scene
- **THEN** it produces a skinny linear-HDR image and a reference linear image of identical dimensions ready for metric evaluation

#### Scenario: Comparison uses linear data
- **WHEN** skinny output is read for comparison
- **THEN** the harness uses the linear accumulation image, never the tonemapped/sRGB `render_headless` pixels

### Requirement: Error metrics

The harness SHALL compute **relMSE** (`mean((a-b)²/(b²+ε))`) on linear data and a
**FLIP** perceptual score on identically-tonemapped copies of both images, and
SHALL report both per scene. A single global imaging scalar MAY be applied to
align absolute exposure before relMSE.

#### Scenario: Both metrics are reported
- **WHEN** a corpus scene is evaluated
- **THEN** the harness emits a relMSE value and a FLIP value for that scene

#### Scenario: Identical images score zero
- **WHEN** an image is compared against itself
- **THEN** relMSE and FLIP are both zero (within floating-point epsilon)

### Requirement: Parity gate

Each corpus scene SHALL pass its per-scene tolerance; a scene exceeding tolerance
SHALL fail the test suite. The gate SHALL run without a pbrt binary present,
relying solely on the checked-in reference EXRs.

#### Scenario: Within-tolerance scene passes
- **WHEN** skinny's image is within the manifest tolerance of the reference
- **THEN** the parity test for that scene passes

#### Scenario: Regression fails the gate
- **WHEN** a change pushes a scene's relMSE/FLIP above its tolerance
- **THEN** the parity test fails and names the offending scene and metric

#### Scenario: No pbrt binary required
- **WHEN** the parity suite runs in an environment with no pbrt installed
- **THEN** it still executes using the committed reference EXRs

### Requirement: Parity matrix documentation

The change SHALL maintain a parity matrix (in `docs/PbrtImport.md`) listing each
supported pbrt feature as matched, approximated, or unsupported, with the measured
per-scene error for corpus coverage. The matrix SHALL be updated whenever feature
coverage or measured error changes.

#### Scenario: Feature coverage is documented with measured error
- **WHEN** a reader consults the parity matrix
- **THEN** they see, per pbrt feature, its support status and the latest measured relMSE/FLIP for the covering corpus scene

### Requirement: Divergence transparency

Features that are approximated or unsupported SHALL be surfaced consistently in
both the per-import translation report and the parity matrix; the system SHALL
NOT present an approximated render as an exact match.

#### Scenario: Approximations are not labeled exact
- **WHEN** a scene relies on an approximated feature (e.g. a spot light or spectral-heavy material)
- **THEN** that approximation appears in the translation report and the parity matrix, and its scene tolerance reflects the expected divergence
