## ADDED Requirements

### Requirement: Every emissive triangle is sampleable (no fixed cap)

The renderer SHALL include every emissive triangle in the scene in the
emissive-light NEE distribution, with no fixed upper bound that silently drops
triangles. The emissive-triangle and its CDF buffers SHALL be sized to the actual
emissive-triangle count (growing and rebinding as needed), and the count SHALL be
logged.

#### Scenario: High-poly emissive mesh keeps all its energy
- **WHEN** an emissive mesh tessellated into more than the previous 256-triangle cap is rendered
- **THEN** all of its triangles contribute, the scene is not biased dark, and the converged energy matches an equivalent low-poly emitter (within noise)

#### Scenario: Emissive-triangle count is reported
- **WHEN** a scene with emissive meshes is loaded
- **THEN** the renderer logs the total emissive-triangle count (so a large emitter is visible, never silently clipped)

### Requirement: Power-weighted emissive-triangle selection

The renderer SHALL select an emissive triangle for next-event estimation with
probability proportional to its power, `w_i = area_i × luminance(emission_i)`
(Rec.709 luminance), via a cumulative-power CDF and a binary-search upper-bound
(mirroring the environment importance-sampling distribution), rather than
uniformly by index. The per-triangle selection probability SHALL be
`p_i = w_i / Σw`.

#### Scenario: Bright triangles are sampled more often
- **WHEN** a scene mixes a bright emissive mesh with dim stray emissive triangles
- **THEN** NEE samples the bright triangles in proportion to their power, and equal-spp variance is materially lower than uniform-by-index selection

#### Scenario: Degenerate zero-power scene is safe
- **WHEN** no emissive triangle has positive power (Σw = 0)
- **THEN** the emissive-light NEE path is skipped (the `numEmissiveTriangles == 0` early-out behaviour) with no division by zero or invalid CDF

### Requirement: Power-weighted selection is unbiased and MIS-consistent

The power-weighted selection SHALL remain an unbiased NEE estimator: the area pdf
SHALL be `pdfArea = p_i / triArea_i`, the derived solid-angle pdf and the MIS
power-heuristic companion SHALL use that same pdf, and the BSDF-hit emission
accounting SHALL be unchanged (no double-counting).

#### Scenario: Small area light is unchanged
- **WHEN** the `diffuse_arealight` corpus scene (a small quad emitter) is rendered after the change
- **THEN** its exposure-aligned relMSE / FLIP versus the pbrt reference stay within the corpus parity tolerance (no regression)

#### Scenario: Converged image is unbiased
- **WHEN** a multi-emitter scene is rendered to convergence with the power-weighted distribution
- **THEN** the expected image matches the uniform-selection result (same mean energy within noise) — only the variance differs

### Requirement: ReSTIR inherits the emissive distribution

ReSTIR DI's emissive-triangle light candidates SHALL use the power-weighted
distribution: the candidate index draw SHALL go through the same shared
power-CDF selection helper as NEE so the actual draw matches the `selectionPdf`
reported through `light.samplePoint` (a uniform draw against a power pdf would
bias the RIS estimate), while the ReSTIR reservoir / RIS / GRIS reuse code SHALL
be unchanged. The distribution SHALL also apply at secondary bounces, which
ReSTIR (primary-hit only) does not cover.

#### Scenario: ReSTIR primary-hit candidates use power weighting
- **WHEN** ReSTIR DI is active and generates emissive-triangle light candidates at the primary hit
- **THEN** those candidates are selected with the power-weighted probability `p_i`, not uniform-by-index, via the shared selection helper, with no change to the ReSTIR reservoir/RIS code
