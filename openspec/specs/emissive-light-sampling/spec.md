# emissive-light-sampling Specification

## Purpose
Govern how the renderer selects emissive-triangle (mesh) area lights for
next-event estimation: every emissive triangle in the scene participates (no
fixed cap), selection is power-weighted (∝ area × luminance) via a
cumulative-power CDF, the estimator stays unbiased and MIS-consistent, and
ReSTIR DI draws through the same distribution.
## Requirements
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

### Requirement: BDPT eye-side NEE inherits the power-weighted emissive distribution

BDPT eye-side next-event estimation to emissive triangles SHALL select the
emissive triangle through the same power-weighted cumulative-power CDF as the
unidirectional path tracer (`sampleEmissiveTriangle`, selection probability
`p_i = w_i / Σw`), so the draw matches the `pdfArea = p_i / triArea` reported by
`samplePoint`. This applies to `connectT1` in `bdpt.slang`, shared by the
megakernel and wavefront backends. Selecting uniformly by index while dividing by
the power-weighted pdf is biased and SHALL NOT be used.

#### Scenario: Many-triangle emissive room matches the path tracer
- **WHEN** a scene whose area lights are many emissive triangles of uneven power (e.g. a pbrt-imported room with ~1.5k emissive triangles) is rendered with BDPT and with the path tracer in the same lighting configuration
- **THEN** BDPT's mean linear-HDR energy tracks the path tracer's (no systematic darkening of the indirect emissive fill), rather than reading far darker as it did under uniform-by-index selection

#### Scenario: Small area light is unchanged
- **WHEN** the `diffuse_arealight` corpus scene (a small quad emitter) is rendered with BDPT after the change
- **THEN** its exposure-aligned parity versus the pbrt reference stays within the corpus tolerance, and the BDPT absolute-energy and emissive-NEE gates do not regress

#### Scenario: Estimator stays unbiased
- **WHEN** the many-triangle emissive scene is rendered to convergence
- **THEN** BDPT's expected image matches the path tracer's within Monte-Carlo noise (the change alters the selection distribution and its matching pdf, not the integrand)

### Requirement: BSDF-sampling MIS complement for emissive-triangle hits

The path tracer SHALL combine the two sampling strategies for emissive-triangle
area lights — next-event estimation and BSDF sampling — with multiple importance
sampling, so the emissive-triangle direct-lighting estimator is unbiased
regardless of the light's solid angle.

Because `allLightsNEE` weights the emissive-triangle NEE sample with the power
heuristic (`wNEE < 1`), the complementary BSDF-sampling strategy SHALL contribute
the remaining weight: when a non-delta bounce ray lands on an emissive triangle,
the renderer SHALL add `throughput · emission · wBSDF` with
`wBSDF = powerHeuristic(bsdfPdf, pdfLightSA)`, where `bsdfPdf` is the solid-angle
pdf of the bounce that spawned the ray and `pdfLightSA` is the solid-angle pdf the
NEE strategy would have used to sample that same hit. The renderer SHALL NOT drop
this BSDF-hit emission (the previous behaviour, which biased area lights dim by a
solid-angle-dependent factor), and SHALL NOT add it at full weight (which would
double-count against NEE).

The full-weight branches SHALL be preserved for the cases that have no NEE
partner: the primary (camera) ray, a ray spawned by a delta/perfectly-specular
lobe, and a scene with no emissive triangles. These take the emission at weight 1.

`pdfLightSA` SHALL be reconstructed at the hit **without the emissive-triangle
buffer index**. Under power-weighted selection (`p_i = area_i·lum_i / ΣW`) with
uniform-area sampling within the chosen triangle, the per-point area-measure pdf
is `lum_i / ΣW` (the per-triangle area cancels), so the solid-angle pdf is
`pdfLightSA = Rec709-lum(emission) · d² / (emissiveTotalPower · cosLight)`, where
`d²` is the squared shading-point-to-hit distance, `cosLight` is the cosine at the
light hit, and `emissiveTotalPower = ΣW = Σ(area_i · Rec709-lum(emission_i))` is
supplied as a single `FrameConstants` scalar. A grazing or degenerate hit
(`cosLight → 0`) SHALL drive `wBSDF → 0` so NEE owns that sample (no double-count).

The same estimator SHALL be applied in the megakernel and wavefront path tracers
so that `megakernel ≡ wavefront` for the Path integrator.

#### Scenario: a large area light is no longer dim

- **WHEN** a diffuse surface is lit by a large-solid-angle emissive-triangle area
  light and rendered with the path tracer
- **THEN** the converged luminance matches the reference (no `(1 − wNEE)` dim
  bias), and the skinny-vs-pbrt luminance ratio is independent of the light's
  size (a large light no longer scores a larger dim ratio than a small one)

#### Scenario: small area lights are unchanged

- **WHEN** the emissive triangle subtends a small solid angle (`pdfLightSA ≫
  bsdfPdf`, so `wNEE ≈ 1`)
- **THEN** the BSDF-hit complement contributes ≈ 0 and the estimate is
  statistically unchanged from NEE alone

#### Scenario: megakernel and wavefront agree

- **WHEN** the same emissive-triangle scene is rendered with the path tracer in
  megakernel and in wavefront execution mode
- **THEN** the two converge to the same image (self-consistency relMSE ≈ 0), i.e.
  both apply the BSDF-hit MIS complement identically

