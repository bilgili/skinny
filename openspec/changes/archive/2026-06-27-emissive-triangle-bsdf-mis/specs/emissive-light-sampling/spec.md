## ADDED Requirements

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
