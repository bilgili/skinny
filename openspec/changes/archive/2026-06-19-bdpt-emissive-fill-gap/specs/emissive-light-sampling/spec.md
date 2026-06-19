## ADDED Requirements

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
