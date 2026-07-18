## ADDED Requirements

### Requirement: BDPT eye-subpath emitter-hit MIS partition

BDPT SHALL account for its eye-subpath emitter-hit strategy (`t = 0` — an eye
random walk that terminates on an emissive triangle) with multiple importance
sampling, using the **same MIS partition** that BDPT's other strategies for that
path already use (`connectT1`'s `t = 1` NEE, the `t ≥ 2` generic connections, and
the `s = 1` light-tracer splat), so that every strategy which can generate the
path shares one power-heuristic partition summing to 1. This is the BDPT analogue
of the path tracer's "BSDF-sampling MIS complement for emissive-triangle hits"
requirement.

When a non-delta eye bounce lands on an emissive triangle at vertex
`z = eye[s-1]` and a NEE partner exists, BDPT SHALL add
`throughput · emission · w` with `w = misWeight(eye, s, ·, t = 0, …)` rather than
dropping the emission (the previous behaviour, which discarded the BSDF-sampling
strategy's share and biased area lights dim) and rather than adding it at full
weight (which double-counts against NEE). BDPT SHALL NOT weight this term with a
standalone 2-strategy `powerHeuristic(bsdfPdf, pdfLightSA)`: that heuristic
ignores the `t ≥ 2` and `s = 1` alternatives (the reason it was already replaced
for `connectT1`), so the resulting partition would not sum to 1.

The two reverse pdfs the partition needs at the emitter end SHALL be reconstructed
**without the emissive-triangle buffer index**, matching how `connectT1` and the
path tracer reconstruct them:

- `z.pdfRev` (area measure) — the pdf a light-sampling strategy would assign to
  `z`'s position. Under power-weighted selection (`p_i = area_i·lum_i / ΣW`) with
  uniform-area sampling within the triangle, the per-triangle area cancels, so
  `z.pdfRev = Rec709-lum(z.emission) / emissiveTotalPower`, where
  `emissiveTotalPower = ΣW = Σ(area_i · Rec709-lum(emission_i))` is the existing
  `FrameConstants` scalar.
- `eye[s-2].pdfRev` (area measure) — the emitter's diffuse directional emission
  pdf (`cosOut / π`, `cosOut` the cosine at `z` toward `eye[s-2]`) converted to
  area measure at `eye[s-2]`.

The full-weight branches SHALL be preserved for the cases with no NEE partner:
the primary/first eye hit (`s == 2`), a ray spawned by a delta/perfectly-specular
bounce into `z` (`eye[s-2].isDelta`), and a scene with no emissive triangles
(`numEmissiveTriangles == 0`). These take the emission at weight 1.

The same estimator SHALL be applied identically in the BDPT megakernel and
wavefront transports (RGB `megakernel ≡ wavefront` preserved), and the spectral
BDPT transport SHALL apply it over its per-wavelength emission using the same
scalar MIS weight.

#### Scenario: one-bounce area-light emission is no longer dim

- **WHEN** a diffuse surface lit by an emissive-triangle area light is rendered
  with BDPT and with the path tracer in the same configuration (e.g. the
  `mat_emissive` suite scene)
- **THEN** BDPT's converged mean linear-HDR energy matches the path tracer's (the
  ~3% dim bias is removed) and BDPT's pbrt-truth relMSE moves down toward the path
  anchor, never up

#### Scenario: small area lights are unchanged

- **WHEN** the emitter subtends a small solid angle, so the eye-hit BSDF pdf far
  exceeds the reconstructed light pdf (`wNEE ≈ 1`)
- **THEN** the `t = 0` MIS weight contributes ≈ 0 and the estimate is
  statistically unchanged from the NEE-dominated result

#### Scenario: megakernel, wavefront, and spectral agree

- **WHEN** the same emissive-triangle scene is rendered with BDPT in megakernel
  and wavefront execution modes (RGB) and with `--spectral`
- **THEN** RGB megakernel and wavefront converge to the same image
  (self-consistency relMSE ≈ 0), and the spectral BDPT tracks its RGB sibling on
  a neutral emitter (within the recorded spectral self-consistency floor)

#### Scenario: estimator stays unbiased

- **WHEN** the scene is rendered to convergence
- **THEN** BDPT's expected image is unchanged in mean (the change reweights an
  existing strategy across the partition; it does not alter the integrand), only
  the direct/one-bounce emission that was previously discarded is restored
