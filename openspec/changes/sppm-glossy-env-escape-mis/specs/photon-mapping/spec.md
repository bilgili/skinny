## MODIFIED Requirements

### Requirement: SPPM reconstructs glossy / near-specular reflectors

SPPM SHALL reconstruct sharp inter-object reflections on glossy, near-specular
flat reflectors (e.g. polished metals) rather than losing them to the photon
gather. The eye-walk continue-vs-store decision SHALL treat a sampled lobe whose
roughness is below a configurable threshold (`sppmGlossyContinueRoughness`) as a
caustic carrier: the walk SHALL follow the BSDF-sampled direction one bounce and
store the visible point at the next surface that is not itself glossy-continued,
so the reflection is reconstructed at the reflected surface and accumulated
across progressive passes. A glossy-continued vertex SHALL be treated like a
specular vertex by the photon stage (no photon deposit at it), preserving the
disjoint direct (NEE) / indirect (photon) split. A threshold of `0` SHALL
reproduce the prior delta-only behavior.

The default threshold SHALL be expressed in perceptual (USD) roughness and SHALL
reach pbrt-imported polished metals: pbrt perceptual roughness `r` imports as
`usd = r**0.25`, so the default SHALL be high enough that a polished
pbrt-roughness-`0.1` conductor (usd `≈ 0.562`) is glossy-continued while a
pbrt-roughness-`0.3` metal (usd `≈ 0.740`) remains on the photon-gather side.

A glossy-continued (non-delta) carrier vertex runs both env and emissive-triangle
NEE, so any light its carrier ray subsequently reaches SHALL be MIS-weighted
against that NEE (never taken at full weight, which would double-count):

- On an **environment escape**, the escaped-env radiance SHALL be weighted by
  `powerHeuristic(bsdfPdf, envPdf(dir))`.
- On an **emissive-triangle hit**, the emission SHALL be weighted by
  `powerHeuristic(bsdfPdf, pdfLightSA)`, where `pdfLightSA` is the NEE solid-angle
  pdf reconstructed without the triangle index (`lum·d² / (emissiveTotalPower·cosLight)`),
  matching the path tracer's emissive-hit MIS.

Only a **perfectly-specular (delta)** carrier (`bs.pdf <= 0`) has no NEE partner
and SHALL take the reached light at full weight; transmitted lobes and furnace
mode SHALL suppress the env companion. (Prior to this change the eye walk treated
every continued vertex — glossy included — as delta for the emissive-hit gate,
which double-counted an emitter seen in a glossy metal; the higher default
threshold makes that path reachable, so it is corrected here.) This makes a glossy
metal reflecting an environment or an emitter converge to the path-traced reference.

The roughness gating and the reached-light MIS SHALL be the only behavioral
changes — flat-only scope, wavefront-only execution, both-backend parity, and the
caustic-parity gate from PM-1 remain in force.

#### Scenario: Glossy metal reflects neighbouring objects under SPPM

- **WHEN** the three-materials demo (a polished-metal sphere beside diffuse/wood
  spheres) is rendered under SPPM with `sppmGlossyContinueRoughness` at its
  default
- **THEN** the metal sphere SHALL show the neighbouring spheres reflected in it
- **AND** the reflected content SHALL trend toward the path-traced reference as
  passes accumulate (not remain absent as with delta-only continuation)

#### Scenario: Polished pbrt metal under an environment converges to path

- **WHEN** a polished pbrt-roughness-`0.1` conductor sphere lit only by a constant
  infinite light (`conductor_infinite`) is rendered under SPPM
- **THEN** the sphere SHALL be glossy-continued (its usd roughness `≈ 0.562` is
  below the default threshold), so its env reflection is reconstructed via the
  carrier ray rather than a degenerate photon gather (no photon deposits on the
  sole metal surface)
- **AND** the escaped-env radiance SHALL be MIS-weighted against the vertex's env
  NEE, so the `sppm|wavefront` render matches the `(path, wavefront)`
  self-consistency anchor within tolerance

#### Scenario: Threshold of zero preserves PM-1 behavior

- **WHEN** `sppmGlossyContinueRoughness` is `0`
- **THEN** the eye walk SHALL continue only through perfectly-specular (delta)
  lobes, reproducing the PM-1 visible-point placement and the existing caustic
  parity result

#### Scenario: Direct lighting is still not double-counted

- **WHEN** a scene is rendered under SPPM with glossy continuation enabled
- **THEN** photons SHALL still deposit only at non-specular, non-glossy-continued
  vertices after at least one bounce
- **AND** the SPPM-vs-path energy ratio SHALL remain within the PM-1 tolerance
  band (no double-counted direct term)
