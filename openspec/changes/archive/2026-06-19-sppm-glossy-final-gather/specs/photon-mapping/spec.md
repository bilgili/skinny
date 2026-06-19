## ADDED Requirements

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
reproduce the prior delta-only behavior, and the roughness gating SHALL be the
only behavioral change — flat-only scope, wavefront-only execution, both-backend
parity, and the caustic-parity gate from PM-1 remain in force.

#### Scenario: Glossy metal reflects neighbouring objects under SPPM

- **WHEN** the three-materials demo (a polished-metal sphere beside diffuse/wood
  spheres) is rendered under SPPM with `sppmGlossyContinueRoughness` at its
  default
- **THEN** the metal sphere SHALL show the neighbouring spheres reflected in it
- **AND** the reflected content SHALL trend toward the path-traced reference as
  passes accumulate (not remain absent as with delta-only continuation)

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
