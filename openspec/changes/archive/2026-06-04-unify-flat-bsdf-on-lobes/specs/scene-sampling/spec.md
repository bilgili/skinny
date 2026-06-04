## ADDED Requirements

### Requirement: BSDF proposal density is self-consistent across draw and weight

The BSDF directional proposal SHALL draw directions and report densities from a
**single** BSDF model, so that the density used to **draw** a bounce (the
material's `sample()`) equals the density used to **weight** it in the mixture
pdf and in NEE's companion pdf (the material's `evaluate()`), for **all**
materials including layered / coated ones. The proposal mixture's unbiasedness
depends on this equality; a material whose `sample()` and `evaluate()` disagree
SHALL be treated as a defect, not an accepted approximation.

#### Scenario: layered material stays unbiased under the mixture

- **WHEN** the `{bsdf, env}` proposal renders a layered coat+metal material (e.g.
  brass) IBL-only
- **THEN** the converged image matches the bsdf-only and BDPT references with no
  bias, because the drawing density and the weighting density come from the same
  model

#### Scenario: draw-time and weight-time pdf agree

- **WHEN** a non-delta bounce direction `wi` is produced for any flat /
  `std_surface` material
- **THEN** the BSDF proposal's draw-time pdf (`sample().pdf`) and weight-time pdf
  (`evaluate().pdf`) for `(wo, wi)` are equal
