# photon-mapping — delta

## ADDED Requirements

### Requirement: Photon emission group selection is power-proportional

The SPPM photon pass SHALL select the photon-emission group (emissive triangles,
sphere lights, distant lights, environment) with probability proportional to each
group's total emitted power, computed host-side and uploaded per frame:
emissive `π·Σ(area·luminance)`, sphere `4π²·Σ(luminance·r²)`, distant
`πR²·Σ luminance`, environment `πR²·envIntensity·∫L dω` (sin θ-weighted luminance
integral of the equirect map), with `R` the scene bounding-sphere radius used by the
emission geometry. Each emission branch SHALL divide its flux by the **actual**
selection probability (replacing the uniform `1/G`), so the estimator stays unbiased
while per-photon flux equalises across groups (`Φ_g / p_g ≈ Φ_total`). A group SHALL
receive zero probability when absent under the same presence predicates the shader
uses (`count > 0`; env: `furnaceMode == 0 && envIntensity > 0`). When the total
power is zero or non-finite the pmf SHALL fall back to uniform over the present
groups. Selection SHALL be wavelength-independent (the spectral build shares the
identical scalar pmf).

#### Scenario: Mixed weak-local-light + environment scene loses the env fireflies

- **WHEN** `glass_caustics_test.usda` (sphere light r = 0.2 + environment) is
  rendered by SPPM at matched spp against the path anchor, before and after the
  change
- **THEN** whole-image `noise_sigma` (from `metrics.compute_metrics`) SHALL drop
  versus the uniform-selection baseline (measured −11%: 4.66e-4 → 4.14e-4 at
  384²/48 spp; the group power ratio on this scene is ~5.4×, so the equalisation
  win is bounded), **AND** the caustic-region `noise_sigma` (masked to the
  caustic footprint — the whole-image number is background-dominated and blind
  to a caustic regression) SHALL NOT regress versus the uniform-selection
  baseline (measured −11%: 1.06e-3 → 9.4e-4), and the median-of-ratio SPPM/path
  SHALL remain within the recorded unbiasedness tolerance of 1.0 (measured:
  median 1.0000, mean 1.0120 vs the uniform baseline's 1.0118)

#### Scenario: Single-group scene is unchanged

- **WHEN** a scene with exactly one present emission group is rendered
- **THEN** that group's selection probability is 1 and the per-photon flux matches
  the pre-change value (selection pmf cancels identically)

#### Scenario: Probe override forces a group

- **WHEN** the host probe override (`_sppm_group_pmf_override`) is set to
  `[0, 0, 0, 1]`
- **THEN** the packed pmf is exactly that value, preserving the existing
  forced-env flux-normalization probe under the new selection path

#### Scenario: Zero total power falls back to uniform

- **WHEN** the host pmf helper receives all-zero (or non-finite) group powers with
  at least one present group
- **THEN** it returns the uniform distribution over the present groups and zeros
  for absent groups
