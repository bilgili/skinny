# photon-mapping (delta)

## ADDED Requirements

### Requirement: Update stage applies eye throughput at resolve

The update stage SHALL apply the visible point's stored eye throughput
(`VisiblePoint.beta` — camera-to-visible-point path throughput excluding the
visible point's BSDF) to the per-pass photon flux when resolving the pixel's
indirect radiance estimate, i.e. `L_indirect = (beta ⊗ Φ) / (N_emitted · π · r²)`,
matching pbrt-v4's fold of `vp.beta` at pass end. In spectral mode the product
SHALL be per-λ at the shared pass wavelengths used by both the eye and photon
stages, applied before the λ→sRGB resolve. The radius/N advance
(`sppmUpdate`) is flux-independent and unchanged.

#### Scenario: Directly viewed visible point is unchanged

- **WHEN** the visible point is reached directly from the camera (throughput ≈ 1)
- **THEN** the resolved SPPM estimate is unchanged within accumulation noise
  relative to the pre-change renderer

#### Scenario: Photon term through a tinted/lossy eye chain is not over-counted

- **WHEN** the photon-map indirect term is observed through a tinted or lossy
  specular/glossy eye chain (throughput ≠ 1, e.g. tinted glass)
- **THEN** the photon-indirect term is scaled by that throughput exactly as the
  NEE direct term already is (no over-bright, un-tinted photon contribution),
  and the SPPM render agrees with the BDPT reference on that region within the
  parity tolerance
