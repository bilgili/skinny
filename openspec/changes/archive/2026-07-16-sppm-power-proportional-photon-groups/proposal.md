# SPPM power-proportional photon group selection

## Why

Change `sppm-env-indirect-transport` (59fbc89) added the environment as a 4th SPPM
photon-emission group, selected **uniformly** among present groups (`gsel = 1/G`).
The estimator is unbiased (means match the path anchor) but per-photon flux is wildly
unequal across groups: an env photon carries `beta = L·πR²/(gsel·pdf)` — with the
scene-bbox disc `πR² ≈ 85+` on `glass_caustics_test.usda` — while a sphere-light
photon (r = 0.2) carries orders of magnitude less. Env deposits are therefore sparse,
huge splats → heavy firefly speckle on any scene mixing a weak local light with an
environment (~1.7× noise vs the path tracer at matched spp).

## What Changes

- `sppmEmitPhoton` (`src/skinny/shaders/integrators/wavefront_sppm.slang`) selects the
  photon-emission group **proportionally to each group's emitted power** (pbrt-style
  light power distribution) instead of uniformly, and divides each branch's flux by the
  **actual** selection probability — estimator stays unbiased, per-photon flux
  equalises across groups (`Φ_g / p_g ≈ Φ_total` for every group).
- Host (`renderer.py`) computes the four group powers from data it already owns —
  emissive `π·Σ(area·lum)` (the existing `emissiveTotalPower` sum), sphere
  `4π²·Σ(lum·r²)`, distant `πR²·Σlum`, env `πR²·envIntensity·∫L dω` (the
  sin θ-weighted luminance integral already computed for the env CDF) — normalises
  them into a selection pmf, and uploads it as a new `FrameConstants` scalar tail
  (4 floats, packed before the Metal-only `tileOriginY`).
- Fallback: when total power is zero or non-finite the pmf degrades to uniform over
  the present groups (pre-change behavior).
- All existing emission geometry, flux formulas, validity guards, and the
  `depth ≥ 1` env-indirect partition are unchanged — only the selection pmf and its
  divisor change.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `photon-mapping`: photon-emission group selection changes from uniform over present
  groups to power-proportional (new requirement; the existing "Environment light emits
  photons" requirement already words flux as `1/p_sel` generically and stays valid).

## Impact

- Shaders: `wavefront_sppm.slang` (selection + per-branch `p_sel`), `common.slang`
  (`FrameConstants` + packer-order contract). All wavefront SPPM kernels and the
  megakernel share `FrameConstants` → full shader recompile (RGB `.spv` byte-change is
  expected and confined to the fc struct).
- Host: `renderer.py` (`_upload_sphere_lights` / `_upload_distant_lights` power sums,
  env luminance integral at `_ensure_env_uploaded`, pmf assembly + packing in
  `_pack_uniforms`), `environment.py` (return the CDF's luminance integral).
- Tests: hostless pmf unit tests + shader-source wiring guards
  (`tests/pbrt/test_sppm_selection.py` pattern); GPU A/B on
  `glass_caustics_test.usda` (noise_sigma via `compute_metrics` vs path at matched
  spp, median-of-ratio for unbiasedness); sppm parity/suite gates unchanged.
- No new descriptor bindings; spectral build unaffected in kind (pmf is scalar,
  λ-independent).
