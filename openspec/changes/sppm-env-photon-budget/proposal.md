# Change: SPPM environment-aware per-pass photon budget

## Why

The env photon group (change `sppm-env-indirect-transport`) leaves SPPM renders
of mixed weak-local-light + environment scenes visibly speckled at interactive
pass counts — `assets/glass_caustics_test.usda` at 384²/48 spp has whole-image
`noise_sigma` 0.0272 vs the pre-env baseline's 0.0162, and power-proportional
group selection (change `sppm-power-proportional-photon-groups`) only recovered
−11%. Probes show the residual is **deposit sparsity, not a flux bug**: with a
constant gray env (L/pdf exactly uniform) the speckle persists; forcing the pmf
to sphere-only restores the baseline noise floor (σ 0.0154); and multiplying the
photon budget ×8 shrinks the env-only noise component by exactly √8
(√(0.0272²−0.0154²)=0.0224 → 0.0083). Env photons are emitted from the whole
bounding disc over the full sphere of env directions, so most miss the scene,
hit backfaces, or escape after one bounce — the few that deposit carry fat flux.
The estimator is unbiased (means match the path anchor); it just needs more
env photons than one-per-pixel to look converged.

## What Changes

- The per-pass photon count stops being a flat `width·height` and becomes
  **env-aware**: `N = round(pixels / max(1 − pmfEnv, 1/CAP))` with `CAP = 8`,
  where `pmfEnv` is the environment entry of the existing power-proportional
  group pmf. Expected **non-env** photon count stays exactly `pixels`
  (`N·(1−pmfEnv) = pixels` in the uncapped regime), so scenes without a live
  env are bit-identical, and env-lit scenes ride the extra budget entirely on
  the env group.
- `_sppm_photons_override` keeps absolute precedence (unchanged semantics).
- Normalisation is already per-emitted-count in the update stage
  (`fc` carries the emitted photon count), so the change is host-side only —
  no shader edit, no new binding.

## Impact

- Affected specs: `photon-mapping` (ADDED requirement).
- Affected code: `src/skinny/renderer.py` (`_pack_uniforms` SPPM tail — pmf is
  computed before the photon count and feeds it), `tests/test_sppm_selection.py`
  (budget formula unit tests), `docs/PhotonMapping.md`, `CHANGELOG.md`.
- Perf: photon stage scales ≤ ×8 only on env-lit SPPM scenes; measured wall
  cost on the repro is negligible (48 passes at ×8 ≈ 1.5 s at 384² on Metal).
  The Metal photon dispatch is already breadth-tiled
  (`sppm-photon-dispatch-tiling`), so the watchdog bound per command buffer is
  unchanged.
