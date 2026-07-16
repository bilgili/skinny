# Tasks — sppm-env-photon-budget

## 1. Implementation

- [x] 1.1 Reorder the SPPM tail in `renderer._pack_uniforms` so the group pmf
      is computed before the photon count, and derive
      `sppm_photons = round(pixels / max(1 − pmfEnv, 1/CAP))` (CAP = 8 module
      constant) unless `_sppm_photons_override` is set (override keeps absolute
      precedence).
- [x] 1.2 Hostless unit tests (extend `tests/test_sppm_selection.py`): pmfEnv=0
      → pixels; pmfEnv=0.84 → ×6.25; pmfEnv=1 → capped ×8; override wins.

## 2. Validation (GPU, Metal, guarded)

- [x] 2.1 A/B `assets/glass_caustics_test.usda` 384²/48 spp: noise_sigma
      0.0272 → ≈0.0165 (within ~15% of the 0.0162 pre-env floor), mean ratio
      vs path anchor within tolerance.
- [x] 2.2 Env-free bit-identity probe: an SPPM scene with envIntensity 0
      renders bit-identical pre/post change.
- [x] 2.3 SPPM-relevant suite/parity subset green (int_caustic(+mtlx),
      samp_env_glossy(+mtlx), conductor_infinite, diffuse_arealight).

## 3. Docs + review

- [x] 3.1 Update `docs/PhotonMapping.md` (photon budget), `CHANGELOG.md`.
- [x] 3.2 Pre-merge review (codex; review-subagent fallback if codex
      unavailable), fold findings.
- [x] 3.3 `openspec validate sppm-env-photon-budget`, merge to main, archive.
