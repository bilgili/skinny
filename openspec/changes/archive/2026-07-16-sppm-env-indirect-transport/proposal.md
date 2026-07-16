# Proposal: sppm-env-indirect-transport

## Why

SPPM is measurably dim versus the path/BDPT anchors on environment-lit scenes: on the fair
null-sun glass-caustic scene the shadow-region ratio is 0.78 and the caustic-mask ratio 0.936
(recorded follow-up (C) of `distant-light-caustic-parity`). Root cause: the photon pass emits
from emissive triangles, sphere lights, and distant lights only — the environment light emits
no photons, so all env-INDIRECT transport (sky → bounce(s) → visible point) is missing. The
earlier `sppm-caustic-dimness` fix covered env DIRECT only (env-miss companion at the terminal
visible point). A prior env-photon attempt (obs 1410/1417) produced ~8× over-brightness on the
force-env probe and was abandoned; its flux formula `beta = L_env·πR²/(gsel·es.pdf)` actually
matches pbrt `ImageInfiniteLight::SampleLe` (`Le/(pdfPos·pdfDir·pSel)`, `pdfPos = 1/(πR²)`), so
the ~8× must be located elsewhere (leading suspects: first-hit deposits double-counting env
DIRECT already owned by VP env NEE + the env-miss companion; envIntensity applied twice;
probe methodology comparing direct+indirect deposits against a path indirect-only term).

Secondary latent bug folded in (follow-up (D)): `sppmDepositPhoton` computes
`phi = beta_photon · f_r` and never applies `vp.beta` (the eye throughput to the visible
point, stored in `sppm_state.slang` but unused). pbrt folds the visible point's beta into τ.
Invisible for directly-viewed VPs (beta ≈ 1) but under-counts caustics seen through glass or
along glossy eye chains.

## What Changes

- Add the environment as a fourth photon-emission group in `sppmEmitPhoton`
  (`wavefront_sppm.slang`), following pbrt's `ImageInfiniteLight::SampleLe`: importance-sample
  a direction with `sampleEnvDir`, emit inward from the scene-bounding disk, flux
  `beta = L_env·πR² / (gsel·es.pdf)`. Gated identically to env NEE (off in furnace mode /
  zero intensity).
- Root-cause and fix the prior ~8× over-brightness before enabling the group; the design must
  first re-measure with a sound probe (median, matched transport terms) and eliminate any
  env-DIRECT double count between first-hit photon deposits and the terminal-VP env lighting
  already owned by the eye pass.
- Apply `vp.beta` to deposited flux (at deposit or resolve, matching pbrt's τ fold) so
  through-glass / glossy-eye-chain caustics are weighted correctly.
- Spectral: env photon flux uses the same illuminant upsampling as env NEE under
  `SKINNY_SPECTRAL`, honouring the shared per-pass wavelengths (D5 design).
- Both backends (Vulkan + Metal) — same Slang source, no new bindings expected.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `photon-mapping`: photon emission gains an environment-light group so env-INDIRECT
  transport is present; the "direct lighting is not double-counted" requirement extends to
  env photons vs the terminal-VP env lighting; deposits must carry the visible point's eye
  throughput (`vp.beta`). Convergence gates: env-lit scenes must close the recorded 0.78
  shadow / 0.936 caustic gap against the path anchor.

## Impact

- `src/skinny/shaders/integrators/wavefront_sppm.slang` — `sppmEmitPhoton` (new group),
  `sppmDepositPhoton` / `wfSppmUpdate` (vp.beta application, possible depth-0 deposit gate).
- `src/skinny/shaders/integrators/sppm_state.slang` — vp.beta already stored; no layout change
  expected.
- `src/skinny/shaders/environment.slang` — read-only reuse of `sampleEnvDir` / `envImagePdf`.
- Recompiled kernels: all SPPM wavefront `.spv` (RGB + spectral); megakernel untouched.
- Tests: `tests/pbrt/` parity gates (env-lit scenes), new discriminating gate for env-indirect
  (fair null-sun scene), existing "no double count" scenario re-validated; recorded
  self-consistency baselines for sppm env scenes should tighten, never loosen.
- Risk: photon budget dilution (4th group lowers per-group photon counts); Metal watchdog
  unaffected (emission cost per photon unchanged).
