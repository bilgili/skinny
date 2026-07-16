# Design — sppm-env-photon-budget

## Problem

SPPM env-indirect transport is unbiased but under-sampled at the default
one-photon-per-pixel budget. Diagnosis chain on
`assets/glass_caustics_test.usda` (384², 48 spp, Metal, constant "Neutral Gray"
env at intensity 0.5, sphere light r=0.2):

| probe | noise_sigma | conclusion |
|---|---|---|
| pre-env baseline (ecfdac5) | 0.0162 | target floor |
| HEAD (power-prop pmf) | 0.0272 | still speckled |
| pmf forced sphere-only | 0.0154 | machinery fine; env photons are 100% of the excess |
| pmf forced env-only | 0.0234 | env transport itself is the noise |
| constant gray env | speckled | L/pdf spikes ruled out (radiance/pdf uniform) |
| photons ×8 | 0.0175 | env component 0.0224 → 0.0083 = ÷√8 exactly |

The env-only noise component scales exactly as 1/√N_photons → pure Poisson
deposit sparsity. Cause: env photons launch from the whole bounding disc
(area πR², R = bounding-sphere radius ≈ 7.5 here) over all env directions —
most miss geometry, hit the ground backface (below-horizon flux), or escape
to the sky after one diffuse bounce (a flat ground plane cannot re-light
itself). The surviving deposits are sparse and carry Φ_env/N flux each.

## Decision

Host-side budget scaling, no shader change:

```
pmfEnv = sppm_pmf[3]                      # already computed for group selection
N = round(pixels / max(1 - pmfEnv, 1/CAP))   # CAP = 8
```

Properties:

- **Non-env expectation invariant:** E[non-env photons] = N·(1−pmfEnv) =
  `pixels` exactly in the uncapped regime — non-env-scene photon statistics and
  cost are unchanged, and `pmfEnv = 0` gives N = `pixels` → **bit-identical**
  renders for every env-free scene (and for path/bdpt, which don't read this).
- **Env rides on top:** the extra budget goes to the env group in expectation,
  attacking exactly the measured noise component.
- **Cap:** pmfEnv → 1 would send N → ∞; CAP=8 bounds the photon stage at ×8
  (measured ≈ free on the repro: 1.5 s / 48 passes at ×8, 384²). On the repro
  pmfEnv = 0.84 → ×6.25 → predicted σ ≈ 0.0165 ≈ the 0.0162 floor.
- **Unbiased by construction:** the update stage already normalises by the
  emitted count from `fc` (`self._sppm_photons_emitted` → packed field), and
  `sppmEmitPhoton` divides each branch by its actual selection probability.
  Changing N is variance-only.
- `_sppm_photons_override` retains absolute precedence (probe/tuning hook).

## Alternatives considered

1. **Raise the flat default (pixels × K for everyone)** — taxes every SPPM
   scene for an env-only problem; rejected.
2. **Emission importance (aim disc offsets at geometry, restrict to
   above-horizon flux)** — bigger win in principle (~×7 waste factor here) but
   requires shader-side pdfPos reweighting, is scene-shape-dependent, and
   composes with this change anyway; deferred as a follow-up.
3. **Eye-side env final gather (partition the env→X→VP one-bounce family out
   of the photon pass)** — converges like path tracing but is a transport
   redesign with a new double-count partition to prove; deferred.
4. **Deposit flux clamp** — biased; rejected.
5. **Revert env photons** — regresses shipped correctness
   (`sppm-env-indirect-transport`: env-lit scenes 0.78 → ~1.0); rejected.

## Risks

- Suite/parity SPPM images on env-lit scenes change (less noise). Gates compare
  against ceilings (`baseline` relMSE / self-consistency floors), so reduced
  variance passes; re-run the SPPM-relevant suite subset to confirm.
- Metal watchdog: photon dispatch is already breadth-tiled into flushed
  sub-batches (`sppm-photon-dispatch-tiling`); ×8 photons = more sub-batches,
  per-command-buffer bound unchanged.
- Grid capacity: the SPPM hash grid stores visible points, not photons —
  photon count does not touch its sizing.
