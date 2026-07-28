# Change: mlt-cross-backend-equivalence

## Why

The documentation says that MLT renders bit-identically on Vulkan and native
Metal. The renderer does not do this. The claim is an over-generalization of one
measurement at one small budget.

`CLAUDE.md`, `README.md`, `docs/Wavefront.md`, and `docs/Spectral.md` state
"bit-identical at equal budget, RGB and spectral", with the figure "measured
maxdiff 0, mean 0.2759076145 on `int_caustic`". That figure comes from change
`spectral-mlt`, task 4.3: `int_caustic` at **64x64, 8 spp, 512 chains**, with
`SKINNY_MLT_METAL_CHAIN_BATCH=512`. The suite scene now renders at **128x128,
512 spp, 16384 chains** — change `spectral-mlt` raised the budget from 256 spp
to 512 spp in the same change that recorded the claim. Nobody re-measured the
claim at the new budget.

At the manifest budget the two backends do not agree bit-for-bit:

| Mode | Metal mean | Vulkan mean | max abs diff | relMSE | PSNR | FLIP |
|---|---|---|---|---|---|---|
| RGB | 0.2519518466 | 0.2519517879 | 1.348e-4 | 5.302e-10 | 135.67 | 2.949e-6 |
| spectral | 0.2531574248 | 0.2531572506 | 4.181e-3 | 5.150e-8 | 114.94 | 1.779e-6 |

**This is not a defect.** The difference is bounded by the film-splat
quantization, not by chain divergence. Each backend stays bit-reproducible
across processes, and each backend passes the same pbrt-truth and
self-consistency gates on its own. See `design.md` for the measurement and the
mechanism.

The defect is the documentation. A reader who trusts "bit-identical" writes a
cross-backend equality assertion, and that assertion fails. The living spec is
the second half of the problem: it says only that MLT "SHALL run on native Metal
and Vulkan at parity" and never defines the equivalence class, so the docs
invented one.

## What Changes

- **Give the cross-backend equivalence class one owner.** The
  `metropolis-light-transport` spec states what "at parity" means for MLT: the
  same expected image, each backend bit-reproducible with itself, and a
  cross-backend difference bounded by the Q24.8 film-splat quantum. The
  documentation derives from that statement.
- **Restate the claim in every document that carries it** — `CLAUDE.md`
  (compatibility matrix, MLT backends row), `README.md` (compatibility matrix,
  integrator table), `docs/Wavefront.md`, `docs/MetropolisLightTransport.md`,
  `docs/Spectral.md` — with the measured numbers, the budget they were measured
  at, and the mechanism.
- **Record the mechanism where it is actionable**: the Q24.8 truncation in
  `mltFilmSplat` is the amplifier that makes a last-bit arithmetic difference
  visible. `wavefront_mlt.slang` gets that note beside `MLT_SPLAT_SCALE`.

No renderer behavior changes. No tolerance and no baseline is loosened; no gate
is touched. The parity matrix keeps the tolerances it has.

## Impact

- Affected specs: `metropolis-light-transport`
- Affected code: `src/skinny/shaders/wavefront/wavefront_mlt.slang` (comment
  only)
- Affected docs: `CLAUDE.md`, `README.md`, `docs/Wavefront.md`,
  `docs/MetropolisLightTransport.md`, `docs/Spectral.md`
