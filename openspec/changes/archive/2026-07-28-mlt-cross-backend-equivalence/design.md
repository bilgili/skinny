# Design — MLT cross-backend equivalence

## Question

Is the "bit-identical across backends" claim stale, or does MLT diverge across
backends because of a defect?

Answer: **the claim is stale.** The renderer is correct. The measurement below
identifies the exact mechanism and bounds it.

## Measurement

Scene `int_caustic` from `tests/pbrt/corpus/manifest.json`, combo
`integrator=mlt, execution_mode=wavefront`, one backend per process, through
`skinny.pbrt.parity.render_combo`. Metrics come from `metrics.compute_metrics`.

### At the current manifest budget (128x128, 512 spp, 16384 chains)

| Mode | Metal mean | Vulkan mean | max abs diff | mean abs diff | pixels that differ | relMSE | PSNR | FLIP |
|---|---|---|---|---|---|---|---|---|
| RGB | 0.2519518466 | 0.2519517879 | 1.3481e-4 | 2.793e-6 | 44.0 % | 5.302e-10 | 135.67 | 2.949e-6 |
| spectral | 0.2531574248 | 0.2531572506 | 4.1812e-3 | 1.906e-6 | 0.83 % | 5.150e-8 | 114.94 | 1.779e-6 |

### At the budget the old claim was measured at (64x64, 8 spp, 512 chains)

| Mode | Metal mean | Vulkan mean | max abs diff | pixels that differ | relMSE | PSNR |
|---|---|---|---|---|---|---|
| RGB | 0.2631299827 | 0.2631291333 | 7.728e-4 | 4.4 % | 1.770e-8 | 119.85 |

The small budget also does not give `maxdiff 0` today. The recorded mean
(0.27545003 in `spectral-mlt` task 4.3, 0.2759076145 in `CLAUDE.md`) matches
neither backend now, and the two recorded figures do not match each other, so
the original configuration is not exactly reproducible. The claim is therefore
restated from new measurements, not repaired from the old ones.

### Control: is each backend reproducible with itself?

Yes. A second process on the same backend, same budget, gives a **pixel-identical**
image on both Metal and Vulkan (`maxdiff 0.0`). The divergence is cross-backend
only. This is what the parity gate depends on, and it holds.

## Mechanism

The difference is an **integer number of film-splat quanta**.

`mltFilmSplat` accumulates radiance as Q24.8 unsigned fixed point:

    InterlockedAdd(lightSplatBuffer[base + 0u], uint(radiance.x * MLT_SPLAT_SCALE), orig);

`MLT_SPLAT_SCALE` is 256.0 and the conversion truncates. `wfMltResolve` scales
the frame's splats by `b / mpp_actual` and the accumulation buffer averages over
the frames, so one splat unit is worth

    q = b / (256 * mpp_actual * frames)

in final image units. For the manifest budget, `mpp_actual = 1` and
`frames = 512`, so `q = 1.9270e-6`.

Divide the measured maximum difference by `q`:

| Pair | max abs diff | q | ratio |
|---|---|---|---|
| RGB, 128x128 512 spp | 1.3481e-4 | 1.9270e-6 | **69.959** |
| RGB, 64x64 8 spp | 7.7280e-4 | 1.2880e-4 | **6.000** |

The ratios are integers, to within the float32 rounding the accumulation buffer
adds on top. The per-pixel differences cluster on the small integer multiples
±1, ±2, ±3 — the RGB histogram at the manifest budget puts 2520 pixels
at +1q and 2349 at −1q, and falls off from there.

An integer-quantum difference is the signature of a **truncation-boundary flip**,
not of a diverged Markov chain. Slang compiles the same source through two
different back ends — SPIR-V for Vulkan, MSL for native Metal. The two back ends
are free to contract multiply-add pairs differently and to use different
transcendental implementations, so the same radiance arrives at
`uint(radiance * 256.0)` differing in its last bits. When those last bits
straddle an integer boundary, the splat lands one unit apart. Over 512 frames
these ±1 events accumulate into the observed few-quanta spread.

Chain divergence would look nothing like this. Two MLT runs with different chain
seeds differ at the scale of the Markov noise itself — the same scale as the
scene's 0.12 relMSE gate. The measured RGB relMSE is 5.3e-10, eight orders of
magnitude below it.

The host normalization constant `b` shows the same last-bit story from the other
side: Metal computes 0.2525743171504189 and Vulkan 0.25257431707728983, a
relative difference of 2.9e-10. `b` comes from a float64 host CDF over float32
GPU bootstrap weights, so a last-bit difference in the weights is visible in `b`.
That difference is far too small to explain the image difference on its own
(2.9e-10 against 1.1e-5), which is further evidence that the amplifier is the
splat quantization and not the normalization.

Spectral behaves the same way with a longer tail: 99.2 % of pixels agree
exactly, most of the rest differ by ±1q or ±2q, and a handful reach hundreds of
quanta (the maximum is 2161q = 4.18e-3 on one pixel). Hero-wavelength sampling
inverts a wavelength pdf, so a last-bit difference there can move one sampled
wavelength and change one contribution's magnitude, instead of only flipping a
truncation boundary. The result is still unbiased and still six orders below the
gate.

## Decision

State the equivalence class, do not chase bit-identity.

- **Rejected: force bit-identity across back ends.** This needs contraction
  disabled and matched transcendental implementations on both targets. It would
  cost performance on both back ends, it is not enforceable against future
  compiler versions, and it buys a property no gate needs.
- **Rejected: round instead of truncate in `mltFilmSplat`.** Rounding does not
  remove the boundary — it moves it. It changes every MLT image, which would
  invalidate the recorded MLT baselines for a cosmetic reason.
- **Rejected: add a cross-backend gate to the parity harness.** The harness
  renders one backend per process and gates each backend against pbrt truth and
  against the same self-consistency anchor. A cross-backend axis would double
  the GPU sweep to assert a property that the two existing gates already cover
  from both sides.
- **Chosen: one owner for the claim.** The `metropolis-light-transport` spec
  defines the equivalence class. Every document that used to assert bit-identity
  states the measured bound, the budget, and the mechanism instead.

## The general rule this exposes

A cross-backend identity measured at one budget does not hold at another. The
quantum `q` scales with `b / (256 * mpp_actual * frames)`, so a smaller budget
has a coarser quantum and fewer chances to straddle a boundary — which is why
the original small-budget run could plausibly read as exactly zero. Any future
"identical on both backends" claim must record the budget it was measured at,
and must not be generalized past it.
