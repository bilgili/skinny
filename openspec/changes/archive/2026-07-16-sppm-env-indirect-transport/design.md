# Design: sppm-env-indirect-transport

## Context

The wavefront SPPM photon pass (`wfSppmPhotonTrace`) emits from three groups —
emissive triangles, sphere lights, distant lights (`sppmEmitPhoton`,
`wavefront_sppm.slang:545`) — selected uniformly (`gsel = 1/G`). The environment
light emits nothing, so env-INDIRECT transport is absent: recorded 0.78
shadow-box / 0.936 caustic-mask dimness vs the path anchor on the fair null-sun
scene. Env DIRECT is already complete at the terminal visible point (env NEE in
`ld` + the env-miss companion from `sppm-caustic-dimness`).

Deposits are already partitioned from direct lighting: photons deposit only at
non-specular vertices with `depth >= 1u` (`wavefront_sppm.slang:789`), so a
light→VP first hit never double-counts the eye stage's NEE. This gate predates
the earlier env-photon attempt, which makes the prior ~8× over-brightness
(obs 1410/1417) *not* explainable as a depth-0 double count. The prior flux
formula `beta = L_env·πR²/(gsel·es.pdf)` is the textbook pbrt
`ImageInfiniteLight::SampleLe` form (`pdfPos = 1/(πR²)` on the bounding disk,
`pdfDir = es.pdf` solid-angle). The formula is not the obvious culprit; the
diagnosis below is a mandatory first stage.

`sampleEnvDir` (`environment.slang:135`) returns solid-angle pdf
`envImagePdf(iu,iv)/(2π²·sinθ)` — the same distribution env NEE uses under MIS,
which passes parity gates, so its normalization is trusted.

Secondary latent bug: `sppmDepositPhoton` computes `phi = beta_photon · f_r`
and omits `vp.beta` (eye throughput to the VP, stored in `sppm_state.slang` but
never read at deposit or resolve). pbrt folds vp.beta into Phi at deposit
(pbrt-v4 `sppm.cpp`, Phi += beta·f·vp.beta analogue). A/B on the shipped
caustic scene showed no effect (directly-viewed VPs, beta≈1); the miss is real
for VPs reached through glass or glossy-continued eye chains.

## Goals / Non-Goals

**Goals:**
- Env photon emission as a 4th group, flux-correct (probe ratio ≈ 1.0 vs path
  indirect on a flat-ground scene, median-based).
- Root cause of the prior ~8× identified and written down (even if it turns out
  to be probe methodology).
- `vp.beta` applied to deposited flux.
- Close the recorded env-scene gaps: null-sun scene shadow-box 0.78 → ≈1.0,
  caustic-mask 0.936 → ≈1.0 vs path anchor (median-of-ratio, matched spp).
- RGB `.spv` byte-identical for non-SPPM kernels; spectral path uses the shared
  per-pass λ (D5) with `upsampleIlluminantBound`.

**Non-Goals:**
- Megakernel SPPM (does not exist).
- Portal / sun-sky env sampling improvements; `sampleEnvDir` is reused as-is.
- SPPM dispersion (recorded v1 limit).
- Power-weighted group selection (uniform `1/G` stays; see Risks).
- Fixing the recorded sppm env-INDIRECT 0.77 self-consistency baseline for any
  scene other than by this transport addition (baselines re-measured, not
  hand-edited).

## Decisions

### D1 — Diagnosis-first: re-run the force-env probe with sound methodology
The prior probe (force `chosen=3` for all photons) reported 8× and the attempt
was abandoned without localization. Before shipping the emission group, re-run
the probe with three fixes, each a candidate explanation for the 8×:
1. **Forced selection must set `gsel = 1.0`** for the forced group. If the old
   probe kept `gsel = 1/G` while forcing env, every photon was over-weighted by
   G (×2–3 depending on scene lights).
2. **Check envIntensity single-application**: `sampleEnvDir(u, fc.envIntensity)`
   already folds intensity into `es.radiance`; beta must not multiply the env
   fetch again.
3. **Compare like with like, median not mean**: probe deposits are
   indirect-only (depth≥1 gate); the reference must be path's indirect-only
   term on the same flat-ground ROI (path total − path direct, or a
   direct-light-index-off pair rendered the *path* way — never
   `direct_light_index=1` on the SPPM side, which invalidates SPPM
   comparisons). Median-of-ratio per the sppm-caustic-dimness lesson.

Exit criterion: probe ratio in [0.9, 1.1] or a written root cause for the
residual. Alternative rejected: shipping the group and tuning a fudge factor —
banned (never hide a real divergence behind a tolerance).

### D2 — Flux formula: keep the pbrt SampleLe form
`beta = L_env(ω) · πR² / (gsel · es.pdf)` with `R = 0.5·|sceneBoundsExtent|`,
origin on the disk `center + ω·R` offset by a sqrt-uniform in-disk sample
(reuse the distant branch's exact `rr = R·sqrt(u.x)` sampling at line 623 —
same `1/(πR²)` pdf as a concentric map, and code symmetry keeps the pdf
assumption obvious), direction `-ω`. Identical geometry to the distant-light
branch (line 615), differing only in the non-delta direction pdf. Rationale:
matches pbrt exactly; the distant branch (same disk, delta pdf) is
parity-validated by the shipped `distant-light-caustic-parity` gates.

Design review verified the formula symbolically against pbrt-v4 `sppm.cpp` +
`ImageInfiniteLight::SampleLe` (`AbsCosTheta ≡ 1` for infinite lights,
`pdfPos = 1/(πR²)`, `pdfDir = es.pdf`): **exact match**. The formula is
proven; D1's probe is validation + methodology-bug-hunt, not a formula audit.
If the probe still reads 8×, the residual must live in shared code
(`envImagePdf` scale or `es.radiance`) — which would also break the passing
env-NEE MIS gates, bounding the plausible real error to ~0.

**Validity guard (review F1, mandatory):** `sampleEnvDir` returns
`es.pdf = 0.0` at the poles and for a degenerate distribution; dividing yields
`inf` beta that poisons Russian-roulette and the whole photon walk (the
deposit NaN guard only catches it per-VP). Mirror the other groups' guards
(`wavefront_sppm.slang:577/601`): `if (es.pdf <= 0.0) return false;`
immediately after `sampleEnvDir` (optionally skip near-zero `es.radiance` to
avoid wasted walks).

### D3 — Gate the group like env NEE
`hasEnvGroup = (fc.furnaceMode == 0u && fc.envIntensity > 0.0)`. Furnace mode
excluded: the constant furnace env is analytic in the eye stage and adding
photons would double it into the closure gates. Zero intensity excluded so a
disabled env doesn't dilute the photon budget. Group order appended after
distant (`chosen == 3u`) to keep existing RNG consumption for the other groups
unchanged as far as possible (still perturbed by G changing — accepted, SPPM is
stochastic across passes anyway).

Note (review F4): env NEE gates on `furnaceMode == 0` only (`nee.slang:102`,
no envIntensity check) — outcomes agree since at `envIntensity == 0` NEE
radiance is ~0 and photons are absent. `hasEnvGroup` deliberately *inherits*
env NEE's standing precondition that the env importance distribution
(binding 31, `envDistCdf`) is valid whenever not in furnace mode; this change
introduces no new validity assumption the renderer doesn't already guarantee.

### D4 — Reuse the existing depth≥1 deposit gate; no new partition logic
Env-DIRECT stays owned by the eye stage (env NEE + env-miss companion); env
photons contribute only depth≥1 deposits. The existing gate at line 789 already
implements exactly this split — no code change, but the double-count scenario
in the spec must be re-validated with env photons live (render the env-only
scene, confirm energy vs path anchor, not just "compiles").

### D5 — vp.beta: SHIPPED SEPARATELY (main d13c016, change sppm-vp-beta-resolve)
Superseded: a parallel session landed the vp.beta fix on main at **resolve**
(`wfSppmUpdate` scales `lIndirect` by `vp.beta`, per-λ before the spectral
resolve) rather than at deposit. Mathematically identical to the at-deposit
form for a per-pass-constant vp.beta (the same VP multiplies every deposit in
its accumulator), and it keeps the fixed-point flux buffers vp.beta-free — the
quantization watch item vanishes. This change REBASES on that commit and adds
no vp.beta code; only the env-emission scope remains here.

## Validation results (Metal, GPU-measured)

- **D1 probe** (D7 probe gate): constant-white env sppm/path totals ≡ 1.00 on
  ground incl. an 84%-indirect surface; HDR-env 0.99–1.05; prior 8× fully
  attributed to probe methodology. Formula proven pbrt-correct.
- **No-double-count** (review F2): env-indirect two-surface scene sppm≡path; the
  single-plane sanity scene = photons-add-zero (median 1.00, no deposits).
- **Megakernel `.spv` byte-identical** (2.4): worktree main_pass ≡ main-source
  main_pass — the probe `#if` blocks and the sppm edits leave every non-sppm
  compile unchanged.
- **Hostless** (5.1): 6 sppm-selection guards (env gate, G-count, F1 pole guard,
  pbrt SampleLe flux, probe-defines-off) + 2 render-log guards, green.
- **Metal kill harness** (5.2, metal-dispatch-hygiene): 3 gpu tests pass
  (clean-exit, SIGKILL-mid-render→GPU-usable, atexit) — the photon-kernel length
  change stays under the watchdog.
- **Full GPU suite sweep** (4.x): the env-lit **RGB** sppm gate the change
  targets — `samp_env_glossy sppm|wavefront` — PASSES. Every RGB combo passes.

### Pre-existing spectral GPU-gate failures (NOT this change)

The full sweep failed 10 suite scenes — mat_emissive(+mtlx), int_caustic(+mtlx),
spec_prism(+mtlx), mat_pbr_gold/copper/glass/plastic_pc — **all exclusively on
`|spectral` combos** (bdpt/path `|spectral` pbrt-truth + spectral
self-consistency; two `sppm|wavefront|spectral` rows). **Confirmed pre-existing
on origin/main (ecfdac5): the identical 10 fail with identical numbers without
this change.** Root cause: spectral-wavefront merged CPU-verified only, its
spectral GPU baselines never measured (CLAUDE.md: "Wavefront: NOT yet
GPU-render-validated — GPU self-consistency / prism-BDPT / white-furnace gates
pending interactive-Metal follow-up"). This is the first full GPU suite run since
that merge. This change's only interaction: `spec_prism sppm|wavefront|spectral`
(dome present → env-branch active) shifts 0.1554→0.1570 (~1%, a real env-indirect
+ RNG-reseed delta; both fail the same missing baseline). All 8 other failing
scenes fail on bdpt/path spectral rows this change cannot reach. Recording the
spectral GPU baselines is a filed follow-up, not in scope here.

### D6 — Spectral: same illuminant upsampling as env NEE
Under `SKINNY_SPECTRAL`, env photon flux = `upsampleIlluminantBound(es.radiance,
sw)` with the shared per-pass wavelengths from `sppmPassWavelengths()` —
identical to the sphere-light branch pattern. No dispersion (recorded SPPM v1
limit). RGB kernel bytes for non-SPPM shaders must stay identical.

### D7 — Validation gates (all median-of-ratio, matched spp, Metal backend)
- **Probe gate**: force-env probe ratio vs path indirect ∈ [0.9, 1.1].
- **Null-sun scene**: shadow-box and caustic-mask sppm/path → ≈1.0 (from
  0.78 / 0.936). Exact tolerances set from measurement, tightening the recorded
  numbers, never loosening.
- **No-double-count** (scene fixed per review F2): a single diffuse plane is
  vacuous — env→plane is depth 0 (skipped) and the bounce escapes with no
  second surface, so photons deposit *nothing* and the gate can't catch a
  double count. Use a scene with genuine env-indirect transport (open box /
  plane + occluder — two diffuse surfaces) so photons actually deposit at
  depth≥1, then assert SPPM total ≡ path total. Keep the single-plane case as
  a separate "photons-add-zero" sanity check.
- **Furnace closure**: furnace suite unchanged (group gated off).
- **Parity matrix**: full sppm rows re-run; the recorded env-INDIRECT 0.77
  self-consistency baseline for the affected scene is re-measured and lowered.
- **Regression**: non-env scenes (analytic lights only) byte-stable or
  statistically unchanged (G unchanged when no env → identical group math).

## Risks / Trade-offs

- [Photon budget dilution: 4th group cuts per-group photons by up to 25% on
  scenes with all groups present] → uniform selection is the existing
  convention; caustic gates from distant-light-caustic-parity re-run to confirm
  firefly count stays 0 at default radius. Power-weighted selection is a
  recorded future improvement, not this change.
- [vp.beta application changes every SPPM scene with non-trivial eye chains,
  including shipped glossy-continuation gates] → re-run sppm-glossy suite;
  expected direction is *more* energy through glass, none for direct-view VPs.
- [8× root cause could be real and in shared code (e.g. envImagePdf scale)] →
  D1 exit criterion blocks implementation until localized; if envImagePdf is
  implicated, env NEE gates would also need re-examination (they currently
  pass, which bounds the plausible error to the emission-side combination).
- [RNG stream perturbation when env group present changes existing sppm
  renders' noise patterns] → gates are median-based, not per-pixel.
- [Review F3 — zero-power authored DistantLight still occupies a group slot:
  on the fair null-sun gate scene the phantom-suppressing zero-power light
  counts in `G` (`wavefront_sppm.slang:558`) and emits beta=0 photons, so env
  gets ≤½ of N useful photons on exactly that scene] → variance only, not
  bias; median-of-ratio at matched spp still ≈1.0, but the confidence interval
  on the 0.936→≈1.0 caustic-mask gate widens — do not misread a noisy gate as
  a transport error. Power-weighted selection stays the recorded future fix.
- [Metal watchdog: more photon work per pass if env photons walk long paths] →
  emission cost per photon unchanged; RR from depth>1 bounds walk length; the
  photon-dispatch tiling (sppm-photon-dispatch-tiling) already bounds per-batch
  work.

## Open Questions — RESOLVED (D1 probe executed, Metal, 512², glass-caustics
## null-sun derivative scenes)

**D1 verdict: formula correct, prior 8× was probe methodology.** Evidence
(median-of-ratio of TOTALS vs the path render at matched spp — totals, not
indirect-extraction: per-pixel indirect ratios are noise-skewed low when the
extracted term is small; totals with an exactly-matching direct partition
(direct ratio ≡ 1.000 in every config) are the reliable gate):

| config | open ground | under-sphere | box face (84% indirect) |
|---|---|---|---|
| constant white env, forced (gsel=1) | 1.00 | 1.00 | 1.01 |
| HDR env, forced | 0.99 | 0.94 | 1.00 |
| HDR env, shipping (G=2 with zero-power distant) | 1.01–1.02 | 1.05 | 1.08 |
| distant-light control (env off, validated branch) | ~1 (noisy) | 1.05 | 0.89 (noisy) |

- Constant-env totals ≡ 1.00 including a heavily indirect surface ⇒ the
  `beta = L·πR²/(gsel·es.pdf)` chain, the depth≥1 partition, and the resolve
  normalization are all correct end-to-end on the GPU.
- The prior 8× decomposes as probe-method error: forced selection kept
  `gsel = 1/G` (×G over-weight), compared photon-only deposits against a
  path term that wasn't indirect-only, and used mean (firefly-pulled) not
  median. None of it was the flux formula.
- Under-sphere/box excess (+5–8%) in the shipping config is plausibly REAL
  transport the path reference cannot sample — env-through-glass SDS caustics
  (photons carry them; unidirectional path structurally cannot; same lesson as
  distant-light-caustic-parity where SPPM was the most-correct integrator).
  Final adjudication belongs to the parity-matrix gates, not this probe.
- `envImagePdf` normalization confirmed trustworthy transitively: constant-env
  closure at 1.00 plus passing env-NEE MIS gates bound any distribution error
  to ~0.
