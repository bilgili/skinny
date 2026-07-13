# Investigate SPPM vs path/bdpt dimness on the caustic scene

## Why

SPPM (`--integrator sppm`, wavefront) does not visually match path/bdpt on
`assets/glass_caustics_test.usda`. path and bdpt agree to <0.2% (ground truth);
SPPM differs. Characterised robustly (256², RGB, Metal, **same sample count**
=2048, linear-HDR, **firefly-robust median** per region):

| region | SPPM / path |
|--------|-------------|
| flat lit ground | 0.75 |
| back wall       | 0.84 |
| caustic+shadow  | 0.77 |
| whole image     | 0.84 |

**SPPM is uniformly ~15–25% dimmer** than path/bdpt — no region is genuinely
brighter. The perceived "over-bright caustics / brighter shadows" is **fireflies
+ a sharper caustic edge that the ACES tonemap compresses harder** than path's
smoother, noisier version — over a slightly-dim background — not an energy
excess.

### Measurement caveats (recorded so the next pass avoids them)

Two artifacts contaminated the first analysis and must be avoided:

1. **Firefly-contaminated means at low spp.** A box **mean** at 1–16 spp is
   dominated by a few firefly pixels — path itself read 263× the converged value
   at 1 spp. Use a **median** (or high-spp) statistic, never a low-spp mean.
2. **A per-spp scaling in the linear-HDR (`--format hdr --tonemap linear`)
   export** common to *both* integrators — path's own flat-ground median "halves"
   per spp doubling, so cross-spp linear comparisons are invalid. **Compare
   path vs SPPM at the SAME spp**, where the artifact cancels (the ~0.75–0.84
   ratio is stable across 512/1024/2048).

An earlier draft of this proposal claimed a catastrophic sample-count-dependent
"1/N over-brightness / 58× direct-term bias." **That was these two artifacts, not
a real integrator bug — retracted.** The genuine discrepancy is the modest,
spp-invariant dimness above.

## What Changes

- Investigate the ~15–25% SPPM dimness vs path/bdpt at matched spp: is it
  (a) missing/under-weighted indirect transport (SPPM's photon term under a
  finite radius), (b) a normalization constant, or (c) genuinely slower
  convergence (test: does the same-spp ratio move toward 1.0 at very high spp, or
  hold at ~0.8?).
- Reduce SPPM fireflies if they prove to be a deposit/radius artifact rather than
  ordinary variance.
- Gate: same-spp SPPM-vs-path region medians within tolerance on the caustic
  scene.

## Impact

- Affected specs: `photon-mapping`.
- Affected code: `shaders/integrators/wavefront_sppm.slang` (deposit / radius /
  resolve) — TBD by the investigation.
- Independent of `sppm-photon-dispatch-tiling` (the wedge/tiling fix, which is
  correct and unrelated).
- **Not yet root-caused.** This change captures the corrected characterisation +
  a clean measurement methodology; the fix follows the investigation.
