# Tasks: sppm-env-indirect-transport

## 1. Diagnosis — localize the prior 8× (D1; blocks everything else)

- [x] 1.1 Build the probe scaffold: flat-ground env-lit scene (reuse or derive from
      tests/assets), headless Metal render path, median-of-ratio metric via
      `metrics.compute_metrics` ROI helpers — no hand-rolled formulas
- [x] 1.2 Extract the path indirect-only reference correctly (path total − path direct on the
      same ROI; never `direct_light_index=1` on the SPPM side)
- [x] 1.3 Implement the forced-env probe in `sppmEmitPhoton` behind a temporary debug path with
      `gsel = 1.0` for the forced group; render and record the ratio
- [x] 1.4 Audit envIntensity application: confirm `sampleEnvDir` folds intensity once and beta
      does not re-multiply the env fetch
- [x] 1.5 Write down the 8× decomposition (probe-method error vs real code error) in design.md
      Open Questions — exit gate: probe ratio ∈ [0.9, 1.1] or localized residual cause

## 2. Env photon emission (D2, D3)

- [x] 2.1 Add `hasEnvGroup` gate (`furnaceMode == 0 && envIntensity > 0`) and 4th group
      (`chosen == 3u`, appended after distant) to `sppmEmitPhoton`
- [x] 2.2 Emit from the scene-bounding disk along `-ω` with sqrt-uniform in-disk offset
      (reuse distant branch's `rr = R·sqrt(u.x)`); `beta = L_env·πR²/(gsel·es.pdf)` (RGB path);
      **guard `if (es.pdf <= 0.0) return false;` after `sampleEnvDir`** (review F1 — pole/degenerate
      pdf yields inf beta that poisons RR and the walk; mirror guards at :577/:601)
- [x] 2.3 Spectral branch: `upsampleIlluminantBound(es.radiance, sw)` with the shared pass λ
      (mirror the sphere-light branch)
- [x] 2.4 Recompile all SPPM wavefront kernels (RGB + `-DSKINNY_SPECTRAL`); verify RGB `.spv`
      byte-identical for every non-SPPM kernel and the megakernel

## 3. vp.beta at deposit (D5) — DROPPED, shipped separately

- [x] 3.1 SUPERSEDED: sppm-vp-beta-resolve landed on main (d13c016) applying vp.beta at
      resolve (mathematically identical; fixed-point buffers stay vp.beta-free). This
      change rebased onto it; no vp.beta code here
- [x] 3.2 Obsolete with at-resolve form (flux buffers unscaled — no quantization risk)

## 4. Validation gates (D7; Metal backend, median-of-ratio, matched spp)

- [x] 4.1 Probe gate: forced-env ratio ∈ [0.9, 1.1] recorded
- [x] 4.2 Null-sun glass-caustic scene: shadow-box 0.78 → ≈1.0, caustic-mask 0.936 → ≈1.0 vs
      path anchor; set measured tolerances (tighten, never loosen)
- [x] 4.3 No-double-count (review F2 — single plane is vacuous, zero deposits): env-only scene
      with genuine indirect (open box / plane + occluder), SPPM total ≡ path total; plus
      separate single-plane "photons-add-zero" sanity check
- [x] 4.4 Directly-viewed VP regression: shipped glass-caustic gate statistically unchanged
      (vp.beta ≈ 1 there); fireflies stay 0 at default radius
- [x] 4.5 SUPERSEDED by sppm-vp-beta-resolve (d13c016): through-glass moves toward BDPT via the
      merged at-resolve vp.beta (tinted glass 0.00102→0.00067 vs bdpt 0.00074, validated there)
- [x] 4.6 Furnace-closure suite unchanged; non-env analytic-light scenes unchanged
- [x] 4.7 Full sppm parity-matrix rows re-run; re-measure and LOWER the recorded env-INDIRECT
      0.77 self-consistency baseline; update manifest `measured` fields
- [x] 4.8 Glossy-continuation suite (sppm-glossy-final-gather + env-escape-MIS gates) re-run

## 5. Hostless tests + hygiene

- [x] 5.1 Hostless test: group-count/gsel math includes env group iff gated on (mirror existing
      _SppmStub-style tests)
- [x] 5.2 Run `tests/test_metal_cleanup.py` gpu harness (shader-length change touches photon
      kernel) per metal-dispatch-hygiene
- [x] 5.3 `ruff check src/`, hostless pytest sweep

## 6. Docs + archive

- [x] 6.1 Update docs (Architecture.md if bindings/kernel list touched; ReSTIR/Wavefront
      untouched; CHANGELOG.md entry; CLAUDE.md compatibility matrix note if SPPM env scope
      wording changes)
- [ ] 6.2 `openspec validate` clean; codex pre-merge review (or review-subagent fallback);
      merge from worktree; `openspec archive --yes` + `git add -f` the archive
