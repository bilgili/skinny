# Tasks — renderer chart selection

## 1. Shader chart maps (nf-chart-shader)
- [x] 1.1 `NF_CHART` build define (`#ifndef` → default V1, byte-identical); integer codes V0=0/V1=1/V2=2(reserved)/V5=5
- [x] 1.2 V0 cylindrical `nf_chart_square_to_dir`/`nf_chart_dir_to_square` (φ=2πu, cosθ=v) — transliterate `square_to_hemisphere`/`hemisphere_to_square` (V0 inverse does NOT clamp off the boundary; matches spline_flow)
- [x] 1.3 V5 equirectangular (φ=2πu, θ=(π/2)v) + per-sample log-Jac `log(π²)+log sinθ`
- [x] 1.4 `nf_chart_logjac(z)` returns `NF_LOG2PI` (V0/V1) or `π²·sinθ` (V5); dispatch z-only (no `wo`)
- [x] 1.5 Swap `NF_LOG2PI` → `nf_chart_logjac(z)` at the two pdf sites under `#if NF_CHART`; V1 path verbatim → V1 render reproduces the pre-change numbers bit-for-bit
- [x] ~~1.6 V2 frame threading~~ DEFERRED — needs local `wo` absent from the `.nrec` schema (see proposal)

## 2. Host wiring (nf-chart-host)
- [x] 2.1 `NeuralBuildConfig.chart: str = "V1"`; `slang_defines` emits `-D NF_CHART=<n>` for non-V1; `cache_tag` chart slug (verified: V1→() , V0→NF_CHART=0, V5→NF_CHART=5)
- [x] 2.2 Loader chart guard: PTG1-trailer chart string validated vs built `cfg.chart` (`expect_chart`); wired at the two renderer load sites; clear mismatch error
- [x] 2.3 `guiding_variance_sweep`: `AVAILABLE_CHARTS = {V0,V1,V5}`; `_neural_config` threads `cell["chart"]`
- [x] 2.4 `nis_train_offline.py --chart` widened to `{V0,V1,V5}`

## 3. Parity + measurement (nf-chart-parity)
- [x] 3.1 `nis_chart_parity.py`: numpy mirror of each shader chart (V0/V1/V5) vs `spline_flow` `CHART` (both directions + log-Jac) — max|Δ| ≤ 1e-14 (PARITY OK). Render smoke: all charts UNBIASED (mean ~0.225; V5's π²·sinθ Jacobian correct)
- [x] 3.2 Default V1 byte-identity: built `NF_CHART=V1` config emits no `-D NF_CHART`; V1 render reproduces the prior E0 numbers exactly
- [ ] 3.3 Glass chart sweep (train per chart × coupling, render multi-seed, dump all PNGs) — does any chart beat V1? [8-seed run in flight]
