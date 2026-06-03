# Tasks — ReSTIR DI

> **COMPLETE (2026-06-03).** ReSTIR DI ships the canonical integration (RIS owns
> primary direct: unified light domain with light + BSDF candidates, depth-0
> BSDF-hits-light gated), the unbiased GRIS spatiotemporal combination, a biased
> toggle, selectable regimes (default Spatial only), live UI tuning, and a
> variance-reduction demo. Converges to stock NEE on cornell_box_sphere/emissive +
> three_materials (glossy), A/B-verified against megakernel-PT / BDPT / wavefront-
> NEE. Two findings amended the plan (see design "Implementation outcome" +
> spec "Variance reduction" amendment): (1) the unbiased combination is the GRIS
> generalized balance heuristic (per-domain p̂ re-eval via the existing hit buffer),
> not a fat G-buffer + bare 1/Z; (2) progressive-temporal reuse double-counts
> correlated history on the accumulator, so it is progressive-limited and the
> default is spatial-only — proper deep temporal is the reprojected P3 follow-on.
> Build history is in git + the project memory.

## 1. Reuse-seam Slang module

- [x] 1.1 `shaders/sampling/reuse.slang` `reuseDirect` seam (both backends; identity ⇒ stock NEE; depth-0 ReSTIR gate). Parity bit-identical. Commit 35be502.
- [x] 1.2 `RESTIR_DI` reuse-mode constant (`fc.reuseMode == 1`) + the per-pixel G-buffer record (`{pos, normal}` in `restir_primary.slang`). Per-neighbour material for the unbiased combination is re-loaded from `wfHits` via `restirLoadLane` (the GRIS approach), so no fat materialId/wo G-buffer is needed.

## 2. Reservoir core + initial RIS (P1)

- [x] 2.1 `shaders/restir/reservoir.slang`: `LightSampleRef` + `Reservoir` + pure RIS ops, unit-tested. Commit a4a4ce1.
- [x] 2.2 `shaders/restir/light_ris.slang`: initial RIS over the unified light set — light-sampled (sphere / emissive-tri / env, ~uniform over active techniques) + BSDF-sampled (proposal direction traced to sphere lights / env) candidates; unweighted `p̂ = lum(f·Le)`, balance-heuristic mixture source pdf; no shadow rays. (Emissive triangles are NEE-only in the stock renderer ⇒ light-technique only — still unbiased.)
- [x] 2.3 G-buffer fill (pos/normal per pixel from the primary hit) in `restirFill`.

## 3. Spatial reuse + unbiased combination (P1)

- [x] 3.1 `restirSpatial`: gather `spatial_k` neighbours in `spatial_radius`; reject on G-buffer normal/depth dissimilarity; merge reservoirs.
- [x] 3.2 Unbiased combination: the GRIS generalized balance heuristic `m_s = M_s·p̂_s(z_s)/Σ_j M_j·p̂_j(z_s)` — the survivor's target re-evaluated in every source's own domain; DI same-light-point reconnection ⇒ Jacobian 1; domain check via the G-buffer. (Generalizes the planned m_i + 1/Z + Jacobian.)

## 4. Resolve + integration gate (P1)

- [x] 4.1 `restirResolve`: one shadow ray for the survivor; direct = `f(y)·V(y)·W` → path radiance (+ directional NEE).
- [x] 4.2 Integration gate: depth-0 `allLightsNEE` skipped via `reuseDirect`; the path tracer's depth-0 BSDF-hits-sphere (`wf_shade_common`) + env-miss (`wavefront_path`) terms gated under `reuseMode == RESTIR_DI`. `depth ≥ 1` unchanged; identity preserved.

## 5. Host: plugin + passes + buffers (P1)

- [x] 5.1 `sampling/reuse.RestirDiReuse` (reuse_mode + config selector). The reservoir (×2) + G-buffer buffers, ReSTIR config, passes/bindings are realized renderer-side in `vk_wavefront.RestirDiPass` (lifecycle/resize via the wavefront-pass rebuild).
- [x] 5.2 `sampling/registry`: register `RestirDiReuse`; `reuse_modes` → `["None", "ReSTIR DI"]`.
- [x] 5.3 `vk_wavefront.RestirDiPass`: build + schedule the pass set in wavefront mode; reuse-mode switch rebuilds.
- [x] 5.4 `renderer.py`: per-pixel buffers sized to the stream + rebuilt on resize; wavefront-only capability gate (megakernel/Metal → identity); ReSTIR config + reuse mode folded into `_current_state_hash`.
- [x] 5.5 `params.py`: ReSTIR sub-config params — regime, combine (unbiased/biased), `M_light`, `M_bsdf`, neighbours, radius, `M_cap` — data-driven, live (push-constant refresh, no rebuild), accumulation-resetting.

## 6. Progressive temporal (P2)

- [x] 6.1 Temporal merge (prev-frame reservoir at the same pixel, M-capped) folded into the GRIS combination. **Progressive-limited**: it double-counts correlated history on the accumulator (bias ∝ M_cap, glossy) — so it is non-default; deep temporal is the reprojected P3 follow-on.
- [x] 6.2 Ping-pong reservoir A/B; prev reset on accumulation reset (accumFrame-gated).

## 7. Biased toggle + tuning (P4)

- [x] 7.1 `biased` path (`RESTIR_FLAG_BIASED`): ΣM combination, skips the GRIS re-eval (faster). Toggle via the config flag.
- [x] 7.2 Sensible default tuning + documented cost/quality (docs/Wavefront.md §7.1).

## 8. Tests + verification

- [x] 8.1 Converge-to-reference: ReSTIR DI vs stock NEE on cornell_box_sphere/emissive + three_materials (`test_restir_lights.py`, `test_restir_render.py`).
- [x] 8.2 Variance reduction: ReSTIR error < stock-NEE error at low spp on a many-light scene (`test_restir_variance.py`, `assets/restir_variance_demo.usda`).
- [~] 8.3 Temporal beats spatial: **amended to a P3 (reprojected) property** — not achievable on the progressive accumulator (see the spec "Variance reduction" amendment). Diffuse-scene regime convergence is covered by `test_restir_regimes_converge`.
- [x] 8.4 Biased bounded: biased ΣM stays finite + within a small deviation on a diffuse scene (`test_restir_biased_toggle_bounded`).
- [x] 8.5 Capability gate: `reuse=ReSTIR` + megakernel → identity (`test_restir_megakernel_falls_back_to_identity`).
- [x] 8.6 Furnace preserved (env candidates gated off under furnace; the `reuseMode`/`bsdfPdf` gate guards leave furnace + reuse=none untouched — `test_sampling_parity`/`test_lights` pass); pinned-seed runs reproducible.
- [x] 8.7 `slangc` (main_pass + wavefront variants) + `py_compile` clean; `ruff` clean on the change's files (pre-existing unrelated lint debt in renderer.py untouched).

## Out of scope (this change)

- [ ] P3 reprojected temporal (motion vectors + prev-frame G-buffer + disocclusion) — follow-on change. Reserved in the regime selector.
- World-space / secondary-vertex ReSTIR; ReSTIR GI / PT; denoising.
