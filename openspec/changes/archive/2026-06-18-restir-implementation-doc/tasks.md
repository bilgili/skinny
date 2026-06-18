## 1. Verify sources (ground the doc in shipped code)

- [x] 1.1 Re-read the shaders and confirm symbols/equations: `restir/reservoir.slang` (`reservoirUpdate`, `reservoirMerge`, `reservoirFinalize`), `restir/light_ris.slang` (`restirEvalRef`, `_mixPdf`, `restirFillReservoir`, `restirResolveReservoir`, `restirDirectional`, `octEncode/Decode`, `sphereUVFromPoint`), `restir/restir_primary.slang` (`restirLoadLane`, `restirFill`, `restirSpatial`, `restirResolve`, the `RestirPC` push constant + flag bits), `sampling/reuse.slang` (`reuseDirect` gate)
- [x] 1.2 Re-read the host glue and confirm the GUI→push-constant mapping: `renderer.py::_restir_build_config` + `_RESTIR_REGIME_FLAGS = [0x1,0x3,0x2]` + `biased |= 0x4` + the pass-rebuild key; `vk_wavefront.py::RestirDiPass` (three pipelines, set-1 bindings 0–4, 36-byte push constant, `record_primary_direct` dispatch order); `sampling/reuse.py::RestirDiReuse`; `sampling/registry.py`; `params.py` ReSTIR params (paths + ranges); the dedicated **ReSTIR** group in `ui/build_app_ui.py`
- [x] 1.3 Inspect `docs/diagrams/restir_pipeline.svg` and decide whether it matches the shipped `fill → spatial → resolve` flow + the canonical-integration gate, or needs updating

## 2. Write docs/ReSTIR.md

- [x] 2.1 Header + Overview: what ReSTIR DI is, what it accelerates (primary-hit direct lighting variance), the wavefront-only / flat-material-primary scope, and a one-paragraph "how it differs from stock NEE"
- [x] 2.2 Stages of rendering: where ReSTIR sits in the wavefront pipeline (bounce-0 hook in `WavefrontPathPass`), the three passes `fill → spatial → resolve`, the per-pixel reservoir (double-buffered A/B) + G-buffer, deferred visibility, and the canonical-integration gate (`reuseDirect` zeroes depth-0 inline NEE; depth ≥ 1 unchanged); embed the pipeline SVG
- [x] 2.3 Equations: streaming RIS (`w_i = p̂/p_src`, survivor ∝ `w_i`, `W = Σw_i/(M·p̂(y))`); the unshadowed unweighted target `p̂ = lum(f·Le)`; the balance-heuristic mixture source pdf `p_mix = (M_light·p_light + M_bsdf·p_bsdf·[wi→sphere|env])/M`; light-area→solid-angle pdf `d²·pdfArea/cosθ`; multi-reservoir merge `w = p̂·W·M`; GRIS `m_s = M_s·p̂_s/Σ_j M_j·p̂_j`, `w_s = m_s·p̂_q·W_s`, `W_out = Σw_s/p̂_q`; biased `ΣM` combination; DI reconnection (identity Jacobian). Cite the backing paper inline at each
- [x] 2.4 Equation→implementation mapping: a table/section pairing each equation with the exact shader symbol/file that realizes it (per design D3)
- [x] 2.5 Design choices: the six locked decisions (regimes, primary-hit scope, unified light domain, unbiased default + biased toggle, canonical integration, wavefront-only) and the shipped deviations (GRIS over bare `1/Z`, spatial-only default, octahedral env refs, separate RNG stream); reference the brainstorm doc for history
- [x] 2.6 GUI controls: document the dedicated **ReSTIR** group and each of the eight controls (`Reuse`, `ReSTIR regime`, `ReSTIR combine`, `M light`, `M bsdf`, neighbours, radius, `M cap`) as control → param path → push-constant field/flag → effect; note live push-constant tuning vs reuse-mode/regime pass rebuild, and accumulation reset
- [x] 2.7 Limits & caveats: wavefront-only capability gate (megakernel/Metal fall back to identity), flat-material-only primary hits, directional lights as plain NEE outside the RIS, temporal-on-progressive bias (∝ `M_cap`, glossy), reprojected temporal reserved as the P3 follow-on
- [x] 2.8 References section: Talbot 2005, Bitterli 2020, Lin 2022, Veach 1997, Wyman et al. ReSTIR course — full citations
- [x] 2.9 Verification notes: how ReSTIR is checked against stock NEE (`tests/test_restir.py`, `test_restir_lights.py`, `test_restir_render.py`, `test_restir_variance.py` + `assets/restir_variance_demo.usda`)

## 3. Diagram

- [x] 3.1 If task 1.3 found the SVG stale, update `docs/diagrams/restir_pipeline.svg` to the shipped flow (SVG only, no ASCII); otherwise confirm it embeds correctly in `docs/ReSTIR.md`

## 4. Cross-links

- [x] 4.1 Add a ReSTIR cross-link/pointer in `docs/Architecture.md` (module map / reuse seam) and `docs/Wavefront.md` (the bounce-0 reuse hook section)
- [x] 4.2 Add a `docs/ReSTIR.md` pointer to `README.md` and to the docs list in `CLAUDE.md`

## 5. Verify

- [x] 5.1 Cross-check every equation in the doc against its named shader symbol (no transcription drift); confirm GUI rows match `params.py` ranges + `renderer.py` flag mapping
- [x] 5.2 Confirm no ASCII/text-art diagrams remain in the new/edited docs; all diagrams are SVG
- [x] 5.3 Check all intra-doc and cross-doc links resolve
- [x] 5.4 `openspec validate restir-implementation-doc --strict` passes
