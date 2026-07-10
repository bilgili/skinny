# Tasks: SPPM glossy-continuation env-escape MIS + threshold reach

## 1. Shader — env-escape + emissive-hit MIS
- [x] 1.1 Carry `misBsdfPdf` in `wfSppmEye` (init `-1.0`; set on each glossy-continue
      bounce to `bs.pdf` when reflective + non-furnace, else `-1.0`).
- [x] 1.2 Weight the `!hit` env escape by `powerHeuristic(misBsdfPdf, envPdf(dir))`
      when `misBsdfPdf > 0`, else full weight. Mirror `integrators/path.slang`.
- [x] 1.3 Emissive-hit MIS: `spawnedBySpecular` becomes delta-only (`bs.pdf<=0`),
      carry `prevBsdfPdf`; a non-delta glossy-continued emissive hit adds only
      `powerHeuristic(prevBsdfPdf, pdfLightSA)` (reviewer finding — glossy-continue
      was wrongly full-weight, latent double-count the threshold raise exposes).
- [x] 1.4 slangc SPIR-V compile-probe clean (Metal in-process compile at gate time).

## 2. Host — threshold reach
- [x] 2.1 `_SPPM_GLOSSY_ROUGHNESS_DEFAULT` `0.5 → 0.6`; document the pbrt-import
      `usd = r**0.25` mapping and the alpha `≲ 0.36` polished-metal cutoff.

## 3. Verify
- [x] 3.1 `test_scene_matrix_gate[conductor_infinite]` `sppm|wavefront` green on Metal.
- [x] 3.2 `test_suite_matrix_gate[mat_conductor / mat_conductor_mtlx]` SPPM baselines
      unchanged; furnace conductor scenes unaffected.
- [ ] 3.3 Update `openspec/specs/photon-mapping/spec.md` (archive step folds the delta).
- [ ] 3.4 CHANGELOG entry; codex pre-merge review.
