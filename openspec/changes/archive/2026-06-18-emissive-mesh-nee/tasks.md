## 1. Correctness — remove the 256-triangle cap

- [x] 1.1 Add a headless correctness test (reuse `skinny.pbrt.parity.render_linear`, native Metal via `select_backend`): a synthetic emissive mesh tessellated to >256 triangles vs an equivalent low-poly emitter — assert they diverge today (high-poly is biased dark / energy lost) — captures the truncation bug before the fix
- [x] 1.2 In `renderer.py` `_upload_emissive_triangles`, drop `n = min(len(records), EMISSIVE_TRI_CAPACITY)`; size the emissive-triangle buffer to the actual count with a growth quantum + reallocate/rebind on increase (mirror `material_capacity` grow-and-rebind); `log` the final emissive-triangle count
- [x] 1.3 Audit readers of `EMISSIVE_TRI_CAPACITY` / `_num_emissive_tris` (descriptor buffer range, struct-layout/struct tests) so dynamic sizing doesn't desync a fixed descriptor range
- [x] 1.4 1.1 now passes the energy half (high-poly emissive mesh converges to the low-poly energy within noise); selection still uniform (variance unaddressed yet)

## 2. Host — power-weighted CDF

- [x] 2.1 In `_upload_emissive_triangles`, compute per-triangle weight `w_i = area_i × luminance(emission_i)` (Rec.709: 0.2126/0.7152/0.0722) and a normalized inclusive cumulative weight `cw[i] = Σ_{j≤i} w_j / Σw` plus per-triangle prob `p_i = w_i / Σw`; guard `Σw > 0` (else keep the `numEmissiveTriangles == 0` early-out path). A test-only `_emissive_uniform_selection` toggle builds the CDF uniform (`cw[i]=(i+1)/n`, `p_i=1/n`) for the power-vs-uniform A/B
- [x] 2.2 Pack the CDF **inline** into the existing emissive-triangle buffer (binding 18) — no new binding: store `cw[i]` in the record's `_v0.w` lane and `p_i` in `_v1.w` (host packing); add `cw`/`pi` properties to `EmissiveTriangle` in `common.slang`. (Avoids the Metal 31-buffer cap; mirrors how the env distribution rides one slot.)
- [x] 2.3 No new descriptor binding/range needed — the CDF rides binding 18. Fold binding-18 re-write into the emissive-buffer grow-and-rebind (`_rebind_emissive_descriptors`, Vulkan; Metal picks up the realloc at dispatch)

## 3. Shader — power-weighted selection + pdf

- [x] 3.1 Add a shared `sampleEmissiveTriangle(u, n)` helper in `scene_lights.slang` (binary search for smallest `i` with `cw[i] > u`, reading `emissiveTriangles[mid].cw`); in `nee.slang` `allLightsNEE` and `materials/skin/skin_direct.slang` replace `idx = uint(rng·N)` with this helper
- [x] 3.2 In `emissive_triangle_light.slang` `loadEmissiveTriangleLightImpl`, set `selectionPdf = tri.pi` (was `1/N`); `pdfArea = selectionPdf / triArea_i` and `pdfSolidAngle` already derive from `selectionPdf` (unchanged) so the MIS power-heuristic companion stays matched
- [x] 3.3 Confirm the megakernel + wavefront NEE both route through the shared `nee.slang` (one definition) so the change applies to both execution modes; recompile via the runtime

## 4. Variance + unbiasedness tests

- [x] 4.1 Promote 1.1 into the full gate: equal-spp relMSE on a multi/uneven-emitter scene drops materially vs the pre-change uniform baseline (power-weighted < uniform) — the variance assertion
- [x] 4.2 No-regression: `diffuse_arealight` corpus stays within the parity tolerance vs the pbrt reference (small lights unchanged)
- [x] 4.3 Unbiasedness: the converged multi-emitter image mean matches the uniform-selection mean within noise (only variance differs, not the expected image)

## 5. ReSTIR inheritance

- [x] 5.1 In `restir/light_ris.slang`, switch the emissive candidate index draw (the uniform `rng·N`) to the shared `sampleEmissiveTriangle` helper so the actual draw matches the power `selectionPdf` it reports through `loadEmissiveTriangleLightImpl` (a uniform draw + power pdf would bias RIS); reservoir/RIS/GRIS code untouched. Confirm existing ReSTIR tests (`test_restir_lights`/`test_restir_variance`) still pass

## 6. Documentation and close-out

- [x] 6.1 Update `docs/Architecture.md` (emissive buffer binding-18 note: CDF packed inline in the record `.w` lanes, dynamic sizing — no new binding) and `docs/Megakernel.md` (emissive-triangle NEE: power-weighted selection, no cap)
- [x] 6.2 Update `CHANGELOG.md`
- [x] 6.3 Run `ruff check src/`, the correctness + variance + regression tests (GPU, native Metal), and `openspec validate emissive-mesh-nee --strict`; verify no doc drift
