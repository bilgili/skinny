## 1. Correctness — remove the 256-triangle cap

- [ ] 1.1 Add a headless correctness test (reuse `skinny.pbrt.parity.render_linear`, native Metal via `select_backend`): a synthetic emissive mesh tessellated to >256 triangles vs an equivalent low-poly emitter — assert they diverge today (high-poly is biased dark / energy lost) — captures the truncation bug before the fix
- [ ] 1.2 In `renderer.py` `_upload_emissive_triangles`, drop `n = min(len(records), EMISSIVE_TRI_CAPACITY)`; size the emissive-triangle buffer to the actual count with a growth quantum + reallocate/rebind on increase (mirror `material_capacity` grow-and-rebind); `log` the final emissive-triangle count
- [ ] 1.3 Audit readers of `EMISSIVE_TRI_CAPACITY` / `_num_emissive_tris` (descriptor buffer range, struct-layout/struct tests) so dynamic sizing doesn't desync a fixed descriptor range
- [ ] 1.4 1.1 now passes the energy half (high-poly emissive mesh converges to the low-poly energy within noise); selection still uniform (variance unaddressed yet)

## 2. Host — power-weighted CDF

- [ ] 2.1 In `_upload_emissive_triangles`, compute per-triangle weight `w_i = area_i × luminance(emission_i)` (Rec.709: 0.2126/0.7152/0.0722) and build a normalized cumulative-power CDF `cdf[0..n]` (`cdf[0]=0`, `cdf[n]=Σw`); guard `Σw > 0` (else keep the `numEmissiveTriangles == 0` early-out path)
- [ ] 2.2 Upload the CDF as a `StructuredBuffer<float>` at a new descriptor binding, mirroring `_upload_env_distribution` (binding 31/32) — buffer sized `n+1`, grown with the triangle buffer; declare the binding in `bindings.slang`
- [ ] 2.3 Wire the binding into the descriptor set layout + write (`renderer.py` descriptor binding map) alongside the emissive-triangle buffer

## 3. Shader — power-weighted selection + pdf

- [ ] 3.1 In `nee.slang` `allLightsNEE`, replace `idx = uint(rng·N)` with a binary-search upper-bound over the emissive CDF (reuse the env `upperBound` helper) → triangle `i` with `p_i = (cdf[i+1]-cdf[i]) / cdf[n]`
- [ ] 3.2 In `emissive_triangle_light.slang`, carry `p_i` and set `selectionPdf = p_i`, `pdfArea = p_i / triArea_i` (was `1/N`); confirm `pdfSolidAngle` + the MIS power-heuristic companion derive from the same pdf
- [ ] 3.3 Confirm the megakernel + wavefront NEE both route through the shared `nee.slang` (one definition) so the change applies to both execution modes; recompile via the runtime

## 4. Variance + unbiasedness tests

- [ ] 4.1 Promote 1.1 into the full gate: equal-spp relMSE on a multi/uneven-emitter scene drops materially vs the pre-change uniform baseline (power-weighted < uniform) — the variance assertion
- [ ] 4.2 No-regression: `diffuse_arealight` corpus stays within the parity tolerance vs the pbrt reference (small lights unchanged)
- [ ] 4.3 Unbiasedness: the converged multi-emitter image mean matches the uniform-selection mean within noise (only variance differs, not the expected image)

## 5. ReSTIR inheritance

- [ ] 5.1 Confirm `restir/light_ris.slang` emissive-triangle candidates draw through the same `light.samplePoint` / `selectionPdf` seam and now use `p_i` with no ReSTIR-specific edit; add/extend a test asserting ReSTIR-on emissive selection is power-weighted (or document why the existing ReSTIR tests cover it)

## 6. Documentation and close-out

- [ ] 6.1 Update `docs/Architecture.md` descriptor binding map (new emissive CDF binding) and `docs/Megakernel.md` (emissive-triangle NEE: power-weighted selection, no cap)
- [ ] 6.2 Update `CHANGELOG.md`
- [ ] 6.3 Run `ruff check src/`, the correctness + variance + regression tests (GPU, native Metal), and `openspec validate emissive-mesh-nee --strict`; verify no doc drift
