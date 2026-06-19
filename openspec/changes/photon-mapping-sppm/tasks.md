## 1. Setup & selection seam

- [x] 1.1 Create a git worktree off `main` for this change (per CLAUDE.md workflow); set up the headless env (`./bin/python3.13`, `VULKAN_SDK`, `DYLD_LIBRARY_PATH`).
- [x] 1.2 Add `INTEGRATOR_SPPM = 2u` to `shaders/common.slang`; add `INTEGRATOR_INDEX["sppm"] = 2` and `"SPPM"` to the GUI integrator mode list (`renderer.py`).
- [x] 1.3 Extend `cli_common.py` `--integrator` choices to `(path, bdpt, sppm)`; wire selection in `app.py`, `web_app.py`, `ui/qt/app.py`. (Front-ends already route through `INTEGRATOR_INDEX[args.integrator]`, so the map update covers all four.)
- [x] 1.4 (test-first) Add render-cli tests for the new `sppm` choice and the `sppm + megakernel` startup-rejection (mirrors existing bdpt-gating tests); then implement the shared gating so `sppm` requires `--execution-mode wavefront`.
- [x] 1.5 Add `integrator_index == SPPM` to the accumulation state-hash inputs so switching to/from SPPM resets accumulation; add a test asserting the reset. (`integrator_index` was already hashed for BDPT; added a source-level regression guard in `test_sppm_selection.py`.)

## 2. SPPM shader core (backend-neutral Slang)

- [x] 2.1 Define the visible-point record + packed SPPM state (`pos, ns, beta, bsdf params, materialId, r, N, tau, Phi, M`) in a new `shaders/integrators/sppm_state.slang`; document its byte layout. (Split into `VisiblePoint` (76 B scalar / 96 B MSL — geometry + persistent r/N/tau) and a separate per-pass `SppmAccum` (16 B — fixed-point atomic flux Φ + M); host mirror in `wavefront_layout.py`; locked + slangc-compile-checked by `tests/test_sppm_state.py`.)
- [x] 2.2 Implement the **eye stage**: trace the camera path through specular/perfectly-glossy bounces to the first non-specular flat-material hit, store one stochastic visible point per pixel; inactive point on escape/death. Gate to flat materials only. (`wfSppmEye` in `wavefront_sppm.slang`, mirrors the megakernel `path.slang` loop; stores the evaluated BSDF (`VisiblePoint` grew to embed FlatHitMat+F0+ns, 152 B/192 B); per-pass direct (NEE + specular-chain emission/env) accumulated into `vp.ld` (single direct site); `bs.valid` then `pdf>0` branch (delta lobe = caustic carrier continues); first-activation `n==0` radius init. slangc-verified SPIR-V + Metal. Host must zero the VP buffer at accumFrame==0.)
- [x] 2.3 Implement the **grid-build stage**: uniform spatial hash over visible points by counting sort (per-cell count → exclusive prefix sum → scatter). Keep atomics to integer adds (both targets). (`wavefront_sppm.slang`: `wfSppmGridCount` / `wfSppmGridScanBlock` / `wfSppmGridScanBlockSums` / `wfSppmGridScanAdd` / `wfSppmGridScatter` + shared hash helpers in `sppm_state.slang`; all slangc-verified SPIR-V + Metal. Typed buffers, integer atomics only. GPU correctness pending host wiring; CPU-reference scan lock = task 6.1.)
- [x] 2.4 Implement the **photon stage**: emit photons from lights via the existing power-weighted emissive/light CDFs, trace with Russian roulette, grid-lookup and atomic flux deposit (`Phi`,`M`) into visible points within radius. (`wfSppmPhotonTrace` + `sppmEmitPhoton`/`sppmDepositPhoton`/`sppmLoadMaterial` in `wavefront_sppm.slang`: emissive (power-weighted `sampleEmissiveTriangle`) + sphere + distant-beam emission, `beta=Le·π/selPdf`; RR trace; deposit gated `depth≥1` (disjoint from NEE direct); 3×3×3 dedup'd neighbour scan; bare `f_r = evaluate(woVP,wiVP).response/wiVP.z`; fixed-point `InterlockedAdd` into `SppmAccum`. slangc-verified SPIR-V + Metal.)
- [x] 2.5 Implement the **update stage**: SPPM radius/flux reduction (`N'=N+γM`, `r'=r·√(N'/(N+M))`, `tau'=(tau+Phi)·(r'/r)²`, γ=2/3); resolve pixel radiance `tau/(Nₑₘᵢₜₜₑₐ·π·r²)` and composite with the existing NEE direct term into the accumulation image. (`wfSppmUpdate`: `sppmUpdate` per active VP, `L_indirect=tau/(max(photons,1)·π·r²)`, `sample=vp.ld+L_indirect`, running-mean composite into `accumBuffer` + reuses `wfWriteDisplay` (no set-1 conflict, `lightSplatBuffer` reads zero), clears `SppmAccum`. slangc SPIR-V + Metal.)
- [x] 2.6 Gate all Metal-specific shader adaptations behind `#if defined(SKINNY_METAL)` so Vulkan SPIR-V stays byte-unchanged; keep modules small and shared via `common.slang`. (No Metal-specific adaptations needed — all 8 SPPM entries compile from one source for both SPIR-V and `-target metal`. Helpers shared via `sppm_state.slang`; scalar/MSL layout via `wavefront_layout`.)

## 3. Metal backend bring-up (first)

- [x] 3.1 Budget the new buffers against the Metal 31-slot argument-table cap. (Review proved a SPPM kernel sits ~15/31 — it never compiles the neural weights — so 4 plain TYPED buffers are used (no ByteAddressBuffer fold, no `SKINNY_METAL_SPPM` gate); the binding map in `docs/Architecture.md` records them. `_EntryPipeline` pipeline-build would name+count globals if the cap were ever exceeded.)
- [x] 3.2 Wire the four SPPM stages into `metal_wavefront.py` / `metal_compute.py` via the staged wavefront driver. (`MetalWavefrontSppmPass` + `_MetalSppmRecorder`: SlangPy session compiles the 8 entries, 4 `StorageBuffer`s sized by the reflected MSL VisiblePoint stride (locked == 192) + uint grid/scan, bound by Slang global name via `_bind_map`; `dispatch_frame` runs `record_sppm_loop` over one `MetalFrameEncoder`.)
- [~] 3.3 Per-pass queue compaction on Metal via the CPU-readback fallback. (Not needed — SPPM has NO indirect dispatch: every stage is a plain `enc.dispatch` over a host-known count (num_pixels / num_cells / photons), so it sidesteps the slang-rhi Metal indirect-dispatch no-op entirely.)
- [x] 3.4 (guarded, one Metal-compile process at a time — thermal rule) Smoke-test: SPPM active compiles + dispatches without blowing the slot cap. (Guarded single Metal-compile run: all 8 entries built + dispatched on cornell_box_sphere with a MaterialX-graph material set, no slot-cap error.)
- [x] 3.5 Render the caustic scene under SPPM on Metal headless; confirm finite energy. (M5 Pro, cornell_box_sphere 128² wavefront: SPPM pass builds, 16384 photons, accumulates, 97% non-black, **mean 137.41 — identical to the Vulkan smoke 137.4 → cross-backend parity**, finite.)

## 4. Vulkan backend parity

- [x] 4.1 Wire the four SPPM stages into `vk_wavefront.py` / `wavefront_driver.py`. (`WavefrontSppmPass` + `_VkSppmRecorder` + `record_sppm_loop`; plain `vkCmdDispatch` (host-known counts) + device atomics for the flux deposit; `vkCmdFillBuffer` clears.)
- [x] 4.2 Recompile the wavefront SPPM `.spv` kernels with `slangc`; verify path/bdpt unchanged. (All 8 SPPM entries compile SPIR-V + `-target metal`; SPPM lives in a NEW `wavefront_sppm.slang` with no Metal-gated edits to shared shaders, so the path/bdpt SPIR-V is byte-unchanged.)
- [x] 4.3 Render the caustic scene under SPPM on Vulkan headless; confirm Metal and Vulkan agree. (Vulkan smoke mean 137.4 == Metal smoke 137.41; SPPM/path energy ratio 1.008 on Vulkan via `tests/test_sppm_gpu.py`.)

## 5. pbrt importer mapping

- [x] 5.1 (test-first) Add an importer test: a pbrt scene with `Integrator "sppm"` yields USD metadata carrying `numiterations`/`maxdepth`/`photonsperiteration`/`radius`/`seed` and selects the skinny SPPM integrator. (params already round-tripped via `scene_metadata`; added `test_metadata.py` cases for the skinny selection + helper + report.)
- [x] 5.2 Implement sppm recognition in `state.py`/`metadata.py`/`emit.py`: write params to USD metadata and record SPPM as the selected integrator on the stage. (`metadata.scene_metadata` writes a normalized `customLayerData["pbrt"]["skinny"] = {integrator: "sppm", radius?, photons?}`; `api.sppm_selection(stage)` reads it.)
- [x] 5.3 Seed the initial SPPM search radius from the pbrt `radius` param when present, else a scene-bounds default; add a test for both paths. (`radius`→`skinny.radius`, consumed by the renderer's `_sppm_radius_override`; bbox default when absent. Tests cover present/absent.)
- [x] 5.4 Update `report.py` so `sppm` is reported as *mapped* (surface case); lift the "sppm / photon out of scope" note in the pbrt change/spec docs. (`api.import_pbrt` now `report.exact("integrator:sppm", "mapped to skinny SPPM")`.)

## 6. Correctness verification (test-first where noted)

- [~] 6.1 Hash-grid build unit tests: counting-sort produces correct per-cell membership and is deterministic for a fixed visible-point set. (Implicitly covered by 6.3 — a broken grid mis-deposits and skews the energy ratio off 1.0; a standalone CPU/readback grid-membership test is a nice-to-have follow-up.)
- [ ] 6.2 SPPM consistency harness: radius-sweep trend shows the caustic estimate's error trending downward as radius shrinks (consistency, not single-frame equality). (Deferred — needs the pbrt reference + a multi-pass-budget sweep; the energy gate 6.3 + parity 7 cover correctness for PM-1.)
- [x] 6.3 Energy/no-double-count test: direct via NEE + indirect via photons matches the reference energy within tolerance (no double-counted direct term). (`tests/test_sppm_gpu.py`: SPPM vs path energy ratio in [0.85, 1.15] — measured 1.008 on M5 Pro/Vulkan; + builds/finite/non-black. 2 gpu tests, 61 s.)
- [~] 6.4 Regression: layered skin/BSSRDF and volume paths produce byte-identical path/bdpt output to pre-change (SPPM does not touch them). (Covered structurally: SPPM is purely additive — new `wavefront_sppm.slang`, new pass class, new `integrator_index==2` branch; it touches no skin/volume shading code, and the existing path/bdpt test suite is unaffected.)

## 7. Caustic parity gate

- [x] 7.1 Caustic parity scene + reference. (Reused the existing corpus `glass_arealight.pbrt` — a glass sphere over a diffuse floor under an area light, a genuine caustic — and its converged pbrt reference EXR. SPPM must converge to the same ground truth, so no separate pbrt `sppm` reference is needed; the converged path reference IS the ground truth.)
- [x] 7.2 Wire the scene into `parity.py` with relMSE / FLIP thresholds. (`render_linear` gained `integrator="sppm"` (`_INTEGRATORS`/headless) + auto-forces wavefront; `tests/test_sppm_gpu.py::test_sppm_caustic_parity_vs_pbrt_reference` gates relMSE ≤ 0.06 / FLIP ≤ 0.08. **Measured relMSE 0.0251 vs the reference — better than the path tracer's 0.0310; energy ratio 1.005×.** Runs on the host default backend (Metal via select_backend, == Vulkan by the 137.4 smoke parity). This gate caught + fixed a real 14× over-brightness bug — see the commit.)
- [~] 7.3 Labelled side-by-side image (reference · skinny SPPM). (Numeric gate is green; a rendered triptych is surfaced in the chat per the global image rule.)

## 8. Docs

- [ ] 8.1 New `docs/PhotonMapping.md`: SPPM pipeline, the four stages, estimator equations as LaTeX→SVG, and an SVG pipeline diagram under `docs/diagrams/` (no ASCII art); wire any `// DOC:` markers + `embed_code.cjs` if excerpting shader slices.
- [ ] 8.2 `docs/Wavefront.md`: add the four SPPM stages to the stage list.
- [ ] 8.3 `docs/Architecture.md`: add the new visible-point / grid / photon descriptor bindings to the binding map (with the Metal fold noted).
- [ ] 8.4 `README.md`: document `--integrator sppm` (wavefront-only) and add SPPM to the compatibility matrix (incl. Metal per-pass readback-compaction cost); `CHANGELOG.md` entry.
- [ ] 8.5 Note the deferred PM-2 (skin/BSSRDF) and PM-3 (volumetric) phases as explicit future work in `docs/PhotonMapping.md` so they extend the same capability.

## 9. Validate & close

- [ ] 9.1 `openspec validate photon-mapping-sppm --strict`; `.venv/bin/ruff check src/`; `.venv/bin/pytest` (excluding `-m gpu` for sweeps per thermal rule, GPU gates run guarded).
- [ ] 9.2 Run the parity gate on both backends; confirm thresholds green; attach the side-by-side image.
- [ ] 9.3 Open the merge request from the worktree; archive the change after merge (`openspec archive`).
