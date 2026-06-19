## 1. Setup

- [x] 1.1 Worktree off `main` (per CLAUDE.md); headless env (`./bin/python3.13`, `VULKAN_SDK`, `DYLD_LIBRARY_PATH`). slangc on PATH.
- [x] 1.2 Add `sppmGlossyContinueRoughness` to the FrameConstants tail (`common.slang` + `_FC_SCALAR_FIELDS` in `renderer.py` + `_pack_uniforms`); bump the Vulkan scalar-blob pin (540→544) and re-pin the MSL fc size in `tests/test_metal_msl_uniform_offsets.py`.
- [x] 1.3 Add `--sppm-glossy-roughness` to `cli_common.py` (default tuned for polished metals) + front-end persistence.

## 2. Shader

- [x] 2.1 (test-first) Capture the current brass-reflection-absent state as a baseline metric (three_materials_demo brass-region energy under SPPM vs path) before changing the shader.
- [x] 2.2 `wfSppmEye`: derive the sampled lobe's roughness, add the `glossyContinue` predicate, and route glossy-continue through the existing carrier branch (no VP stored). Threshold `0` ⇒ delta-only (PM-1 behavior).
- [x] 2.3 `wfSppmPhotonTrace`: treat a glossy-continued vertex as specular (no deposit), keeping the direct/indirect split disjoint.
- [x] 2.4 Recompile SPPM `.spv` (slangc) + Metal target; confirm path/bdpt SPIR-V byte-unchanged.

## 3. Verification

- [x] 3.1 three_materials_demo A/B: brass reflects the wood/marble spheres under SPPM with the default threshold; reflection trends toward the path reference as passes accumulate. Labelled side-by-side image artifact.
- [x] 3.2 Regression: `sppmGlossyContinueRoughness == 0` reproduces the PM-1 caustic parity result (relMSE within tolerance of the merged baseline) on both backends.
- [x] 3.3 Energy/no-double-count: SPPM-vs-path energy ratio stays in the PM-1 band with glossy continuation on.
- [x] 3.4 Both backends green (Vulkan + guarded Metal — SPPM compiles only the small wavefront kernels, `MIN_FREE_GB=6` safe).

## 4. Docs & close

- [x] 4.1 `docs/PhotonMapping.md`: glossy-continuation section + the limitation it lifts + the deferred final-gather note. `README.md` flag. `CHANGELOG.md`.
- [x] 4.2 `openspec validate sppm-glossy-final-gather --strict`; ruff; pytest (`-m "not gpu"` for sweeps, GPU gates guarded).
- [ ] 4.3 MR from the worktree; archive after merge.
