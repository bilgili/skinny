# Tasks

## 1. Failing gate (TDD)
- [ ] 1.1 gpu test: `cornell_box_python_material.usda` under wavefront BDPT + SPPM
  renders non-black ≈ wavefront-path anchor. FAILS pre-fix (black).

## 2. Un-gate the shader fallback
- [ ] 2.1 `wfBdptWalk` / `wfBdptGenEye` (wavefront_bdpt.slang) + `wfSppmEye`
  (wavefront_sppm.slang): drop the `SUBSURFACE || SKIN` gate → path-integrate every
  non-flat first hit.

## 3. Bounded eye submit on Metal
- [ ] 3.1 Add `flush()` to the recorder protocol: `_MetalWavefrontRecorder` /
  `_MetalSppmRecorder` → `enc.flush()`; `_VkPathRecorder` / `_VkSppmRecorder` +
  any others → no-op. Add to the `WavefrontRecorder` protocol type.
- [ ] 3.2 `record_bdpt_loop` / `record_sppm_loop`: new `bound_heavy_eye` param;
  when set, `rec.flush()` per eye tile (BDPT: end of each tile; SPPM: after each
  phase-1 eye tile).
- [ ] 3.3 Host flag: renderer computes `has_heavy_nonflat` (VOLUME/PYTHON in
  `_material_types`) and threads it to both Metal record loops.

## 4. Verify
- [ ] 4.1 Render `cornell_box_python_material` wave bdpt/sppm on Metal → shaded ≈
  path. Parity gate green.
- [ ] 4.2 Volume smoke: a `clouds`/`disney_cloud` scene under wave bdpt renders
  (eye-visible volume shaded, not black) and completes.
- [ ] 4.3 **Kill harness**: `tests/test_metal_cleanup.py -m gpu` green (SIGKILL
  mid-render on a heavy-fallback frame → GPU usable).
- [ ] 4.4 Regression: flat scene wave-bdpt byte-identical; subsurface/skin
  (terminal) unchanged; Vulkan slangc compiles.

## 5. Docs + review + merge
- [ ] 5.1 `docs/Wavefront.md`, README compat, corpus manifest.
- [ ] 5.2 codex review; fold findings.
- [ ] 5.3 `openspec validate`; merge; archive.
