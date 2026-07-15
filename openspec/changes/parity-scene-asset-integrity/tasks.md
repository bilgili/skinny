# Tasks — parity-scene-asset-integrity

## 1. Root cause (done during diagnosis)
- [x] 1.1 Bisect disney_cloud 0.075→0.584: src trees at 10542c1 and 09b77f1
      render bit-identically → not code; dangling
      `assets/light_infinite_f620_const.hdr` reference found in the untracked
      `disney_cloud.usda`; restoring the baked env restores mean 0.878×
      (recorded 0.88×).
- [x] 1.2 subsurface_infinite gate failure reproduced: spectral combo admitted
      (manifest lacks `material_class`) → renderer scene-level spectral refusal
      `SystemExit` kills the test.
- [x] 1.3 Post-fix sweep exposes `path|megakernel` vs anchor self-consistency
      0.0362/0.0554 (512 spp); bisected to merge a1e4234
      (`pbrt-subsurface-3d-walk`) — by-design 3D-walk (wavefront) vs 1D-slab
      (megakernel) split; before it mega≡wave EXACT (0.0).

## 2. Data restoration + tracking
- [x] 2.1 Restore `assets/light_infinite_f620_const.hdr` via re-import of the
      source pbrt (content-identical baked constant, verified vs sibling
      const.hdr files).
- [x] 2.2 Git-track the 173-byte const `.hdr` only (force-add); the usda stays
      untracked — it bakes a machine-absolute `.nvdb` path (codex P1: tracking
      it breaks test_corpus_scene_imports_cleanly on other checkouts).

## 3. Harness + loader fixes
- [x] 3.1 Manifest: `subsurface_infinite` += `material_class: "subsurface"`,
      per-scene `self_consistency` `mode` override 0.05/0.07 (measured
      0.0362/0.0554), re-measured `measured` 0.1197/0.1109, notes.
- [x] 3.2 `usd_loader._extract_dome_light`: stderr warning on authored-but-
      missing `texture:file` (fallback unchanged).
- [x] 3.3 Hostless integrity meta-tests in `tests/pbrt/test_matrix.py`
      (usd `texture:file` sweep + non-flat material_class cross-check).

## 4. Verification
- [x] 4.1 Hostless: `pytest tests/pbrt -m "not gpu"` green (new meta-tests
      included).
- [x] 4.2 GPU: `test_scene_matrix_gate[disney_cloud]` green at manifest spp.
- [x] 4.3 GPU: `test_scene_matrix_gate[subsurface_infinite]` green at manifest
      spp.
- [x] 4.4 Pre-merge reviews: design subagent (P1 spec-scenario loss + 4 P2, folded)
      and codex gpt-5.5 (P1 tracked-usda portability, folded by untracking).
- [ ] 4.5 CHANGELOG + memory; `openspec validate`; merge; archive.
