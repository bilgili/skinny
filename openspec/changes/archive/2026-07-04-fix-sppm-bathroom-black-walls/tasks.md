# Tasks — fix-sppm-bathroom-black-walls

(Tasks reshaped after investigation: the proposed graph-material eye-pass gap
was disproved — graph eval already works in the SPPM eye pass. True root cause:
`sppmLoadMaterial` rebuilt the deposit-time material without the Stage-2 rich
inputs → undefined values → zero photon flux scene-wide. Artifacts rewritten.)

## 1. Reproduce & root-cause

- [x] 1.1 Worktree off `main`; headless bathroom renders path vs SPPM (Metal
      wavefront, 64 spp): SPPM black_frac 0.35 vs path 0.01 — reproduced
- [x] 1.2 Mechanism pinned by GPU visible-point readback: albedo image fully
      textured (graph eval CORRECT — original hypothesis disproved), ld image =
      direct-only interior, `τ == 0` at 100% of VPs on bathroom AND
      cornell_box_sphere → photon deposit flux-dead globally
- [x] 1.3 Cause isolated: `FlatHitMat` Stage-2 rich inputs
      (`transmissionColor`/`specularColor`/`diffuseRoughness`, af4ffb5) have no
      VP slot; `sppmLoadMaterial` left them undefined at deposit `evaluate()`.
      Defaults-only patch flipped cornell τ 0 → mean 8.51 (81% VPs non-zero),
      bathroom black_frac 0.35 → 0.00 — confirmed
- [x] 1.4 Design decision: store the three fields in the VP (bathroom authors
      `specular_color` ×25 / `transmission_color` ×16 → defaults-only would
      bias SPPM vs path); artifacts updated to the true root cause

## 2. Fix

- [x] 2.1 `sppm_state.slang`: `VisiblePoint` grows
      transmissionColor/specularColor/diffuseRoughness (scalar 152→180 B,
      MSL 192→240 B; struct docs updated)
- [x] 2.2 `wavefront_sppm.slang`: `sppmStoreVisiblePoint` stores the fields;
      `sppmLoadMaterial` rebuilds them
- [x] 2.3 `wavefront_layout.py`: `VISIBLE_POINT_FIELDS` mirror + stride
      comments (both backends size from it; Metal asserts reflected stride)
- [x] 2.4 Stride locks in `tests/test_sppm_state.py` → 180/240
- [x] 2.5 NEW hostless completeness locks: every `FlatHitMat` field (exempt:
      `emission`) must have a VP slot + `sppmStoreVisiblePoint` write +
      `sppmLoadMaterial` rebuild

## 3. Verify

- [x] 3.1 Bathroom A/B re-render: walls lit, black_frac 0.35 → 0.00; labelled
      path · before · after grid shown
- [x] 3.2 Harness-spec re-measure (256², 128 spp): `sppm_vs_path` relMSE
      64.97 → 2.59, MSE 10.42 → 0.297, linear-mean ratio 1.007; manifest
      `measured` updated + notes; `baselines` untouched (never raised),
      bathroom stays known_divergent
- [x] 3.3 Rebased onto latest main (6f8a2d3: binding-26 fix + hostless-repair
      merges); hostless sweep 610 passed / 0 failed
- [x] 3.4 SPPM GPU suite 3/3 green on rebased base — raw-Vulkan (MoltenVK)
      finite + energy-in-band, Metal caustic-parity-vs-pbrt; removed the
      re-armed energy gate's `xfail(strict)` marker as it demanded
- [x] 3.5 Metal dispatch hygiene: kill harness 13 hostless + 3 gpu green
- [x] 3.6 `ruff check src/` clean
- [x] 3.7 Cross-backend spot check covered by 3.4 (energy gate runs the raw
      Vulkan context; caustic gate runs backend-select → Metal)

## 4. Docs & spec hygiene

- [x] 4.1 `docs/PhotonMapping.md` — VisiblePoint listing + stride, mirror-as-
      contract paragraph; `CHANGELOG.md` entry; manifest notes
- [x] 4.2 Artifacts (proposal/design/specs/tasks) rewritten to the true root
      cause; `openspec validate` green
- [x] 4.3 Committed on worktree branch `worktree-fix-sppm-bathroom-black-walls`
      (46955b6 fix + 9b159c5 test/docs), rebased on main 6f8a2d3; merge
      decision with user
