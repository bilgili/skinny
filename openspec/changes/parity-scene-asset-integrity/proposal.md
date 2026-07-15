# Parity scene-asset integrity (disney_cloud + subsurface_infinite gate repair)

## Why

Two parity-matrix gates fail on main for reasons that have nothing to do with
the renderer's current code — both were diagnosed by bisection to be **harness /
scene-data integrity holes**, not transport regressions:

1. **disney_cloud** (pbrt-truth relMSE 0.075 → 0.584): the checked-in-by-hand
   (but *git-untracked*) `assets/disney_cloud.usda` references a baked constant
   sky env side-file `light_infinite_f620_const.hdr` (blue sky,
   L = [0.03, 0.07, 0.23]) that was deleted from `assets/` sometime after the
   Jul 3 baseline measurement. `usd_loader._extract_dome_light` **silently**
   returns `None` for a missing texture file, so the render fell back to the
   default gray studio env — a completely different illuminant — and the gate
   read the shift as a renderer regression. Jul-3 src and current src render
   the scene **bit-identically**; only the on-disk asset changed.

2. **subsurface_infinite** (gate dies with `SystemExit` in 18 s): the manifest
   entry predates `material_class` and defaults to `"flat"`, so the spectral
   envelope admits spectral combos for a subsurface scene; the renderer's
   scene-level spectral refusal (`raise SystemExit`, renderer.py) then kills
   the test outside the harness's `except Exception` guard.

Repairing #2 exposed a third, recorded-by-design divergence: since change
`pbrt-subsurface-3d-walk` (Jun 26) the **wavefront** subsurface interior walk is
a true 3D per-segment walk while the **megakernel** keeps the watchdog-safe 1D
slab — so `path|megakernel` vs the wavefront anchor on a subsurface-dominated
scene is *not* noise-limited (measured relMSE 0.0362 / FLIP 0.0554 at 512 spp
vs the default `mode` tolerance 0.02/0.03). The suite's other subsurface scenes
sit below the default tolerance only because their SSS contribution is small.

## What Changes

- **Restore the lost assets** (data, main checkout): re-import the pbrt scenes
  and restore `assets/light_infinite_f620_const.hdr` **and**
  `assets/bunny_cloud.usda` (deleted outright by the same untracked-asset loss —
  its `test_scene_source_resolves[bunny_cloud]` is red on main); git-track the
  173-byte `assets/light_infinite_f620_const.hdr` (force-add past the `*`
  gitignore) so the file that actually vanished is restorable and its deletion
  is visible to git. The usda itself stays **untracked** like its siblings: it
  bakes a machine-absolute `OpenVDBAsset` path to the external 47 MB `.nvdb`,
  so tracking it would make `test_corpus_scene_imports_cleanly[disney_cloud]`
  fail on any checkout without that file (codex P1). A deleted usda fails
  loudly (`test_scene_source_resolves`), unlike the silently-swapped
  illuminant this change fixes. **Recorded
  gap**: `bunny_cloud.usda`/`clouds.usda` reference 33 MB
  `light_infinite_*_env.hdr` maps (tracking the usda without its env would
  leave every worktree permanently red on the integrity sweep), and
  `bathroom.usda` (41 MB) / `dragon_sss.usda` (349 MB) / the external `.nvdb`
  are too heavy — all stay untracked per the repo's assets convention. For
  those the hostless integrity sweep + the loud loader warning are the
  mitigations, and their teeth are on the GPU dev box where the files live.
- **Loud loader fallback**: `usd_loader._extract_dome_light` warns to stderr
  when a DomeLight authors a `texture:file` that does not resolve to an
  existing file (fallback behavior unchanged).
- **Manifest**: `subsurface_infinite` declares `material_class: "subsurface"`
  (excludes spectral combos exactly like `dragon` / `mat_subsurface`), records
  the 1D-slab-vs-3D-walk mode divergence as a per-scene `self_consistency`
  override (`mode: relmse 0.05 / flip 0.07`, measured 0.0362/0.0554), and
  re-measures `measured` (anchor relMSE 0.0792 → 0.1197 / FLIP 0.0894 → 0.1109,
  the recorded wavefront 3D-walk shift; documentation-only — `measured` is not
  a `SceneSpec` field and does not reach the gate; `relmse_tol`/`flip_tol`
  unchanged). The other subsurface combos measured under their default classes:
  `bdpt|megakernel` 0.0362/0.0554 vs `integrator` tol 0.06/0.06 (passes;
  inherits the same slab delta), `sppm|wavefront` / `bdpt|wavefront` /
  `restir` 0.0000 (wavefront non-flat first hits share the path fallback —
  bit-identical).
- **Hostless integrity meta-test** (`tests/pbrt/test_matrix.py`): every
  manifest-referenced `.usda` asset that exists on disk must have all its
  `texture:file` side-references resolvable (catches the disney_cloud failure
  class at hostless-test time); every corpus `.pbrt` scene source that authors
  `Material "subsurface"` / a named medium must declare a non-flat
  `material_class` (catches the subsurface_infinite class).

## Impact

- Affected specs: `render-parity-matrix` (scene-data integrity + recorded
  mode-divergence requirements).
- Affected code: `src/skinny/usd_loader.py` (warning only),
  `tests/pbrt/corpus/manifest.json`, `tests/pbrt/test_matrix.py`,
  `assets/disney_cloud.usda` + `assets/light_infinite_f620_const.hdr`
  (newly tracked), CHANGELOG.
- No renderer/shader behavior change; no SPIR-V change.
