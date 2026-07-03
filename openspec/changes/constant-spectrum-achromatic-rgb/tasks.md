# Tasks — constant-spectrum-achromatic-rgb

## 1. Core fix (hostless)

- [x] 1.1 Add the constant-SPD shortcut to `sampled_spectrum_to_rgb`
      (`src/skinny/pbrt/spectra.py`): if all authored sample values are exactly
      equal, return `[v, v, v]` (both reflectance and illuminant modes);
      update the module/function docstrings.
- [x] 1.2 Unit tests in `tests/pbrt/test_spectra.py`: constant reflectance →
      exact `[v,v,v]`; constant illuminant → exact `[v,v,v]`; tighten
      `test_sampled_reflectance_grey_is_neutral` to exact; colored spectrum
      bit-identical to pre-change projection; near-constant
      (`10` vs `10.000001`) takes projection path.
- [x] 1.3 End-to-end import assertions: `"spectrum sigma_s" [200 10 900 10]`
      through `media.py` yields achromatic coefficients; constant
      `"spectrum reflectance"` through `materials.py` yields achromatic
      reflectance (extend existing media/materials tests).
- [x] 1.4 Run hostless suites: `tests/pbrt/test_spectra.py`,
      `test_media.py`, `test_materials.py`, `test_parser.py`,
      plus `ruff check src/`.

## 2. Regenerate baked assets

- [x] 2.1 Re-import `disney-cloud.pbrt` and `bunny-cloud.pbrt` (pbrt sources
      in `~/projects/pbrt-v4-scenes/`) with `import_pbrt` to refresh
      `assets/disney_cloud.usda` and `assets/bunny_cloud.usda`; verify the
      baked `volume_sigma_s`/`volume_sigma_a` (and disney ground reflectance)
      are now achromatic and everything else in the diff is unchanged.
      DONE: disney `sigma_s [4.801,3.799,3.633]→[4,4,4]` (only line changed);
      bunny `sigma_s→[10,10,10]`, `sigma_a→[0.5,0.5,0.5]` (only sigma lines
      changed, 207-line file otherwise identical). Main-checkout
      `assets/disney_cloud.usda` synced too.
- [~] 2.2 `clouds.pbrt` uses `"string type" "cloud"` (procedural medium), NOT
      nanovdb — unsupported by this branch's importer (`_SUPPORTED_HETEROGENEOUS
      = {"nanovdb"}`), so a re-import drops the medium (regression). Procedural
      "cloud" support lives on the unmerged sibling worktree
      `pbrt-cloud-procedural-medium`; `clouds` is NOT in the corpus manifest.
      The shared `spectra.py` fix already benefits `clouds` when that feature
      regenerates its own asset. SKIPPED here (out of scope on this branch).

## 3. GPU re-measure (Metal backend, headless env per CLAUDE.md)

- [x] 3.1 Re-render `disney_cloud` and `bunny_cloud` via the parity harness
      (manifest res/spp) against the unchanged pbrt refs; collect the
      `metrics.compute_metrics` battery for the swept combos.
- [x] 3.2 Gate direction: new relMSE/FLIP ≤ recorded `measured` values for
      every combo. If any metric regresses, stop and diagnose — do not raise
      a baseline or tolerance.
- [x] 3.3 Re-verify mega≡wave self-consistency on both scenes (EXACT parity
      expected, as recorded).
- [x] 3.4 Update `tests/pbrt/corpus/manifest.json`: new `measured` values and
      `notes` (remove the RGB-vs-spectral tint from the recorded divergence
      wording where it no longer applies).
- [x] 3.5 Render labelled before/after/pbrt-ref side-by-side comparisons at a
      shared tonemap and surface the images in the reply.

## 4. Full gates + docs

- [x] 4.1 Full hostless parity suite: `tests/pbrt/test_matrix.py`,
      `test_metrics.py`, `test_parity.py -m "not gpu"`.
- [x] 4.2 GPU matrix gate on the volume scenes
      (`PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest
      tests/pbrt/test_parity.py -k matrix` scoped appropriately).
- [x] 4.3 Docs sweep: update any Markdown that documents the equal-energy
      tint behaviour (spectra docstring already in 1.1; check
      `docs/Architecture.md` parity/import sections and CLAUDE.md/README
      compatibility-matrix notes for stale wording about the cloud
      divergences).
- [x] 4.4 `openspec validate constant-spectrum-achromatic-rgb`; commit.
