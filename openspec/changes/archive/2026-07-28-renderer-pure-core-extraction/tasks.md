# Tasks: renderer-pure-core-extraction

## 1. Baseline

- [x] 1.1 Capture every module-scope constant value and every packer's output
      bytes for a spread of inputs. This is the identity target.
- [x] 1.2 List every importer of these symbols across `src/` and `tests/`
      (35 references to `pack_flat_material` alone, 9 files).

## 2. Extract, cluster by cluster

- [x] 2.1 Material + std-surface packing and stride constants.
- [x] 2.2 Camera math + `CameraBase`/`OrbitCamera`/`FreeCamera` +
      `_orbit_distance_cap`; repoint `debug_viewport.py:30`.
- [x] 2.3 Film IO: `_write_exr`, `_write_hdr_rgbe`, `FilmParameters`.
- [x] 2.4 SPPM photon budget + group PMF.
- [x] 2.5 `TexturePool` (takes a resource module; hostless test uses a fake —
      share the fake with `renderer-gpu-resource-set`).
- [x] 2.6 Small helpers incl. `_hashable_value`; repoint
      `ui/qt/windows/bxdf.py:22`.
- [x] 2.7 Decide and record: does `SkinParameters` get its own module?
      **Yes** — `skin_params.py`. It is the skin path's own record with its own
      documented std140 layout; the flat-material packers never touch it.

## 3. Enforce

- [x] 3.1 Re-export all moved names from `skinny.renderer` for source callers.
- [x] 3.2 Repoint every **test** import to the new modules — a test importing
      `skinny.renderer` does not demonstrate hostlessness.
- [x] 3.3 Subprocess import gate: each new module imports with `vulkan`
      unavailable. Same shape as the existing no-Qt-import check.
- [x] 3.4 Check for import cycles among the new modules.

## 4. Gates

- [x] 4.1 Constants and packed bytes identical to 1.1.
- [x] 4.2 `ruff check src/`; full hostless `pytest`.
- [x] 4.3 Run the affected tests on a host with no Vulkan SDK and confirm they
      **execute** rather than skip.
- [x] 4.4 GPU smoke: one Metal render, one Vulkan render, images unchanged.
- [x] 4.5 Docs: `docs/Architecture.md` module map, carve-out section;
      `docs/PythonAPI.md` if any re-export is dropped.
- [x] 4.6 `openspec validate renderer-pure-core-extraction --strict`.
