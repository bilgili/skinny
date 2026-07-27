# Tasks: renderer-pure-core-extraction

## 1. Baseline

- [ ] 1.1 Capture every module-scope constant value and every packer's output
      bytes for a spread of inputs. This is the identity target.
- [ ] 1.2 List every importer of these symbols across `src/` and `tests/`
      (35 references to `pack_flat_material` alone, 9 files).

## 2. Extract, cluster by cluster

- [ ] 2.1 Material + std-surface packing and stride constants.
- [ ] 2.2 Camera math + `CameraBase`/`OrbitCamera`/`FreeCamera` +
      `_orbit_distance_cap`; repoint `debug_viewport.py:30`.
- [ ] 2.3 Film IO: `_write_exr`, `_write_hdr_rgbe`, `FilmParameters`.
- [ ] 2.4 SPPM photon budget + group PMF.
- [ ] 2.5 `TexturePool` (takes a resource module; hostless test uses a fake —
      share the fake with `renderer-gpu-resource-set`).
- [ ] 2.6 Small helpers incl. `_hashable_value`; repoint
      `ui/qt/windows/bxdf.py:22`.
- [ ] 2.7 Decide and record: does `SkinParameters` get its own module?

## 3. Enforce

- [ ] 3.1 Re-export all moved names from `skinny.renderer` for source callers.
- [ ] 3.2 Repoint every **test** import to the new modules — a test importing
      `skinny.renderer` does not demonstrate hostlessness.
- [ ] 3.3 Subprocess import gate: each new module imports with `vulkan`
      unavailable. Same shape as the existing no-Qt-import check.
- [ ] 3.4 Check for import cycles among the new modules.

## 4. Gates

- [ ] 4.1 Constants and packed bytes identical to 1.1.
- [ ] 4.2 `ruff check src/`; full hostless `pytest`.
- [ ] 4.3 Run the affected tests on a host with no Vulkan SDK and confirm they
      **execute** rather than skip.
- [ ] 4.4 GPU smoke: one Metal render, one Vulkan render, images unchanged.
- [ ] 4.5 Docs: `docs/Architecture.md` module map, carve-out section;
      `docs/PythonAPI.md` if any re-export is dropped.
- [ ] 4.6 `openspec validate renderer-pure-core-extraction --strict`.
