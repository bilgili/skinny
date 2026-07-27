# Change: renderer-pure-core-extraction

## Why

`renderer.py:16` is a module-scope `import vulkan as vk`. Above the `Renderer`
class sit **1,330 lines of device-free code** — every one of them unreachable
without the Vulkan SDK, on a machine whose default backend is Metal:

- SPPM photon budget and group PMF math (`:104-291`)
- small pure helpers, including `_hashable_value`, re-exported to the BXDF dock
  (`:292-500`)
- `TexturePool` (`:501-608`)
- `pack_flat_material`, `pack_std_surface_params`,
  `pack_std_surface_params_msl` (`:609-988`)
- `SkinParameters` and its std140 `pack` (`:989-1067`)
- camera math and `CameraBase` / `OrbitCamera` / `FreeCamera`, imported by
  `debug_viewport.py:30` (`:1068-1370`)
- the EXR and Radiance writers (`:1371-1415`)
- `FilmParameters` (`:1416-1434`)
- every stride constant: `FLAT_MATERIAL_STRIDE`, `INSTANCE_STRIDE`,
  `DISTANT_LIGHT_STRIDE`, `SPHERE_LIGHT_STRIDE`, `SPECTRAL_EMITTER_STRIDE`, …

**35 test references** to `pack_flat_material` alone, across 9 files, reach
these through `from skinny.renderer import …`. All of them skip when the SDK is
absent — and `tests/test_camera_placement.py` carries an explicit
`_have_renderer()` skip for exactly this reason. A prior incident is recorded
in the project's own history: the sandbox strips `DYLD_LIBRARY_PATH`, the
`vulkan` import fails, and the tests skip **silently** rather than fail.

The consequence is structural, not cosmetic: the packers that produce the bytes
the Metal backend uploads can only be tested on a Vulkan-capable host.

## What Changes

- Move the module-scope device-free code out of `renderer.py` into modules that
  import no GPU package, and re-export the names from `renderer` so no call
  site changes.
- Split by cluster, not into one dumping ground: material packing, camera,
  film/image writers, SPPM budget, texture pool, small helpers.
- Repoint the tests that reach these symbols at the new modules, so their
  hostlessness is enforced rather than incidental.
- Add a gate that fails if any of the new modules acquires a GPU import, in the
  same shape as the existing "no toolkit import" subprocess check used for the
  render-session module.
- Pure move. No logic change, no signature change, no behaviour change.

## Capabilities

### Modified Capabilities

- `renderer-module-structure`: adds a carve-out stage — the module-scope pure
  core leaves `renderer.py` and becomes importable without a GPU package, under
  the existing bit-identity requirement for carve-out stages.

## Impact

- New: several small device-free modules under `src/skinny/` (material packing,
  camera, film IO, SPPM budget, texture pool), plus a hostless import gate.
- Modified: `src/skinny/renderer.py` (−~1,330 lines, gains re-export imports),
  the 9 test files that import these symbols, `debug_viewport.py`'s camera
  import, `ui/qt/windows/bxdf.py`'s `_hashable_value` import.
- Unchanged: every signature, every constant value, every packed byte.
- **Enables**: `flat-material-field-table`'s transposition gate, which is
  otherwise only enforced on Vulkan-capable hosts — the wrong place, since the
  Metal path packs the same bytes.
- **Ordering**: cheapest change in the set and touches only the top of
  `renderer.py`; land it before `renderer-gpu-resource-set` and
  `frame-plan-split`, which edit the middle and bottom.
- Docs: `docs/Architecture.md` module map; `docs/PythonAPI.md` if any re-export
  is dropped rather than kept.
