## Why

In `assets/glass_caustics_test.usda` the **right** glass sphere (`MtlxGlassSphere`,
bound to the MaterialX `standard_surface` `Glass`) renders **opaque** — a solid
ball, not transparent — while the **left** sphere (`GlassSphere`, a
UsdPreviewSurface with `opacity = 0`) renders as correct glass. The breakage is
identical on every backend / execution mode / integrator, which points at a
host-side material-loading bug, not a shader bug.

Root cause: skinny's flat path only refracts through surfaces whose `opacity < 1`
(`flat_material.slang`). `standard_surface`/OpenPBR express glass via a
`transmission` weight, so the loader bridges `transmission → opacity` in
`_derive_opacity_from_transmission` (`usd_loader.py`). That bridge **skips when an
`opacity` is already authored** — a guard meant to preserve a genuine OpenPBR
cutout (`geometry_opacity`) alpha. But a `standard_surface` shader **always
authors `opacity`** (its MaterialX default is the fully-opaque `(1, 1, 1)`), so on
the usdMtlx-plugin intake path (`_extract_material`) the guard fires on the
default and leaves `opacity = (1, 1, 1)`. The transmission weight is dropped and
the sphere stays solid.

Evidence (host-side override dump, `_extract_material`):
- `/Scene/GlassMat` (UsdPreviewSurface): `opacity = 0.0` → refracts → glass. ✓
- `/MaterialX/Materials/Glass` (standard_surface): `transmission = 1.0`,
  `opacity = Gf.Vec3f(1, 1, 1)` → the bridge skipped → renders opaque. ✗

The two intake paths disagree: the `.mtlx` API fallback (`_load_mtlx_materials`)
**pops** `transmission` and sets `opacity = 1 − transmission` unconditionally, so
its glass is transparent; the plugin path lets the default-opaque authored opacity
block the bridge. `opacity = 1` is the default-opaque value and must NOT block the
transmission bridge — only a real cutout (`opacity < 1`) should win over
transmission, mirroring the marble subsurface-opacity gate that aligned the two
subsurface intake paths.

## What Changes

- `_derive_opacity_from_transmission` (`usd_loader.py`) no longer skips on **any**
  authored `opacity`. It skips only when the authored opacity is a genuine cutout
  (some channel `< 1`). A default-opaque `opacity` (scalar `1` or per-channel
  `(1, 1, 1)`) no longer blocks the `transmission → opacity` refraction bridge, so
  a `standard_surface`/OpenPBR glass refracts.
- New helper `_opacity_is_fully_opaque(value)` encapsulates the scalar/color
  "default opaque (≥ 1)" test.
- Host-side material-loading fix — no shader / GPU change, so every integrator,
  execution mode and backend inherits the corrected opacity.
- A genuine cutout (`opacity < 1`, OpenPBR `geometry_opacity`) is still preserved
  over transmission, unchanged.

## Impact

- Affected specs: `flat-bsdf-lobes` (adds the transmission-opacity-gate
  requirement).
- Affected code: `src/skinny/usd_loader.py`.
- Tests: new `test_transmission_opacity_gate_ignores_default_opaque`
  (`tests/test_struct_layout.py`) and a scene-level
  `test_glass_caustics_both_spheres_transparent`
  (`tests/test_glass_caustics_test.py`) that loads
  `assets/glass_caustics_test.usda` and asserts both glass materials reach the
  flat refraction gate (`opacity < 1`). Existing transmission/opacity-bridge and
  struct-layout suites unchanged.
