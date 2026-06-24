## Why

In `assets/three_materials_demo.usda` the marble sphere renders **totally
broken**: a dark, near-black, blue-tinted, speckled ball that does not respond to
the environment light — while the wood and brass spheres render correctly. The
breakage is identical on every backend (Vulkan/Metal), execution mode
(megakernel/wavefront) and integrator (Path/BDPT), which points at a
backend-agnostic, host-side material-loading bug rather than a shader bug.

Root cause: the marble material is a plain Autodesk `standard_surface` with
`subsurface = 0.4` (a *diffuse subsurface-scattering weight*, no interior medium).
The loader's `_derive_opacity_from_subsurface` bridge — added so a pbrt
`Material "subsurface"` (a transmissive dielectric boundary that routes to
`MATERIAL_TYPE_SUBSURFACE`) opens the flat path's refraction gate — fires on
**any** `subsurface > 0`. So it forces `opacity = 0` on the marble. The marble
carries no `subsurface_sigma_a/σ_s`, so `renderer._material_is_subsurface`
classifies it `MATERIAL_TYPE_FLAT`; the flat path then sees `opacity < 1` and
treats it as a clear dielectric, refracting the near-black environment → the
"broken" look.

Evidence (Vulkan megakernel, 200², path):
- before: marble `opacity = 0` in the loaded overrides; linear RGB
  ≈ `(0.00058, 0.00058, 0.00084)`, unresponsive to a 16× env boost.
- the host graph params and the GPU struct read were both verified correct
  (scalar-layout offsets match; the marble nodegraph returns a grey-dominant
  albedo), ruling out the MaterialX codegen / param packing.

The opacity bridge must agree with the subsurface material-type routing: only a
material that actually carries an interior medium (and therefore runs the
volumetric subsurface walk) may have its refraction gate opened.

## What Changes

- `_derive_opacity_from_subsurface` (`usd_loader.py`) gains a medium gate: it sets
  `opacity = 0` **only** when a non-zero `subsurface_sigma_a`/`subsurface_sigma_s`
  is present, mirroring `renderer._material_is_subsurface`. A `subsurface` weight
  with no interior medium leaves `opacity` untouched (stays opaque diffuse).
- New helper `_has_subsurface_medium(overrides)` encapsulates the σ_a/σ_s test
  (shared phrasing with the renderer's classifier).
- No shader / GPU change — the fix is in host material loading, so every
  integrator, execution mode and backend inherits the corrected opacity.
- Genuine pbrt subsurface materials are unaffected: they carry σ_a/σ_s (and set
  `subsurface = 1`), so the gate still opens their refraction boundary
  (`test_subsurface_roundtrip_equivalent` stays green).

## Impact

- Affected specs: `subsurface-scattering` (adds the opacity-gate requirement).
- Affected code: `src/skinny/usd_loader.py` (both intake paths share the helper).
- Tests: new `test_subsurface_opacity_gate_requires_medium`
  (`tests/test_struct_layout.py`); existing subsurface roundtrip / coeffs /
  struct-layout suites unchanged.
