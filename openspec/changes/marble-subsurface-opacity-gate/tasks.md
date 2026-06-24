## 1. Reproduce & root-cause

- [x] 1.1 Render `three_materials_demo.usda` headless; confirm marble broken on
  Vulkan megakernel + wavefront + BDPT (backend/mode/integrator-agnostic).
- [x] 1.2 Rule out MaterialX codegen / param packing: dump marble `uniform_block`
  defaults (correct) and disassemble the GPU `Load<GraphParams_Marble_3D>` struct
  offsets (scalar-layout, match host) → graph eval is correct given P.
- [x] 1.3 Dump loaded marble overrides → `opacity = 0` from the subsurface bridge;
  confirm `material_type = FLAT` (no σ_a/σ_s) → flat path refracts it as glass.

## 2. Fix

- [x] 2.1 Add `_has_subsurface_medium(overrides)` (non-zero σ_a/σ_s), mirroring
  `renderer._material_is_subsurface`.
- [x] 2.2 Gate `_derive_opacity_from_subsurface` on `_has_subsurface_medium`; both
  intake paths (native-USD `_extract_material`, `.mtlx` fallback) share the helper.

## 3. Verify

- [x] 3.1 TDD: `test_subsurface_opacity_gate_requires_medium` (red → green):
  marble weight stays opaque; genuine σ-medium still → `opacity = 0`; all-zero σ
  stays opaque; authored opacity wins.
- [x] 3.2 No regression: `test_struct_layout.py`, `test_subsurface_coeffs.py`,
  `test_mtlx_roundtrip.py::test_subsurface_roundtrip_equivalent` /
  `…coateddiffuse…` green (pre-existing `test_aggregator_emits_all_graphs` and
  `test_glass_overrides_equivalent_via_usdmtlx_plugin` failures are unrelated and
  fail on base too).
- [x] 3.3 GPU render: marble = grey marble with blue veining, responds to env —
  Vulkan megakernel (Path + BDPT) and Vulkan wavefront (Path).
- [x] 3.4 GPU render on native Metal (megakernel, guarded) confirms backend parity.
- [x] 3.5 `ruff check src/skinny/usd_loader.py` clean.

## 4. Docs

- [x] 4.1 Note the gate in the subsurface section of the docs / spec delta.
