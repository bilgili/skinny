# Change: choice-table-wavefront-owners

## Why

`choice-table-owners` gave each host-side render axis one owner. The wavefront
layer has the same shape one level down, and this change owns it.

- **Kernel entry-point names**: 33 of the 34 wavefront kernel names are written
  in three files — `wavefront_driver.py` (the dispatch loop), `vk_wavefront.py`
  and `metal_wavefront.py` (the per-backend pipeline `entries` lists). A rename
  in the shader plus one backend, but not the other, is a runtime dispatch
  failure on that backend, not a build failure.
- **Shared pass constants**: 14 class constants — `MAX_BOUNCES`, `STREAM_CAP`,
  `BDPT_MAX_VERTS`, `VERTEX_STRIDE`, `AUX_STRIDE`, `EYE_BOUNCES`,
  `LIGHT_BOUNCES`, `WALK_MODES`, `RESERVOIR_STRIDE`, `REC_VERTEX_STRIDE`, and the
  ReSTIR `DEFAULT_CONFIG` — are duplicated between the Vulkan and Metal pass
  classes with no test pinning the pair equal.

## What Changes

- The wavefront kernel entry-point names get one backend-neutral table in
  `wavefront_driver.py`, imported by the driver and both backend pass modules.
- The pass constants that must be equal across the backends get one home.
- The constants that legitimately differ per backend (the record-stack sizing
  formula, the Metal rebuild-key elements) stay separate but get a test that
  pins the pair and states the reason.

## Capabilities

### New Capabilities

- `wavefront-kernel-ownership`: one owner for the wavefront kernel entry-point
  names and the shared wavefront pass constants. `wavefront-execution` behaviour
  is unchanged — the dispatched kernel names and constant values are
  byte-identical to before.

## Impact

- Modified: `src/skinny/wavefront_driver.py`, `src/skinny/vk_wavefront.py`,
  `src/skinny/metal_wavefront.py`, and a new pin/golden test.
- Large mechanical churn, no runtime behaviour change (the strings and values do
  not change). Gated by a dual-backend wavefront GPU smoke.
- Docs: `docs/Wavefront.md` gains the kernel-name table.
