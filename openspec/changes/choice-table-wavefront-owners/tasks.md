# Tasks: choice-table-wavefront-owners

## 1. Baseline (recorded in `choice-table-owners` task 1.3)

- [ ] 1.1 Confirm the kernel-name inventory: the 34 entry-point names, and which
      of the driver / `vk_wavefront` / `metal_wavefront` writes each. Confirm the
      14 duplicated pass constants and mark each must-be-equal vs per-backend.

## 2. Kernel-name owner

- [ ] 2.1 Declare one module-level constant per kernel in `wavefront_driver.py`.
- [ ] 2.2 Repoint the driver's dispatch calls to the constants.
- [ ] 2.3 Repoint both backends' `entries` lists to import and use the constants.
- [ ] 2.4 Golden test: each constant equals its historical string; no
      kernel-name string literal remains in the three modules.
- [ ] 2.5 Negative control: rename a constant and confirm the backends fail at
      import, not at render.

## 3. Shared / pinned constants

- [ ] 3.1 Move the must-be-equal constants (bounce counts, `WALK_MODES`, stream
      caps, `DEFAULT_CONFIG`) to one home both pass classes read.
- [ ] 3.2 Pin the per-backend constants (record-stack sizing formula, strides as
      Metal reflection-fallbacks, rebuild-key elements) with a test naming each
      reason.

## 4. Gates

- [ ] 4.1 `ruff check src/`; full hostless `pytest` (the golden + pin tests).
- [ ] 4.2 GPU smoke: one wavefront render per backend (Vulkan + native Metal),
      serialised under ZERO-SWAP, confirming every kernel still dispatches.
- [ ] 4.3 Docs: add the kernel-name table to `docs/Wavefront.md`.
- [ ] 4.4 `openspec validate choice-table-wavefront-owners --strict`.
