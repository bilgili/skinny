# Tasks: choice-table-wavefront-owners

## 1. Baseline (recorded in `choice-table-owners` task 1.3)

- [x] 1.1 Confirm the kernel-name inventory: **35 entry points** (path 7, BDPT
      14, SPPM 8, MLT 4, `wfNeuralProposal`, Vulkan-only `wfIndirectPaint`). The
      other 18 `"wf*"` literals in `metal_wavefront.py` are bind-by-name resource
      names, a different namespace — out of scope. Confirmed the 14 duplicated
      pass constants and marked each must-be-equal vs per-backend.

## 2. Kernel-name owner

- [x] 2.1 Declare one module-level `WF_…` constant per kernel in
      `wavefront_driver.py` (+ `KERNEL_ENTRY_NAMES` for the gate).
- [x] 2.2 Repoint the driver's dispatch calls to the constants.
- [x] 2.3 Repoint both backends' `entries` lists to import and use the constants.
- [x] 2.4 Golden test: each constant equals its historical string; no
      kernel-name string literal remains in the driver (beyond its definitions)
      or either backend (`tests/test_wavefront_kernel_names.py`).
- [x] 2.5 Negative control: a stale/renamed constant is an ImportError, not a
      render (subprocess `from … import WF_…_RENAMED`).

## 3. Shared / pinned constants

- [x] 3.1 Move the must-be-equal constants (`WF_MAX_BOUNCES`, `BDPT_MAX_VERTS`,
      `WF_EYE_BOUNCES`/`WF_LIGHT_BOUNCES`, `WF_NUM_SLOTS`, `WF_STREAM_CAP_PATH`/
      `WF_STREAM_CAP_BDPT`, `WALK_MODES`, `RESTIR_DEFAULT_CONFIG`) to
      `wavefront_driver.py`; both pass classes derive their attributes from them.
- [x] 3.2 Pin the per-backend constants — the vertex/aux/reservoir strides
      (Vulkan real stride vs Metal reflection fallback) — equal, with the reason,
      in the test; the record-stack sizing formula stays per-backend by design.

## 4. Gates

- [x] 4.1 `ruff check src/` clean on the touched files; full hostless `pytest`
      passes (3183 passed after rebasing onto the merged beacon work; the 17
      remaining failures are all pre-existing on `main` — MCP schema, pbrt mtlx
      logic — or worktree-only asset absence — none from this change). Four
      source-inspection tests that assumed the kernel-name string literals were
      repointed to the owner: `test_mlt_host` ×2, the `test_shader_variants`
      kernel-golden regex, and the beacon's `_wavefront_entries_in_source`
      (now reads `wavefront_driver.KERNEL_ENTRY_NAMES` instead of grepping
      `vk_wavefront.py`, which the rebase against `beacon-wavefront-attribution`
      surfaced).
- [ ] 4.2 GPU smoke: one wavefront render per backend (Vulkan + native Metal),
      serialised under ZERO-SWAP, confirming every kernel still dispatches.
      **DEFERRED** — a concurrent session has Metal wavefront work in flight
      (`beacon-wavefront-attribution`), so ZERO-SWAP forbids a second guarded
      Metal process. Run when the GPU is free. The change is byte-identical
      (kernel strings + constant values unchanged), proven hostlessly by the
      golden/pin tests, so the smoke is confirmation, not a correctness gate.
- [x] 4.3 Docs: added the kernel-name & shared-constant owner section to
      `docs/Wavefront.md`.
- [x] 4.4 `openspec validate choice-table-wavefront-owners --strict`.
