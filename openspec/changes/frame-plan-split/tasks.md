# Tasks: frame-plan-split

## 1. Baseline

- [x] 1.1 Transcribe `update()`'s 18 steps and `render()`'s 16 with their
      ordering dependencies. Record which are load-bearing and why.
      → `baseline.md` §1.1.
- [x] 1.2 Diff `render()` against `render_headless()` line-region by
      line-region; list every genuine difference. Same for the four Metal
      twins. → `baseline.md` §1.2. The Metal twins duplicate *decisions*, not
      recording.
- [x] 1.3 Investigate the per-call binding rewrite in `render_headless`:
      genuine target difference, or latent bug windowed rendering avoids?
      Record the verdict — if a bug, fix it as a separate announced change.
      → `baseline.md` §1.3. **Neither.** It was dead compensation for a rebind
      `render()` had stopped doing, and change `review-surfaced-defects`
      (commit `9e6322d`) already removed it. Nothing to move, nothing to fix.
- [x] 1.4 Capture the full parity matrix (both gates) as the identity target.

## 2. Scene sync

- [x] 2.1 Group `update()`'s steps into scene sync, preserving order exactly.
      → `Renderer._sync_scene(dt)`.
- [x] 2.2 Leave the accumulation hash and reset with their registry owner;
      the plan consumes the decision. → the hash and the reset stay in
      `update()`, after `_sync_scene` returns; the plan reads
      `accum_frame == 0` as `first_frame`.

## 3. Frame plan (pure)

- [x] 3.1 Define the plan value: execution mode, pass sequence, accumulation
      state and reset, banding/tiling, optional per-frame work.
      → `frame_plan.FramePlan` + `frame_plan.derive`.
- [x] 3.2 Express banding as a capability-driven decision, not `is_metal` —
      otherwise this change re-imports the branch problem.
      → `frame_plan.megakernel_bands(needs_watchdog_tiling, …)`;
      `Renderer._needs_watchdog_tiling` is the one binding site.
- [x] 3.3 Hostless tests over the plan for every integrator × execution mode ×
      backend-capability combination in the envelope.
      → `tests/test_frame_plan.py`, matrix built from
      `render_envelope.evaluate`.
- [x] 3.4 Assert the pick-drain-before-uniform-pack ordering.
      → `ORDERING_INVARIANTS` + `check_invariants` on every derivation, plus a
      negative control proving the check is not vacuous.

## 4. Execute

- [x] 4.1 One execution body; target supplies output destination, swapchain
      acquire/present, readback. → `Renderer._execute_vulkan_frame(plan,
      target)` with `_SwapchainTarget` / `_OffscreenTarget`.
- [x] 4.2 Delete the duplicated barrier / execution-mode / dispatch block from
      the headless path and the Metal twins. → the Vulkan block is written once
      (`_execute_vulkan_frame` + `_record_frame_dispatch`); the Metal entry
      points now consume `plan.execution_mode` / `plan.integrator` /
      `plan.first_frame` / `plan.mlt_iterations` / `plan.mlt_chain_batch` /
      `plan.megakernel_bands` instead of re-deriving each.
- [ ] 4.3 If `gpu-backend-adapter` stage 3 has landed: execute the plan against
      the recording adapter and assert the dispatch sequence hostlessly.
      **NOT APPLICABLE — `gpu-backend-adapter` is 0/26; stage 3 has not
      landed.** The plan is shaped for it: `plan.steps` is the sequence that
      adapter will replay, and `check_invariants` already gates its order.

## 5. Gates

- [x] 5.1 `ruff check src/`; full hostless `pytest`. → `ruff check src/` clean
      (172 files, non-vacuous); ruff over `src/` + every test file reports the
      same 28 pre-existing errors on both trees, so this change adds none.
      Hostless: 2873 passed, 7 failed — the identical 7 that fail on `main`
      (6 × `test_corpus_scene_imports_cleanly_mtlx`, 1 × `test_mcp_tool_schemas`).
- [x] 5.2 Parity matrix both gates identical to 1.4 — identical, not close.
      → **Vulkan 73/73 identical** (the path this change rewrote); Metal
      1207/1250 + all 20 heavy-scene lines identical, with 43 confined to three
      combo families measured to be nondeterministic run-to-run on an unchanged
      tree at a higher rate than the before/after delta. Full accounting and the
      honest form of the claim in `baseline.md` §1.4.
- [x] 5.3 Per-integrator smoke: path, bdpt, sppm, mlt × megakernel/wavefront ×
      Metal/Vulkan where in envelope. → all four integrators × both execution
      modes on **both** backends, from the matrix metric lines. The Vulkan run
      was necessary, not optional: on a Metal host `render_headless` takes the
      `is_metal` arm, so a Metal-only sweep never executes
      `_execute_vulkan_frame`.
- [x] 5.4 Spectral smoke on at least one combo. → spectral combos present on
      both backends; the Vulkan spectral set (path/bdpt/sppm/mlt × spectral) is
      exactly identical.
- [x] 5.5 `tests/test_metal_cleanup.py` incl. the gpu-marked kill harness
      (dispatch shape changed). → 13 hostless + 3 gpu-marked passed.
- [x] 5.6 Docs: `docs/Architecture.md` per-frame section, `docs/Megakernel.md`,
      `docs/Wavefront.md`.
- [x] 5.7 `openspec validate frame-plan-split --strict`. → valid.

## Note

Land after `renderer-gpu-resource-set` and `gpu-backend-adapter` stage 3.
Highest-risk change in the set: every rendered image goes through this path.

`renderer-gpu-resource-set` has landed. `gpu-backend-adapter` has not, which
costs only task 4.3 — the plan does not depend on the recording adapter; it is
what the adapter will consume.
