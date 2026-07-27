# Tasks: frame-plan-split

## 1. Baseline

- [ ] 1.1 Transcribe `update()`'s 18 steps and `render()`'s 16 with their
      ordering dependencies. Record which are load-bearing and why.
- [ ] 1.2 Diff `render()` against `render_headless()` line-region by
      line-region; list every genuine difference. Same for the four Metal
      twins.
- [ ] 1.3 Investigate the per-call binding rewrite in `render_headless`:
      genuine target difference, or latent bug windowed rendering avoids?
      Record the verdict — if a bug, fix it as a separate announced change.
- [ ] 1.4 Capture the full parity matrix (both gates) as the identity target.

## 2. Scene sync

- [ ] 2.1 Group `update()`'s steps into scene sync, preserving order exactly.
- [ ] 2.2 Leave the accumulation hash and reset with their registry owner;
      the plan consumes the decision.

## 3. Frame plan (pure)

- [ ] 3.1 Define the plan value: execution mode, pass sequence, accumulation
      state and reset, banding/tiling, optional per-frame work.
- [ ] 3.2 Express banding as a capability-driven decision, not `is_metal` —
      otherwise this change re-imports the branch problem.
- [ ] 3.3 Hostless tests over the plan for every integrator × execution mode ×
      backend-capability combination in the envelope.
- [ ] 3.4 Assert the pick-drain-before-uniform-pack ordering.

## 4. Execute

- [ ] 4.1 One execution body; target supplies output destination, swapchain
      acquire/present, readback.
- [ ] 4.2 Delete the duplicated barrier / execution-mode / dispatch block from
      the headless path and the Metal twins.
- [ ] 4.3 If `gpu-backend-adapter` stage 3 has landed: execute the plan against
      the recording adapter and assert the dispatch sequence hostlessly.

## 5. Gates

- [ ] 5.1 `ruff check src/`; full hostless `pytest`.
- [ ] 5.2 Parity matrix both gates identical to 1.4 — identical, not close.
- [ ] 5.3 Per-integrator smoke: path, bdpt, sppm, mlt × megakernel/wavefront ×
      Metal/Vulkan where in envelope.
- [ ] 5.4 Spectral smoke on at least one combo.
- [ ] 5.5 `tests/test_metal_cleanup.py` incl. the gpu-marked kill harness
      (dispatch shape changed).
- [ ] 5.6 Docs: `docs/Architecture.md` per-frame section, `docs/Megakernel.md`,
      `docs/Wavefront.md`.
- [ ] 5.7 `openspec validate frame-plan-split --strict`.

## Note

Land after `renderer-gpu-resource-set` and `gpu-backend-adapter` stage 3.
Highest-risk change in the set: every rendered image goes through this path.
