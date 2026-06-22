# Proposal — neural-temporal-conditioning

## Why

Animated playback is the regime where guiding matters most and the current net helps
least. `Renderer._current_state_hash` includes `clock.current_time_code`
(`renderer.py:8424`), so while the animation plays the accumulation resets **every
frame** — **1 spp in motion**. The neural directional proposal's online path
(`neural_replay.py`) is a recency-weighted ring that *chases* the moving scene and is
always behind at playback fps: by the time it has fit the current time it has forgotten
the last, and the next frame has already moved.

Conditioning the flow on **time** changes the strategy from *chase* to *index*. The
condition vector gains a tenth scalar `t` (normalized USD time), so one network represents
the guiding field across the whole timeline. It interpolates across visited times and,
critically, degrades **gracefully** at unvisited times — raising variance, never bias —
because the proposal MIS-combines neural with BSDF/env and `pdfNeural` is the exact pdf of
the neural sampler at any `(condition, t)`, trained or not. Looping playback then
accumulates a full spatio-temporal guide; timeline scrubbing gets a warm guide at any `t`
instantly.

Time is provably a *conditioner side-input*, not a sampled variable, so it does **not**
touch the flow's Jacobian — the equal-area chart measure, the `NF_LOG2PI` pdf conversion,
and the `∫ q_ω dω = 1` normalization are all unchanged per `t` (see `design.md`).

## What Changes

- **CLI.** Add `--temporal {off,on}` and `--time-encoding {raw,fourier}` to the shared
  neural CLI (`cli_common.py`). `off` is the default and is byte-identical to today.
- **Condition dimension as a build dim.** `--temporal on` builds the neural shaders with
  `NF_COND` 9→10 via a `-D` define, mirroring the `neural-precision-size-study` build-dim
  pattern (`-D NF_HIDDEN=…`). The `.nrec` architecture header already validates the arch
  tuple; it gains `cond_dim` so a temporal net loaded into a static build (or vice-versa)
  is **rejected**, never silently mis-rendered.
- **Canonical time encoding.** Extend `neuralCondition` (the canonical encoding in
  `neural_proposal.slang`, which MUST match the trainer byte-for-byte) to append
  `cond[9] = t`, where `t` is the stage's `current_time_code` normalized to
  `[startTimeCode, endTimeCode] → [-1,1]` (the same normalization style as the position
  condition). Add `fc.timeNorm` to `FrameConstants` — a UBO scalar, **no new descriptor
  binding**, so the Metal 31-slot argument-table cap is untouched.
- **Time-stamped records + t-aware replay.** Stamp each live record drain with the
  frame's `current_time_code` (it rides the existing per-batch `generation` stamp in
  `ReplayBuffer.add`, so the GPU record struct does not widen — every record in one
  drain shares one time code). `build_dataset_np` consumes the stamped `t` into the
  conditioner input. The replay buffer gains a **t-aware sampling mode** so the net
  retains multi-`t` knowledge instead of decaying it away (the recency-vs-temporal
  tension is resolved in `design.md`).
- **Online-first scope.** The renderer trains its own temporal net live via the existing
  in-process trainer (numpy oracle / MLX / CUDA). An *offline* temporal prior baked in
  `spline_flow` is a future dependency (the sibling `flow-temporal-conditioning` change),
  not required for this change; the canonical `t` encoding defined here is the contract
  that offline training must later match.

## Capabilities

### New Capabilities
- `neural-temporal-conditioning`: the neural directional proposal optionally conditions on
  a normalized scene-time scalar, so one network spans an animation's guiding field —
  selectable via `--temporal`, Jacobian-free, with unbiased graceful degradation at
  untrained times and an architecture tag that rejects a static/temporal mismatch.

## Impact

- **Code:** `cli_common.py` (flags), `renderer.py` (`fc.timeNorm`, stamp the record
  drain with `current_time_code`), `neural_proposal.slang` (`neuralCondition` appends
  `t`; `NF_COND` build dim), `neural_trainer.py` / `neural_replay.py` (t-aware dataset +
  sampling), `neural_weights.py` (`.nrec` `cond_dim` tag + arch guard). The flow math,
  chart, and MIS contract are unchanged.
- **Backends:** no new binding; one UBO scalar + a wider condition vector. Metal's
  `SKINNY_METAL_NEURAL` slot-cap path is unaffected (the weight buffers/bindings do not
  change). fp32/fp16/fp8 inference precision is orthogonal and unchanged.
- **Math:** Jacobian unchanged (`design.md` proof — the "not sampled ⟹ not in the
  Jacobian" theorem); per-`t` normalization holds by construction.
- **Docs:** `README.md` compatibility matrix (a temporal column/row), `docs/NeuralGuiding.md`
  (condition encoding + the temporal section), and `CLAUDE.md`'s neural-proposal
  constraints note.
- **Out of scope:** offline temporal training/export in `spline_flow`; chart selection in
  the renderer (the separate change extending `directional-flow-parameterization`);
  Jacobian-mode selection in the renderer (renderer ships `analytic`).
