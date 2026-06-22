# Tasks — neural-temporal-conditioning

## 1. CLI + build dim

- [ ] 1.1 Add `--temporal {off,on}` and `--time-encoding {raw,fourier}` to `cli_common.py`
      (default `off` / `raw`); thread onto the front-end persisted-settings allow-list
- [ ] 1.2 `--temporal on` selects the `NF_COND=10` neural shader build (a `-D NF_COND`
      define, mirroring the `neural-precision-size-study` build-dim pattern); `off` keeps
      `NF_COND=9`
- [ ] 1.3 Reject incompatible combos at startup with a clear message (temporal requires the
      neural proposal active; document interaction with `--execution-mode wavefront`)

## 2. Canonical time encoding (shader + frame constants)

- [ ] 2.1 Add `fc.timeNorm` to `FrameConstants` (UBO scalar, no new binding); pack it in
      `renderer._pack_uniforms` from `clock.current_time_code` normalized to
      `[start,end] → [-1,1]` (clamped; `0` when `has_animation` is false)
- [ ] 2.2 Extend `neuralCondition` in `neural_proposal.slang` to append `cond[9] = t` under
      the `NF_COND==10` build; keep `NF_COND==9` byte-identical
- [ ] 2.3 `--time-encoding fourier` applies the conditioner Fourier band to the `t` index
      only (Jacobian-free; matches the trainer's encoding)

## 3. Time-stamped records + t-aware replay

- [ ] 3.1 Stamp each live record drain with the frame's `current_time_code` (carried
      alongside the existing `ReplayBuffer` `generation` stamp; the GPU record struct does
      NOT widen)
- [ ] 3.2 `build_dataset_np` reads the stamped `t` into the conditioner input when
      `NF_COND==10`
- [ ] 3.3 Add a **t-stratified** sampling mode to `ReplayBuffer` per the `design.md`
      algorithm (`_time` array stamped in `add`; `K=16` strata; recency secondary within a
      stratum, `decay≈0.05`; `sample` returns `(records, times)`); keep pure-recency for the
      non-temporal path; `--temporal off` unchanged.
      **GATE:** build this loop only after `spline_flow` `flow-temporal-conditioning` task 0
      (premise spike) passes; take the encoding + capacity defaults from its result

## 4. Architecture tag + guards

- [ ] 4.1 Add `cond_dim` to the `.nrec` architecture header/validator in
      `neural_weights.py`; a `cond_dim` mismatch (static net into temporal build or
      vice-versa) is rejected, not silently run
- [ ] 4.2 **Zero-init reduction gate (the proof as a test):** a temporal net with zeroed
      time-input weights produces a pdf byte-identical to the `NF_COND=9` static net
- [ ] 4.3 Unbiasedness check: a headless render at an **untrained** `t` is finite and
      matches the no-guiding reference in expectation (variance up, mean unbiased)

## 5. Validate + docs

- [ ] 5.1 `slangc` compiles `neural_proposal.slang` clean for both `NF_COND=9` and `10`
- [ ] 5.2 Headless A/B: static vs temporal net over a short animated USD scene; record
      equal-spp variance at scrubbed times (the temporal-retention win)
- [ ] 5.3 Update `README.md` compatibility matrix, `docs/NeuralGuiding.md` (condition
      encoding + temporal section), and the `CLAUDE.md` neural-constraints note
- [ ] 5.4 `openspec validate neural-temporal-conditioning --strict`
