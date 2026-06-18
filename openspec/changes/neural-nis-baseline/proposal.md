# Proposal — neural-nis-baseline

## Why

The neural directional proposal currently ships only the rational-quadratic (RQ)
spline flow. For the paper's renderer-scene comparison we need the canonical
neural prior — Neural Importance Sampling (NIS, Müller et al. 2019:
piecewise-quadratic coupling + one-blob conditioning) — runnable as a renderer
proposal next to `neural`, so the full-render head-to-head ({baseline, flow
guiding, NIS, ReSTIR-DI, flow+ReSTIR}) covers neural-vs-neural and not just
neural-vs-analytic. The controlled *offline* comparison lands in the
`spline_flow` repo (sibling change `nis-baseline-comparison`); this change is the
**in-renderer** leg: NIS as a real proposal in the wavefront shader, measured on
the same scenes under equal-time variance and equal-variance time.

## What Changes

- **Add `NisProposal`** to `src/skinny/sampling/proposals.py` (`cli_token = "nis"`,
  `mask_bit = 0x8`, the next free bit after `neural`=0x4), and the matching
  shader proposal-bit slot in `shaders/sampling/proposal.slang`. The MIS mixture
  math is unchanged — the proposal divides by the full mixture pdf, with the
  BSDF-only bit-identity default preserved.
- **Add a build-time coupling selector** `NF_COUPLING` to
  `shaders/sampling/neural_flow.slang` (`rqs` default | `nis-pq`), an analogue of
  `NF_CHART`/`NF_ENCODING`. Implement `nis_quad_fwd`/`nis_quad_inv`/`nis_decode`
  (piecewise-quadratic warp, closed-form inverse and `log|dy/dx|`) parallel to
  `nf_rqs_fwd`/`nf_rqs_inv`. With `NF_COUPLING=rqs` the flow is byte-identical to
  the shipped net; the chart sandwich and the constant-`2π` pdf path
  (`NF_LOG2PI`) are untouched (V1 equal-area).
- **Add a one-blob encoding** option to the conditioner-encoding define path
  (`renderer-conditioner-encoding`), matching NIS conditioning, as a Jacobian-free
  side input (it never enters the coupling log-determinant).
- **Tag the coupling in the NFW1 weight format** (`neural_weights.py`): the
  loader SHALL validate that a net's coupling type matches the compiled kernel
  (mismatch errors, like the existing first-layer encoding-dim check).
- **Trainer handoff:** the torch backend consumes the `spline_flow` `nis-pq`
  coupling (the sibling change) so a trained NIS net exports to NFW1 and uploads
  through the existing `networkVersion`-stamped, frozen-render-side path.
- **Eval:** add `nis` to `scripts/guiding_variance_sweep.py` proposals and the
  renderer-scene cells (Veach Ajar static, NVIDIA Emerald Square animated);
  reuse the equal-time variance / equal-variance time / `1/(var·t)` efficiency
  metrics and the renderer unbiasedness gate (a converged NIS render matches the
  reference).

## Capabilities

### New Capabilities
- `neural-nis-baseline`: an in-renderer NIS proposal (piecewise-quadratic coupling
  + one-blob conditioning) for the paper's renderer-scene head-to-head, measured
  under the existing variance harness and unbiasedness gate.

### Modified Capabilities
- `neural-directional-proposal`: gains the `nis` proposal plugin (`mask_bit 0x8`)
  participating in the mixture-MIS proposal.
- `neural-flow-directional-chart`: gains a build-time `NF_COUPLING` selector; the
  RQ path stays byte-identical and the equal-area constant-Jacobian pdf path is
  unchanged.
- `renderer-conditioner-encoding`: gains a one-blob encoding define.
- `neural-guiding-variance-harness`: gains the `nis` method and the
  renderer-scene NIS-vs-flow cells.
- `neural-training-backends`: the torch backend consumes the `spline_flow` NIS
  coupling and exports it to NFW1.

## Impact

- **Code:** `shaders/sampling/neural_flow.slang` (~300 LOC piecewise-quadratic
  coupling + `NF_COUPLING` gate), `shaders/sampling/proposal.slang` (one bit),
  `sampling/proposals.py` (`NisProposal`), `neural_weights.py` (coupling tag +
  loader validation), `sampling/training_backends.py` (handoff), 
  `scripts/guiding_variance_sweep.py` (sweep + scenes). No change to the charts,
  the MIS mixture math, or the constant-`2π` pdf path.
- **Dependency / ordering:** **depends on `spline_flow` change
  `nis-baseline-comparison`** — the piecewise-quadratic coupling, the one-blob
  encoding, and the trained NIS net are defined there and imported by skinny's
  torch trainer. Land that first.
- **Scenes:** Veach Ajar (static) and NVIDIA Emerald Square (animated),
  downloaded from Bitterli Rendering Resources / ORCA and converted to the USD
  loader; licenses recorded for the data-availability statement.
- **Scope honesty:** in-renderer NIS is NIS's *sampler* under the V1 chart and
  the same offline-trained net — not NIS's online-during-render training nor its
  PSS full-path mode. The paper labels it a controlled in-renderer
  re-implementation of NIS's coupling/encoding (consistent with the `spline_flow`
  change's D4).
- **Process:** implement in a git worktree off `main` per project convention;
  validate with `openspec validate neural-nis-baseline --strict`.
