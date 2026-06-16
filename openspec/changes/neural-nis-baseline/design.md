# Design — neural-nis-baseline

## D1 — Coupling is a build-time define, mirroring NF_CHART / NF_ENCODING
The shader flow loop (`neural_flow.slang`) calls `nf_rqs_fwd`/`nf_rqs_inv` by
name. Add `NF_COUPLING` (`0=rqs` default | `1=nis-pq`) selected at build time,
exactly like `NF_CHART` and `NF_ENCODING`, and route the per-coupling warp
through it. `nis_quad_fwd`/`nis_quad_inv`/`nis_decode` implement the
piecewise-quadratic warp (integral of a normalized piecewise-linear PDF over `K`
bins; monotone; per-bin quadratic inverse; closed-form `log|dy/dx|`). The
alternating-mask topology, the conditioner MLP, the chart sandwich, and the
`NF_LOG2PI` pdf conversion are unchanged. *Invariant:* `NF_COUPLING=rqs` is
byte-identical to the shipped kernel (a parity test asserts it, mirroring the
`--chart V0` byte-identity check).

## D2 — Dependency ordering on spline_flow
The torch trainer imports `ConditionalSplineFlow2D` from the `spline_flow` repo.
The piecewise-quadratic coupling and the one-blob encoding are defined **there**
(sibling change `nis-baseline-comparison`) and trained there. This change
consumes the resulting NFW1 net. Therefore `spline_flow nis-baseline-comparison`
MUST land first; skinny only adds the shader path, the proposal plugin, the
weight-format tag, and the eval wiring. The shader math is re-derived to match
the trainer's coupling bit-for-bit (parity test against exported reference
samples/pdfs, like the existing neural parity tests).

## D3 — NFW1 carries a coupling tag; loader validates against the kernel
A net trained with `nis-pq` must not be silently loaded into an `rqs` kernel
(or vice versa) — the warp math differs, so the pdf would be wrong and the
unbiasedness gate would (correctly) trip, but late. Add a coupling-type tag to
the NFW1 header (bump version or use a reserved field) and validate it against
the compiled `NF_COUPLING` at load, erroring on mismatch — the same discipline
as the existing first-layer encoding-dim check. This keeps "wrong net in wrong
kernel" a load-time error, not a silent bias.

## D4 — Mask bit 0x8; MIS mixture math unchanged
`NisProposal` takes the next free proposal bit (`0x8`, after bsdf `0x1`, env
`0x2`, neural `0x4`). The mixture-MIS proposal already divides by the full
mixture pdf over the active bits; adding a bit is additive and the BSDF-only
bit-identity default (NIS off) is preserved. NIS and the RQ neural proposal are
**mutually exclusive in a given run** by construction (one compiled
`NF_COUPLING`), so a run carries at most one neural coupling; the comparison is
across runs, not within one mixture.

## D5 — Chart fixed at V1; the paper's two scenes; both fairness framings
Hold the chart at `V1` (the seam-free equal-area win) so the only moving part vs
the `neural` proposal is the coupling (and the one-blob encoding). Compare on the
restructure plan's scenes — **Veach Ajar** (static) and **NVIDIA Emerald Square**
(animated) — at **equal-time variance** and **equal-variance time**, the same two
operating points the harness already records. Per-sample cost differs
(piecewise-quadratic vs RQ eval), so equal-time captures the real renderer
trade-off; an equal-parameter run (matched layers/bins/width) is recorded
alongside.

## D6 — Renderer unbiasedness gate applies to NIS unchanged
A converged NIS render SHALL match the reference (the same renderer-path
unbiasedness gate the `neural` proposal passes). Because the coupling is exactly
invertible with an analytic log-det under the equal-area chart, NIS is an
unbiased proposal; a badly trained NIS net costs variance, not bias. No NIS
scene number is reported until the converged-render gate passes.

## D7 — NIS's analysis protocol on the rendered scenes
The renderer leg is where NIS's analysis section actually lives, because it
produces images. Adopt its axes:
- **Image-error metric.** Report relative-L2 and MAPE of the rendered image vs a
  high-spp reference (NIS's metrics), per scene, in addition to the harness's
  variance — so the comparison is in the units NIS uses.
- **Convergence curves.** Error vs spp **and** error vs wall-clock time per
  method `{baseline, flow, nis, restir-di, flow+restir}`, the standard NIS plot,
  so the equal-time crossover (where neural inference cost is amortized) is
  visible rather than a single budget.
- **Equal-sample *and* equal-time.** Keep equal-time/equal-variance, add the
  equal-sample-count slice NIS reports.
- **Amortization.** Report the one-time training cost and the per-frame inference
  cost separately, and the frame budget at which the flow pays back its inference
  overhead (the online-use argument NIS makes; here stated for the offline-trained
  net feeding the real-time renderer).
The ablations (coupling depth, PL-vs-PQ, one-blob-vs-identity) are run in the
`spline_flow` controlled comparison (cheaper, exact reference); the renderer leg
reuses the *winning* configuration and reports scene image-error convergence, not
the full ablation matrix, to avoid a combinatorial render cost.

## Alternatives rejected
- **Runtime coupling switch (uniform/function pointer) instead of a build
  define.** Inconsistent with the existing `NF_CHART`/`NF_ENCODING` build-define
  convention and adds per-sample branching to a hot shader; a compiled variant is
  cheaper and matches the harness's existing multi-kernel cache.
- **Re-implement the NIS coupling independently in skinny.** Would risk
  trainer/shader drift; D2 ties the shader math to the `spline_flow` trainer via
  parity tests instead.
- **A NIS-specific chart.** Re-introduces the chart as a hidden variable; D5
  fixes V1 to keep the comparison about the coupling.
