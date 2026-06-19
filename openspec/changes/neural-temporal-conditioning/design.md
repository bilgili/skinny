# Design — neural-temporal-conditioning

## Jacobian proof — time conditions, it does not transform (so |J| is untouched)

The renderer's pdf path is `pdfOmega = exp(log_q_square − NF_LOG2PI)` for the equal-area
chart. Adding `t` to the condition cannot change any term of it. The reason, in one line:
**a change-of-variables Jacobian counts only the variables you sample; `t` is supplied by
the playback clock, never drawn, so it contributes exactly zero.**

The density decomposes as

```
 log q_ω(ω | c) = − logdet_flow(u→z ; θ(c))  −  log|J_chart|(z)
                   └ flow term, c sets θ's VALUE ┘  └ measure term, arg is z only ┘
```

Both Jacobian pieces are 2×2 (because `u` and `z` are 2-D); neither differentiates `c`.
The deep reason is the **triangular coupling determinant**: a coupling layer's logdet is
the product of the *transformed coordinate's* spline derivatives; the block holding all
`cond` dependence sits strictly below the diagonal and never enters the determinant. The
chart's `log_jac` takes `z` only and has no condition argument. The renderer's own
`neuralCondition` is already declared canonical and the trainer must match it byte-for-byte
— appending `t` extends that contract without touching the measure.

The five ways time *could* corrupt the Jacobian, all closed:

| # | Route | Why it is closed |
|---|-------|------------------|
| 1 | a transformed dimension (3-D flow) | requires **sampling** `t`; it is exogenous, never drawn |
| 2 | the chart Jacobian | chart map/`log_jac` take `z` only; no condition argument |
| 3 | a new term in the flow logdet | triangular det excludes the `cond` block |
| 4 | the base density | uniform on `[0,1]²`, no condition |
| 5 | a normalizer `Z(t)` over time | conditional flows self-normalize per `(c,t)`; there is no joint-over-`t` integral |

The only open route is making the **chart** depend on `t` non-measure-preservingly — which
this design forbids: `t` enters the conditioner MLP, never the chart.

**Empirical gate (turns the proof into a regression test).** Initialize the new time-input
weights of every conditioner's first layer to **0**. Then `θ(c,t) ≡ θ(c)` and the temporal
net's pdf is **byte-identical** to the `NF_COND=9` static net. This both proves no
measure change and gives the safe warm-start: temporal = static + a time channel that
learns up from zero.

## Graceful degradation is unbiased, not just "nice"

The sample path and the pdf path use the **same** flow at the **same** `(c,t)`
(`neuralSampleWorld` / `neuralPdfWorld`), so the MIS pdf equals the actual sampling density
for *any* `t`, trained or not. An untrained/extrapolated `t` therefore raises variance and
the MIS chain falls back toward BSDF sampling — it can never bias the image. This is the
renderer-level restatement of "staleness raises variance, not bias," now extended across
the time axis.

## The core decision — recency vs temporal retention

Today's `ReplayBuffer` is recency-weighted *on purpose* (`exp(−decay·age)`) to track moving
geometry by **forgetting** old frames. Temporal conditioning wants the opposite: **retain**
all `t`, indexed by `t`. These fight. Concrete resolution — a `t`-stratified sampling mode
that fits the existing buffer (which already stamps a parallel `_gen` array per `add`):

```
 add(records, time_code):
   stamp _time[slot] = time_code  alongside _gen   ← per-batch scalar (one frame = one t);
                                                     NO GPU/shader change, NO record widening
 sample(n, mode="t-strat"):
   range [t_min, t_max] over _time[:_size];  K strata (default 16)
   for each OCCUPIED stratum:  draw ~n/K_occ records, recency-weighted (exp(−decay·age))
                               restricted to that stratum
   return (records, times)  →  build_dataset_np appends t as condition column 9
 recency: SECONDARY (within-stratum only). decay gentled (~0.05 vs today's 0.5):
          we WANT old t — indexed, not forgotten.
```

- **Convergence over a looping animation:** loop 1 fills strata as the playhead sweeps;
  loop 2+ refines; steady state = full spatio-temporal field, each stratum refreshed once
  per loop, within-stratum recency keeping each estimate fresh against MC noise.
- **Edit-heal:** a USD edit at `t*` invalidates that stratum; within-stratum recency ages the
  stale samples out as post-edit samples arrive. Explicit per-stratum eviction is a hook
  (mirrors the existing `evict_stale`).
- **Caveat:** a long *one-shot* (non-looping) animation can exceed buffer capacity → the ring
  holds a sliding `t`-window (acceptable online; the offline `flow-temporal-conditioning`
  net covers the full timeline).
- **Defaults / gates:** `K=16`, within-stratum `decay≈0.05`, a min-records-per-stratum
  "occupied" threshold; `--temporal off` keeps today's pure-recency behavior exactly. The
  encoding + capacity come from the `flow-temporal-conditioning` Phase-0 spike (do not build
  this loop until that gate passes).

## `t` is a proxy for scene-state — name the caveat

`t` uniquely determines the guiding field **only for a deterministic animation**
(USD time → fixed poses/lights). Interactive USD edits break the `t → scene-state` map, so
the temporal model goes stale at edited `t`. This is MIS-safe (variance, not bias); the heal
path is the existing recency weighting within strata, or an explicit reset on edit. Topology
changes over `t` (object appears/disappears) make the field `t`-discontinuous; `--time-encoding
fourier` exists for that case (raw scalar smears across the discontinuity — still unbiased).

## Capacity

One net now spans space×time, not one frame. `NF_HIDDEN`/`NF_LAYERS` may need to grow for
long/complex animations; the `neural-precision-size-study` `-D` build dims already make this
a dial, not a rewrite. Out of scope to tune here; flagged for the equal-time evaluation.

## Decisions

- `--temporal off` is the default and byte-identical to today (zero-init gate enforces it).
- `t` normalization = `(time_code − start) / (end − start) · 2 − 1`, clamped; matches the
  position-condition style. Defined once in `neuralCondition`; the trainer mirrors it.
- No new descriptor binding — `fc.timeNorm` is a UBO scalar; Metal slot cap untouched.
- Online-first: renderer trains live; offline `spline_flow` temporal prior is a later
  sibling change that must match this canonical encoding.
