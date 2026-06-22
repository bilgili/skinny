# Design — renderer-chart-selection

## Chart set — production vs paper-comparison (be honest about scope)

The `spline_flow` study verdict: **V1 wins both regimes**, V0 is the baseline, V2 only ties
V1 on path (and loses BRDF), V5 is the deliberate non-equal-area control. So the renderer's
selector is tiered, and the change need not pay V5's cost to be useful:

| Chart | Map | `\|J\|` | pdf plumbing | Tier | Cost |
|-------|-----|-------|--------------|------|------|
| V0 | cylindrical equal-area | `2π` const | `NF_LOG2PI` unchanged | production (baseline) | trivial |
| V1 | Lambert (Shirley concentric) | `2π` const | `NF_LOG2PI` unchanged | production (winner) | from `directional-flow-parameterization` |
| V2 | Lambert + frame rotation `reflect(wo)` | `2π` const (rotation `det=1`) | `NF_LOG2PI` unchanged | comparison | modest (one frame rotation/sample) |
| V5 | equirectangular (θ,φ) | `π² sinθ` per-sample | **gated branch** in `sampleNeural`/`pdfNeural` | comparison (control) | heavier — re-exposed pole + per-sample term |

**Minimum viable selector = V0 + V1** (both equal-area, the production claim). V2 is a cheap
add. **V5 is separable** — its `NF_CHART==V5`-gated per-sample-`log(π² sinθ)` term is the
only place the pdf path diverges from the constant-Jacobian plumbing, and it can be deferred
without blocking the V0/V1/V2 selector. The proposal lists all four on the CLI (the user
wants the parametrizations listed) but the implementation can land V5 last or as a follow-up.

## Why V2 stays constant-Jacobian (the reusable fact)

V2 is `Φ_V2(z) = R · Φ_V1(z)` with `R ∈ SO(3)` aligning the pole onto `reflect(wo)`. A
rotation is an isometry on the sphere — `|det R|_tangent = 1` — so `det DΦ_V2 = det DΦ_V1`
and `|J|` is still `2π`. This is why `spline_flow`'s `_ChartV2` inherits `_ChartV1`'s
constant `log_jac`. The renderer mirrors it: V2 reuses the V1 pdf path verbatim, only the
forward/inverse direction map composes the frame rotation (built from `wo`, already in the
condition — no new inputs).

## V5 is the only plumbing change — keep it gated

V5's second coordinate is `θ` (not `cosθ`), so `dω = sinθ dθ dφ` turns the constant `2π`
into a per-sample `π² sinθ`, and the pole (`sinθ → 0`) re-appears. The renderer keeps
`sampleNeural`/`pdfNeural` byte-identical for V0/V1/V2 and adds a `#if NF_CHART==V5` branch
that (a) substitutes `log(π² sinθ)` for `NF_LOG2PI` and (b) clamps the pole. Selecting any
equal-area chart compiles that branch out — zero cost. This is the exact plumbing
`directional-flow-parameterization` flagged as out of its scope.

## Tag validation — close the silent-mismatch hole

`flow-parametrization-cli` bakes `(chart, encoding, jacobian, cond_dim)` into the `.nrec`.
`directional-flow-parameterization` already adds a V0-vs-V1 parity gate; this change
generalizes it: at load, compare the record's chart tag to the requested `--chart` and
refuse on mismatch. A V1-trained net under `--chart V0` is biased (different target measure);
the tag makes that a hard error, not a quiet quality loss.

## Renderer ships analytic; const2pi is a labelled debug

The `--jacobian` axis is a `spline_flow` research knob. The renderer needs only `analytic`.
The one exception is reproducing the paper's V5 non-equal-area bias figure: a clearly-labelled
`--jacobian const2pi` **debug** build that forces `|J|=2π` on V5 to render the biased result
on purpose. Not a default, not for production; documented as a figure-reproduction tool.

## Decisions

- Chart is a **build dim** (`-D NF_CHART`), like `NF_HIDDEN`/`NF_COND` — a recompile per
  chart, not a runtime uniform (the pdf branch for V5 must compile out).
- `--chart` lists all four; **V0/V1 are the supported production charts**, V2/V5 are
  comparison charts (V5 deferrable).
- No new binding; Metal slot cap untouched.
- Depends on `directional-flow-parameterization` (V1 + harness) and `flow-parametrization-cli`
  (`.nrec` tag).

## Open questions (do not block)

1. Land V5 in this change or a follow-up? Default: V0/V1/V2 here, V5 as a tail task or
   separate change if the pole plumbing proves fiddly.
2. Should `--chart` auto-select from the `.nrec` tag (load-says-V1 ⟹ build V1) instead of
   requiring the flag to match? Default: require the flag, validate against the tag (explicit
   beats implicit for a build-dim recompile).
