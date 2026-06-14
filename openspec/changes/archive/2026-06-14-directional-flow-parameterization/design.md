# Design — Lambert directional chart port (V1)

## Context

`neural_flow.slang` sandwiches the RQ coupling flow between two fixed maps. Only the
**output chart** (`z → ω`) carries a Jacobian; it is a probability measure transform,
so `q_ω = q_□ / |J|`. The shipped chart is cylindrical equal-area (`|J|=2π`, the
`NF_LOG2PI` constant). The `spline_flow` study proved the **Lambert azimuthal** chart
(V1) is also equal-area (`|J|=2π`) and wins both regimes, so the port is a pure pair
swap with **no pdf-path change**.

```
sampleNeural:  u --flow_forward--> z --[CHART]--> wi ;  pdfOmega = exp(-logdet - NF_LOG2PI)
pdfNeural:     wi --[CHART^-1]--> z --flow_inverse--> logdet ;  pdf = exp(logdet - NF_LOG2PI)
```

Both lines keep `NF_LOG2PI` because `|J|` is the same constant for V0 and V1.

## The V1 chart (math, y-up; `ω_y = cosθ`)

### Forward `nf_square_to_hemi(z) → ω`

1. **Shirley concentric** square `[0,1]² → unit disk` (continuous, equal-area, seam-free).
   With `a = 2u−1`, `b = 2v−1`:
   - if `a² > b²`: `r = a`, `φ = (π/4)(b/a)`
   - else: `r = b`, `φ = (π/2) − (π/4)(a/b)`  (and `r=φ=0` at the centre)
   - `(c_x, c_y) = r(\cosφ, \sinφ)`  (unit disk, signed-radius form)
2. **Lambert lift** with `r = |(c_x,c_y)| = \sqrt{c_x^2+c_y^2}`:
   - `\cosθ = 1 − r²`,  `\sinθ = \sqrt{1 − \cos^2θ}`
   - `(\cosφ, \sinφ) = (c_x, c_y)/r`  (guard `r→0`: pole `ω = (0,1,0)`)
   - `ω = (\sinθ\cosφ, \cosθ, \sinθ\sinφ)`

### Inverse `nf_hemi_to_square(ω) → z`

- `\cosθ = ω_y`,  `r = \sqrt{1 − ω_y}`  (since `r² = 1 − \cosθ`),  `\sinθ = \sqrt{1 − ω_y^2}`
- `(\cosφ, \sinφ) = (ω_x, ω_z)/\sinθ`  (guard `\sinθ→0`: disk centre)
- `(c_x, c_y) = r(\cosφ, \sinφ)`
- **inverse-concentric** `(c_x,c_y) → (u,v)`, then clamp `(u,v)` to `[1e-5, 1−1e-5]`.

### Why equal-area ⟹ `|J| = 2π` unchanged (explicit determinant)

The recipe for every chart: write `dω = sinθ dθ dφ`, substitute to get `du dv`, read
the determinant `|∂(angles)/∂(u,v)|`.

- **V0 cylindrical** — substitute `z = cosθ` (so `dz = −sinθ dθ`), which *cancels* the
  `sinθ`:  `dω = sinθ dθ dφ = dφ dz`. The `(u,v) → (φ,z)` map is diagonal:
  `det[[2π, 0], [0, 1]] = 2π`  ⟹  `dω = 2π du dv`  ⟹  **|J| = 2π**.

- **V1 Lambert** — the lift `cosθ = 1 − r²` gives `sinθ dθ = 2r dr`, so in disk-polar
  `dω = sinθ dθ dφ = 2r dr dφ = 2·(dc_x dc_y)`. Shirley concentric is area-preserving
  onto the unit disk, so `|∂(c_x,c_y)/∂(u,v)| = π` (the area ratio `π/1`):
  `dω = 2·dc_x dc_y = 2π du dv`  ⟹  **|J| = 2π**. (Equivalently `ρ = 2 sin(θ/2) ⟹
  ρ dρ dφ = dω` on a radius-`√2` disk of area `2π` = hemisphere solid angle.)

- **V2 = R ∘ V1** — a rotation is an isometry of the sphere: `|J| = |J_V1|·|det R| =
  2π·1 = 2π`.

So the coded `|J|` is the **same constant `2π`** for V0 and V1/V2 — `NF_LOG2PI` is
reused verbatim and the MIS path is byte-identical. The single coordinate choice
`z = cosθ` (not `θ`) is what buys this: had we parameterized by `θ` directly (the
rejected V5), the diagonal determinant would be `det[[2π,0],[0,π/2]] = π²` with the
`sinθ` left uncancelled — `|J| = π² sinθ`, a per-sample term and a pole singularity.
That is precisely why V5 was rejected and V1 keeps the plumbing untouched.

## Reference implementation

`spline_flow/train.py` (`_square_to_disk`, `_disk_to_square`, `_ChartV1`) is the
authoritative reference; this port transliterates it to Slang. The study's
`test_charts.py` proofs (round-trip, `∫ q_ω dω = 1` three ways, forward≡inverse) are
the acceptance oracle — the Slang harness must reproduce them in-shader.

### Numerical guard (`_Z_EPS`)

The study found the RQ spline's parameter gradient blows at the exact `[0,1]` knot edge
(NaN backward on MPS) and the Lambert inverse concentrates points there. Training fixed
it by clamping the inverse to `[1e-5, 1−1e-5]`. In the renderer the flow is
inference-only, so the failure mode is a finite-value concern, not a gradient one — but
the same clamp is kept for parity and to avoid `atan2`/`sqrt`-at-edge denormals. The
forward `clamp(z, 0, 1)` already present is retained.

## What stays / what is rejected

| item | decision | reason |
| --- | --- | --- |
| `NF_LOG2PI`, `sampleNeural`/`pdfNeural` body | **keep** | V1 equal-area, `|J|=2π` |
| `wi.y ≤ 0 → 0` hemisphere guard | **keep** | hemisphere-only chart |
| RQ coupling core, MLP, `.nrec` format | **keep** | unchanged |
| V2 frame rotation | **reject** | loses BRDF (0.58×); substitutes for encoding |
| V5 `π² sinθ` per-sample Jacobian | **reject** | non-equal-area; marginal; re-exposes pole |
| path-only `E3` encoding (`nf_encode` + `NF_MLP_IN` split) | **out of scope** | separate change if pursued |

## Weights

V1 is a different target than V0, so a V1-trained net is required (train
`spline_flow` with `chart="V1"`, export via `export_weights.py`). The `.nrec` format
and upload path are unchanged; only the bytes differ. A V0 net run through the V1 chart
would be biased — the parity harness guards this (V1 `pdfNeural` must match the
V1-trained `spline_flow` reference, not the V0 one).

## Process

- Implement in an isolated git worktree off skinny `main` (repo convention).
- Land behind the completed study; this proposal is the planning artifact.
