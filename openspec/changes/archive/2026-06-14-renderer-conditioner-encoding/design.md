# Design — renderer-conditioner-encoding

## The byte-for-byte contract (the whole risk surface)

The encoding is already part of the canonical `neuralCondition` contract: trainer and renderer
must produce the *identical* conditioner input or the net is silently mis-conditioned. So the
single hard requirement is that the shader's `nf_encode` reproduces `spline_flow`'s
`make_cond_encoding(regime="path")` exactly — same dims encoded, same band frequencies, same
`(sin, cos)` interleave, same `include_raw` tail for `E3`.

`spline_flow` reference (read at implementation time, do not re-derive):
- `fourier_gamma(p, L)`: `γ(s) = (sin(2⁰πs), cos(2⁰πs), …, sin(2^{L-1}πs), cos(2^{L-1}πs)) ∈ ℝ^{2L}`
  per scalar, ordered `(sin0,cos0,sin1,cos1,…)`.
- `make_cond_encoding(preset, cond_dim, regime="path", L_dir=4, L_pos=10)`:
  - `E0` / raw / `None` → identity passthrough.
  - `E1` → per-group NeRF-γ bands (path regime: the position group gets `L_pos` bands).
  - `E3` → `E1` **plus** the raw condition appended (`include_raw=True`).

The renderer's condition is the 9-D canonical `(pos, N, wo)` (10-D with temporal). The band
assignment per group must match the path-regime preset exactly; the *task* transliterates it
from `train.py`, the *requirement* is byte-for-byte parity (a parity test asserts the renderer's
encoded vector equals spline_flow's on shared inputs).

## Layout ripple — the first layer only

Encoding changes `NF_MLP_IN` (the conditioner's first linear input width):
`E0: 1 + NF_COND` (today) → `E1/E3: 1 + encoded_dim`. That is the *only* dimensional change —
the hidden layers and the spline-parameter head are untouched. The NFW1 `_layout` first header
`(mlp_in, hidden)` follows `NF_ENCODING`, and the loader validates the trained net's first-layer
`in_dim` against the build (same arch-guard pattern as `NF_COND`/size). A mismatch is refused.

## Why it cannot touch the Jacobian (axis 2, restated)

`nf_encode` maps `cond → encoded` *before* the conditioner MLP. It feeds the spline parameters,
never the `u→z` transform or the chart. By the triangular-coupling argument the logdet is the
product of the transformed coordinate's spline derivatives; `encoded` sits in the conditioning
block, below the diagonal, excluded from the determinant. So `|J|`, `NF_LOG2PI`, and `pdfNeural`
are unchanged. `spline_flow` records the same fact at `train.py` (`FourierEncoding` "side input
to the conditioner only; never the u→z transform"); the shader inherits it.

## Decisions

- `E0` is the default and byte-identical to the shipped net (the encoded path compiles out).
- `NF_ENCODING` is a **build dim** (recompile per encoding), consistent with `NF_CHART` /
  `NF_COND`; an encoding sweep recompiles per value (Metal: one guarded compile at a time).
- Drop `E2` from the renderer CLI choices (matches the paper's canonical `E0/E1/E3` set); the
  spline_flow registry may still sweep it for the synthetic study.
- Tag-validate: `--encoding` must match the loaded `.nrec`'s encoding tag.

## Out of scope / open

- The exact per-group band counts (`L_pos`, and whether `N`/`wo` groups get bands in the path
  regime) are read from `make_cond_encoding` at implementation time — not re-decided here.
- Temporal interaction: when `--temporal on`, the `t` scalar's own band is governed by
  `--time-encoding` (the `neural-temporal-conditioning` change); `nf_encode` here covers the
  spatial condition. The two compose (both Jacobian-free); their band layouts are concatenated
  in canonical order. Coordinate the concatenation order with that change.
