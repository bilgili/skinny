# Proposal — renderer-conditioner-encoding

## Why

The conditioner positional encoding (axis 2 of the parameterization taxonomy: `E0/E1/E3`)
exists and is studied in `spline_flow` (`make_cond_encoding` → `FourierEncoding` /
`fourier_gamma`), but the **renderer applies none of it**. `neural_flow.slang` feeds the raw
condition straight to the conditioner MLP — it is `E0` only. `directional-flow-parameterization`
explicitly punted the in-shader `nf_encode` + `NF_MLP_IN` split as out of scope.

Consequence: a network trained with `E1`/`E3` **cannot be rendered faithfully** in skinny. The
condition encoding is part of the canonical, byte-for-byte trainer↔renderer contract
(`neuralCondition`); if the trainer applies a NeRF-γ feature map and the shader does not, the
net is silently mis-conditioned — variance, not bias, but a real efficiency loss the paper
must not eat unknowingly.

It also blocks the paper's renderer-side variance experiments from sweeping axis 2. To produce
equal-time / equal-variance graphs across the full 4-axis space *in skinny* (not only in
spline_flow's synthetic study), the renderer must apply the same encoding the net was trained
with. This change ports the encoding into the shader and exposes `--encoding`.

## What Changes

- **In-shader encoding.** Port `spline_flow`'s `fourier_gamma` / `FourierEncoding`
  (path-regime presets) into `neural_flow.slang` as `nf_encode(cond) -> encoded[NF_MLP_ENC]`,
  **byte-for-byte matching** `make_cond_encoding(regime="path")` (same band assignment, same
  `(sin,cos)` ordering, same `include_raw` for `E3`).
- **`-D NF_ENCODING {E0,E1,E3}` build dim.** Selects the encoding; `NF_MLP_IN` becomes the
  encoded width (`E0`: `1 + NF_COND`, unchanged and byte-identical; `E1`/`E3`: `1 +
  encoded_dim`). The conditioner's first linear layer input dim — and therefore the NFW1
  layout's first header `(mlp_in, hidden)` — follows `NF_ENCODING`; the loader validates it.
- **`--encoding {E0,E1,E3}` CLI** in `cli_common.py` → `NF_ENCODING`; validate against the
  `.nrec` encoding tag (from `flow-parametrization-cli`) and refuse a mismatch.
- **Jacobian-free by construction.** The encoding is a conditioner side-input (axis 2): it
  changes only the spline parameters the MLP emits, never the measure transform. `|J|`,
  `NF_LOG2PI`, and the pdf path are unchanged — the proven invariant.

## Capabilities

### New Capabilities
- `renderer-conditioner-encoding`: the renderer applies the neural flow's conditioner
  positional encoding (`E0/E1/E3`) in-shader, byte-for-byte with the trainer, selected by
  `--encoding` and validated against the network's encoding tag, so an `E1`/`E3`-trained net
  renders faithfully — Jacobian-free, with `E0` byte-identical to today.

## Impact

- **Code:** `neural_flow.slang` (`nf_encode` + `NF_MLP_IN`/`NF_MLP_ENC`), `neural_proposal.slang`
  (apply at the conditioner call site), `cli_common.py` (`--encoding`), `neural_weights.py`
  (encoding-tag validation + the first-layer `in_dim` in `_layout`). The flow core, chart, and
  MIS are unchanged.
- **Math:** axis 2 ⇒ no Jacobian change; `E0` byte-identical to the shipped net.
- **Backends:** `E1`/`E3` widen the first-layer GEMM only; no new descriptor binding; Metal
  slot cap untouched. `slangc` recompiles per `NF_ENCODING` (a build dim like `NF_CHART` /
  `NF_COND`).
- **Dependencies:** `spline_flow` `flow-parametrization-cli` (the `.nrec` encoding tag).
  Composes orthogonally with `renderer-chart-selection` (axis 1) and
  `neural-temporal-conditioning` (axis 3) — all three are independent build dims.
- **Out of scope:** the `E2` preset (dropped from the canonical CLI set, though the
  `spline_flow` registry still sweeps it); any spline_flow-side encoding work (already exists).
