# Tasks — renderer-conditioner-encoding

## 1. In-shader encoding

- [x] 1.1 Port `fourier_gamma` into `neural_flow.slang` as `nf_gamma(s, L) -> 2L` with the
      `(sin0,cos0,sin1,cos1,…)` ordering matching `spline_flow/train.py`
- [x] 1.2 Add `nf_encode(cond) -> encoded[NF_MLP_ENC]` reproducing
      `make_cond_encoding(regime="path")` for `E0/E1/E3` (per-group bands; `E3` appends the raw
      condition); read the exact per-group band assignment from `train.py`
- [x] 1.3 `-D NF_ENCODING {E0,E1,E3}` build dim; set `NF_MLP_IN` to `1 + NF_MLP_ENC`
      (`E0` ⇒ `1 + NF_COND`, byte-identical); apply `nf_encode` at the conditioner call site
      in `neural_proposal.slang`

## 2. CLI + layout + tag

- [x] 2.1 `--encoding {E0,E1,E3}` in `cli_common.py` → `NF_ENCODING`; persist on the
      interactive front-ends
- [x] 2.2 `neural_weights.py` `_layout`: first header `(mlp_in, hidden)` follows `NF_ENCODING`;
      loader validates the trained first-layer `in_dim` against the build (arch guard)
- [x] 2.3 Validate `--encoding` against the `.nrec` encoding tag; refuse a mismatch with a
      clear error

## 3. Parity + Jacobian-free gates

- [x] 3.1 **Byte-for-byte parity:** the renderer's `nf_encode(cond)` equals `spline_flow`'s
      `make_cond_encoding(regime="path")(cond)` on shared inputs within tolerance, for `E1`/`E3`
- [x] 3.2 **E0 identity:** `NF_ENCODING=E0` produces a pdf byte-identical to the pre-change net
- [x] 3.3 **Jacobian unchanged:** `pdfNeural` / `NF_LOG2PI` are identical across `E0/E1/E3`
      (the encoding is conditioner-side)

## 4. Validate + docs

- [x] 4.1 `slangc` compiles `neural_flow.slang` / `main_pass.slang` clean for each
      `NF_ENCODING`
- [x] 4.2 Render-smoke: an `E1`-trained `.nrec` renders finite, unbiased frames (lower variance
      than its `E0` counterpart on a guiding scene, matching the study's E-axis win)
      — GPU-verified renderer-side: `E1` builds + renders finite non-black frames, and the
      all-zero dummy net is byte-identical across `E0/E1/E3` (rel-mean-diff 0.00000), a
      GPU witness that the encoding is Jacobian-free. The lower-variance-vs-`E0` win needs a
      TRAINED `E1` net (spline_flow/CUDA box) and is the `neural-guiding-variance-harness`
      change's equal-time/equal-variance experiment, not reproducible in this env.
- [x] 4.3 Update `README.md` (`--encoding` flag), `docs/NeuralGuiding.md` (condition encoding
      section)
- [x] 4.4 `openspec validate renderer-conditioner-encoding --strict`
