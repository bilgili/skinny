# Renderer chart selection

## Why

The neural flow's hemisphere chart — the square↔direction map sandwiched after
the coupling layers — is a real axis of the `spline_flow` parametrization study
(`CHART = {V0, V1, V2, V5}`), but the renderer implements **only V1** (the
shipped Lambert chart). `guiding_variance_sweep.AVAILABLE_CHARTS == {"V1"}`
guards the others out, so the chart axis cannot be measured in the renderer — the
NIS comparison can sweep couplings (RQ/PQ) and encodings (E0/E1/E3) end-to-end but
not charts. This change ports the remaining three charts into the shader behind an
`NF_CHART` build define, mirroring `NF_COUPLING`/`NF_ENCODING`, so a baked net's
chart can be selected and rendered.

## What changes

- **Shader** (`neural_flow.slang`): add `NF_CHART` selector (V1 default,
  byte-identical). Add the V0 (cylindrical) and V5 (equirectangular,
  non-equal-area) square↔direction maps as transliterations of `spline_flow`
  `_ChartV0`/`_ChartV5`. Replace the constant `NF_LOG2PI` measure term at the two
  pdf sites with a chart log-Jacobian (`2π` for the equal-area charts V0/V1;
  `π²·sinθ` for V5). V0/V1/V5 are all z-only maps, so `sampleNeural`/`pdfNeural`
  signatures and the V1 SPIR-V are unchanged.
- **V2 DEFERRED**: V2 (V1 + specular-aligned frame) needs the outgoing direction
  in flow-LOCAL space to build its per-sample frame, but the `.nrec` schema stores
  world `wo` + world `normal` and no tangent basis (T,B) — the offline trainer
  cannot reconstruct the local `wo` the renderer would use, so a V2 net would
  train/infer-mismatch. V2 is out of scope until the record schema carries the
  local frame (a separate change); `NF_CHART=V2` is reserved but not implemented.
- **Host** (`neural_weights.py`): add `NeuralBuildConfig.chart` (default `V1`);
  emit `-D NF_CHART=<n>` for non-default charts; fold the chart into `cache_tag`;
  add a loader chart guard validating the NFW1's stamped chart string against the
  built chart (mirrors the coupling/encoding guards).
- **Harness** (`guiding_variance_sweep.py`): expand `AVAILABLE_CHARTS` to
  `{V0, V1, V2, V5}`; thread `cell["chart"]` into the neural build config.
- **Trainer** (`nis_train_offline.py`): allow `--chart {V0,V1,V2,V5}` (already
  parameterized; widen the choices) so a net can be trained on the matching chart.
- **Parity** (`nis_chart_parity.py`): numpy mirror of every shader chart vs the
  `spline_flow` `CHART` reference (both directions + log-Jacobian) within the
  neural parity tolerance.

## Impact

- Affected specs: `neural-flow-directional-chart` (chart selection requirement).
- Affected code: `src/skinny/shaders/sampling/neural_flow.slang`,
  `src/skinny/shaders/sampling/neural_proposal.slang`,
  `src/skinny/sampling/neural_weights.py`, `scripts/guiding_variance_sweep.py`,
  `scripts/nis_train_offline.py`, `scripts/nis_chart_parity.py` (new).
- `NF_CHART=V1` (default) stays byte-identical to the shipped network; existing V1
  nets and tests are unaffected.
