## ADDED Requirements

### Requirement: Build-time selectable coupling transform
The neural flow SHALL select its per-coupling transform at build time via an
`NF_COUPLING` define — `rqs` (rational-quadratic spline, default) or `nis-pq`
(piecewise-quadratic, the NIS baseline) — analogous to `NF_CHART` and
`NF_ENCODING`. `NF_COUPLING=rqs` SHALL be byte-identical to the shipped network.
The alternating-mask topology, the conditioner MLP, the chart sandwich, and the
`NF_LOG2PI` pdf conversion SHALL be unchanged across the selection.

#### Scenario: Default/RQ reproduces the baseline
- **WHEN** the renderer is built with `NF_COUPLING=rqs`
- **THEN** the neural flow's forward, inverse, and log-determinant are
  byte-identical to the shipped network

#### Scenario: NIS coupling is selectable and chart-agnostic
- **WHEN** the renderer is built with `NF_COUPLING=nis-pq` at chart `V1`
- **THEN** the flow uses the piecewise-quadratic warp under the V1 equal-area
  chart, and the constant-`2π` (`NF_LOG2PI`) pdf path is unchanged

### Requirement: NIS coupling matches the trainer bit-for-bit
The shader `nis_quad_fwd`/`nis_quad_inv`/`nis_decode` routines SHALL match the
`spline_flow` trainer's piecewise-quadratic coupling within parity tolerance on
exported reference samples and pdfs, so a net trained offline renders without
drift.

#### Scenario: Shader/trainer parity
- **WHEN** a `nis-pq` net's exported reference samples/pdfs are compared against
  the shader evaluation
- **THEN** the forward, inverse, and pdf agree within the neural parity tolerance
