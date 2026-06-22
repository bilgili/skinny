## ADDED Requirements

### Requirement: Build-time selectable hemisphere chart
The neural flow SHALL select its square↔direction chart at build time via an
`NF_CHART` define — `V1` (Lambert azimuthal equal-area, default), `V0`
(cylindrical equal-area), or `V5` (equirectangular, non-equal-area) — analogous to
`NF_COUPLING` and `NF_ENCODING`. (`V2`, V1 composed with a per-sample
specular-aligned frame, is reserved but NOT implemented: it requires the outgoing
direction in flow-local space, which the path-record schema does not carry.)
`NF_CHART=V1` SHALL be byte-identical to the shipped network: the default config
SHALL emit no `-D NF_CHART` flag and the V1 code path and signatures SHALL be
unchanged.

#### Scenario: Default/V1 reproduces the baseline
- **WHEN** the renderer is built with `NF_CHART=V1` (the default)
- **THEN** the chart sandwich, the `sampleNeural`/`pdfNeural` signatures, and the
  constant-`2π` (`NF_LOG2PI`) pdf conversion are byte-identical to the shipped
  network

#### Scenario: Equal-area charts keep the constant Jacobian
- **WHEN** the renderer is built with `NF_CHART=V0`
- **THEN** the flow uses the cylindrical equal-area chart and the solid-angle pdf
  divides by the same constant `2π` (`NF_LOG2PI`)

#### Scenario: Non-equal-area chart uses a per-sample Jacobian
- **WHEN** the renderer is built with `NF_CHART=V5`
- **THEN** the solid-angle pdf divides by the per-sample `π²·sinθ` chart
  Jacobian (not the constant `2π`), matching `spline_flow` `_ChartV5.log_jac`

### Requirement: Chart maps match the trainer bit-for-bit
The shader chart routines SHALL match the `spline_flow` `CHART[V0/V1/V5]`
reference (forward `square→direction`, inverse `direction→square`, and
`log_jac`) within the neural parity tolerance, so a net trained offline on a
given chart renders without drift. The loader SHALL reject a baked net whose
stamped chart string differs from the built `NF_CHART`.

#### Scenario: Shader/trainer chart parity
- **WHEN** each chart's reference square/direction pairs and log-Jacobians are
  compared against the shader evaluation
- **THEN** the forward, inverse, and log-Jacobian agree within the neural parity
  tolerance

#### Scenario: Chart mismatch is rejected at load
- **WHEN** a net stamped with chart `Vx` is loaded into a build with
  `NF_CHART=Vy` (`x≠y`)
- **THEN** the loader raises a clear chart-mismatch error rather than rendering a
  silently wrong distribution
