# neural-flow-directional-chart Specification

## Purpose
TBD - created by archiving change directional-flow-parameterization. Update Purpose after archive.
## Requirements
### Requirement: Lambert azimuthal equal-area output chart
The neural flow SHALL map its square-space sample to a hemisphere direction through the
Lambert azimuthal equal-area chart (Shirley concentric square→disk followed by the
Lambert lift `cosθ = 1 − r²`), and its inverse, in place of the cylindrical equal-area
map. Because the chart is equal-area, the square→solid-angle Jacobian SHALL remain the
constant `2π`, so the existing `NF_LOG2PI` pdf conversion in `sampleNeural` and
`pdfNeural` is reused unchanged.

#### Scenario: Forward sample uses the Lambert chart with the constant Jacobian
- **WHEN** `sampleNeural` maps a flow sample `z` to a direction
- **THEN** it returns the Lambert-chart direction and a solid-angle pdf
  `exp(log_q_square − NF_LOG2PI)` with `NF_LOG2PI` unchanged from the cylindrical chart

#### Scenario: Inverse density uses the Lambert chart inverse
- **WHEN** `pdfNeural` evaluates the flow's density at a direction `wi` with `wi.y > 0`
- **THEN** it inverts the Lambert chart to recover `z`, runs the inverse flow, and
  returns `exp(logdet − NF_LOG2PI)`, and returns `0` for `wi.y ≤ 0`

#### Scenario: Pole and seam are well-behaved
- **WHEN** a sample maps to the chart centre (the normal direction) or the disk boundary
- **THEN** the direction is finite and unit-length (the pole is the single disk-centre
  point, with no azimuth seam), and the inverse `(u,v)` is clamped off the exact
  `[0,1]` boundary so the rational-quadratic spline is evaluated on its open interior

### Requirement: Lambert chart round-trips and integrates to one in-shader
The Lambert chart SHALL pass an in-shader correctness harness demonstrating that it is a
measure-preserving bijection: square→direction→square is the identity away from the
centre/boundary degeneracies, and the solid-angle density integrates to one.

#### Scenario: Round-trip identity
- **WHEN** an interior square point is mapped to a direction and back (and a direction is
  mapped to a square point and back)
- **THEN** the result matches the input within numerical tolerance

#### Scenario: Solid-angle normalization
- **WHEN** the flow's solid-angle density is integrated over the hemisphere on a dense
  `(θ, φ)` grid with the `sinθ` weight
- **THEN** the result is `1` within tolerance, confirming the coded constant Jacobian
  equals the chart's true area scaling

### Requirement: Parity with the spline_flow V1 reference
A V1-trained network's renderer pdf SHALL match the `spline_flow` reference density
evaluated under the same Lambert chart, ensuring the port introduces no bias and that a
chart-matched (not V0) network is used.

#### Scenario: Renderer pdf matches the reference at sampled directions
- **WHEN** a network trained in `spline_flow` with `chart="V1"` is loaded and `pdfNeural`
  is evaluated at directions drawn from the same flow
- **THEN** the renderer's solid-angle pdf agrees with the `spline_flow` V1 reference pdf
  within tolerance at those directions

#### Scenario: A V0-trained network is not silently used under the V1 chart
- **WHEN** the loaded network was trained against the cylindrical (V0) chart
- **THEN** the parity check fails rather than rendering a biased result, signalling that
  a V1-trained record is required

