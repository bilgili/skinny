# renderer-conditioner-encoding Specification

## Purpose
TBD - created by archiving change renderer-conditioner-encoding. Update Purpose after archive.
## Requirements
### Requirement: Selectable in-shader conditioner encoding
The renderer SHALL apply the neural flow's conditioner positional encoding in-shader,
selected by `--encoding {E0,E1,E3}` mapped to an `NF_ENCODING` build define. `E0` SHALL be a
raw passthrough that is byte-identical to the pre-change network behavior; `E1`/`E3` SHALL
apply the NeRF-γ feature map (with `E3` additionally appending the raw condition).

#### Scenario: E0 reproduces the raw-condition baseline
- **WHEN** the renderer is built with `--encoding E0`
- **THEN** the conditioner input is the raw condition and the network behavior is identical to
  before this change

#### Scenario: E1/E3 apply the feature map
- **WHEN** the renderer is built with `--encoding E1` (or `E3`)
- **THEN** the conditioner input is the encoded condition (`E3` also carrying the raw tail) and
  the network conditions on the encoded features

### Requirement: Encoding matches the trainer byte-for-byte
The shader's encoding SHALL reproduce the trainer's `make_cond_encoding(regime="path")` exactly
— the same encoded dimensions, band frequencies, sin/cos ordering, and raw-tail inclusion — so
the rendered condition equals the condition the network was trained on.

#### Scenario: Encoded condition parity
- **WHEN** the same raw condition is encoded by the renderer and by the trainer for a given
  preset
- **THEN** the two encoded vectors are equal within numerical tolerance

### Requirement: Conditioner encoding does not change the Jacobian
The encoding SHALL be applied only to the conditioner input and SHALL NOT change the flow's
solid-angle Jacobian or pdf normalization. The solid-angle pdf path SHALL be identical across
encodings.

#### Scenario: Jacobian invariant across encodings
- **WHEN** the solid-angle pdf is evaluated under `E0`, `E1`, and `E3`
- **THEN** the Jacobian term and `NF_LOG2PI` are identical (the encoding feeds only the spline
  parameters, never the measure transform)

### Requirement: Encoding is validated against the network tag
The renderer SHALL compare the requested `--encoding` against the loaded network's encoding
tag and SHALL refuse a network whose encoding does not match the build, including a first-layer
input-dimension mismatch, rather than rendering a mis-conditioned result.

#### Scenario: Encoding mismatch is refused
- **WHEN** a network trained under one encoding is loaded with `--encoding` requesting a
  different encoding (or a mismatched first-layer input dimension)
- **THEN** the renderer refuses the network with a clear encoding/architecture-mismatch error

