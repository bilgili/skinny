## ADDED Requirements

### Requirement: NIS proposal plugin in the mixture
The proposal registry SHALL include a `nis` proposal plugin (`cli_token = "nis"`,
`mask_bit = 0x8`) participating in the wavefront mixture-MIS proposal alongside
`bsdf`, `env`, and `neural`. Adding the proposal SHALL be additive: the mixture
pdf division over active bits and the BSDF-only bit-identity default (NIS off)
SHALL be unchanged.

#### Scenario: NIS participates in the mixture
- **WHEN** a render enables `bsdf,nis`
- **THEN** the one-sample mixture-MIS estimator divides by the full mixture pdf
  over the active bsdf and nis bits, and the result is unbiased

#### Scenario: Default unchanged when NIS is off
- **WHEN** NIS is not enabled
- **THEN** the proposal mask and the estimator are bit-identical to the
  pre-change BSDF-only default
