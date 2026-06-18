## ADDED Requirements

### Requirement: One-blob conditioner encoding
The conditioner-encoding define path SHALL include a one-blob encoding (Müller et
al. 2019): each scalar conditioning component is lifted to a soft one-hot over
`b` bins via a Gaussian kernel. One-blob SHALL register through the existing
`NF_ENCODING` mechanism (shader `nf_encode` plus the host `Encoding` accounting),
emit a wider conditioner input, and remain a Jacobian-free side input that never
enters the coupling log-determinant.

#### Scenario: One-blob is build-selectable
- **WHEN** the renderer is built with the one-blob encoding define
- **THEN** the conditioner receives the one-blob-encoded condition and the flow's
  log-determinant and pdf normalization are unchanged from the raw encoding

#### Scenario: Host/shader one-blob parity
- **WHEN** the host one-blob mirror and the shader `nf_encode` are compared
- **THEN** the encoded condition vectors agree within parity tolerance
