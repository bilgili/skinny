## ADDED Requirements

### Requirement: Proposal with GPU state and a wavefront pre-pass
The scene-sampling seam SHALL support a directional proposal that owns GPU buffers and
descriptor bindings and is produced by a wavefront compute pre-pass writing a per-lane
direction and pdf consumed at the bounce, in addition to purely analytic proposals that
own no GPU state. Selecting such a proposal SHALL allocate its GPU state and deselecting it
SHALL release it, without affecting analytic-only selections, and the bounce SHALL consume
the pre-pass's per-lane `(wi, pdf)` through the same mixture-MIS contract as an inline
proposal.

#### Scenario: Stateful proposal allocates a pass and buffers
- **WHEN** a proposal that owns GPU state is selected on the wavefront backend
- **THEN** the renderer builds its pre-pass and buffers and the bounce consumes the
  precomputed per-lane `(wi, pdf)` via the mixture pdf

#### Scenario: Analytic-only selection allocates nothing
- **WHEN** only analytic proposals such as `{bsdf, env}` are selected
- **THEN** no proposal pre-pass, buffer, or extra descriptor binding is allocated
