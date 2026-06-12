## MODIFIED Requirements

### Requirement: ReSTIR DI reuse mode (wavefront-only)

The renderer SHALL provide a `RESTIR_DI` reuse mode, realized as a `ReusePlugin`
that owns wavefront compute passes plus persistent per-pixel buffers and their
descriptor bindings. It SHALL run on the wavefront execution backend on **both**
Vulkan and Metal; its compute passes and persistent reservoir/G-buffer buffers
SHALL be built from backend-neutral resources so the same plugin runs on either
device. Selecting it on the megakernel backend (on either device) SHALL fall back
to identity reuse (stock NEE), mirroring the wavefront-bdpt capability gate.
Switching the reuse mode SHALL rebuild the wavefront pass set (the pass-structural
contract of the scene-sampling reuse hook).

#### Scenario: ReSTIR builds its passes on wavefront, falls back on megakernel

- **WHEN** ReSTIR DI is selected with the wavefront backend on either Vulkan or
  Metal
- **THEN** the reservoir/G-buffer passes and persistent buffers are built and the
  scene renders via ReSTIR on that device

#### Scenario: Megakernel falls back to identity

- **WHEN** ReSTIR DI is selected with the megakernel execution mode (on either the
  Vulkan or Metal device)
- **THEN** no reservoir passes are built and direct lighting is the stock NEE
  result (identity reuse)

#### Scenario: Metal ReSTIR matches Vulkan ReSTIR

- **WHEN** the same scene is rendered with ReSTIR DI on the Metal wavefront path
  and on the Vulkan wavefront path at an equal sample budget
- **THEN** the converged direct-lighting result agrees within the established
  perceptual tolerance across backends
