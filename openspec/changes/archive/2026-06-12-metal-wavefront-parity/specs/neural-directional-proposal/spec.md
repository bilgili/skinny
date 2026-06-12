## MODIFIED Requirements

### Requirement: Wavefront-only neural inference pass
The renderer SHALL run neural proposal inference as a wavefront compute pass that consumes
per-lane hit state and produces a per-lane direction and pdf for the bounce stage, on the
wavefront execution backend on **both** Vulkan and Metal. The neural proposal SHALL NOT be
available in the megakernel execution mode on either device.

#### Scenario: Wavefront produces per-lane proposal
- **WHEN** a render uses the wavefront backend (Vulkan or Metal) with the neural proposal active
- **THEN** a neural pass writes a per-lane `(wi, pdf)` that the bounce MIS mixture consumes

#### Scenario: Megakernel rejects neural
- **WHEN** the neural proposal is requested with the megakernel execution mode (on either device)
- **THEN** the renderer reports it as unsupported rather than silently ignoring the request

#### Scenario: Metal neural matches Vulkan neural
- **WHEN** the same scene is rendered with the neural proposal on the Metal wavefront path and on
  the Vulkan wavefront path at an equal sample budget
- **THEN** the converged result agrees within the established perceptual tolerance, with fp32
  weights giving structural agreement and fp16 weights staying within that tolerance

### Requirement: Frozen offline-trained weights as loadable GPU state
The neural proposal SHALL load frozen, offline-trained network weights from a file and
upload them to GPU buffers it owns on both the Vulkan and Metal backends, allocating those
buffers and their descriptor bindings only while the proposal is active and releasing them
when it is deselected. On Metal the weights SHALL load by buffer upload (`set_data`), not
GPU↔GPU external-memory interop, and SHALL be stored in fp16 where the device reports fp16
support and in fp32 otherwise.

#### Scenario: Activation allocates and loads
- **WHEN** the neural proposal is selected for a render on either backend
- **THEN** its weight buffers are allocated, bound, and populated from the weights file

#### Scenario: Deselection releases
- **WHEN** the neural proposal is deselected
- **THEN** its buffers and bindings are released and no neural GPU state remains

#### Scenario: Metal loads weights by buffer upload
- **WHEN** the neural proposal is selected on the Metal backend
- **THEN** the weight blob is uploaded into a Metal storage buffer via `set_data`, in fp16 when
  the device supports it and fp32 otherwise, with no external-memory handoff
