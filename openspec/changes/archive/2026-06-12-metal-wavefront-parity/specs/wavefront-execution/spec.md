## REMOVED Requirements

### Requirement: Wavefront is a Vulkan-backend feature

**Reason**: The native Metal backend now runs the staged wavefront passes, so
wavefront is no longer a Vulkan-only feature. The Metal-pins-to-megakernel
constraint is lifted and replaced by the multi-backend requirement below.

**Migration**: Selecting wavefront on the Metal backend now renders via the staged
wavefront passes (the backend-neutral bounce-loop driver), not the megakernel.
Front-ends that previously hid the wavefront option on Metal SHALL expose it. No
CLI/flag change is required.

## ADDED Requirements

### Requirement: Wavefront runs on the Vulkan and Metal backends

The wavefront execution mode SHALL be available on both the Vulkan and the native
Metal backend. Both backends SHALL drive the same staged bounce loop (generate →
intersect → build_args → scatter → per-material shade → resolve) from a single
backend-neutral driver, differing only in their dispatch and synchronization
primitives (Vulkan command-buffer recording with `vkCmdDispatchIndirect`; Metal
single-frame command encoding with indirect dispatch or the logged CPU-readback
fallback). The wavefront option SHALL be selectable on every front-end on both
backends, and the megakernel default SHALL remain unchanged on each.

#### Scenario: Wavefront selectable on Metal

- **WHEN** the application runs on the Metal backend and the user selects the
  wavefront execution mode
- **THEN** the scene renders via the staged wavefront passes on Metal, not the
  megakernel, and the option is selectable on the front-end

#### Scenario: Wavefront selectable on Vulkan

- **WHEN** the application runs on the Vulkan backend and the user selects the
  wavefront execution mode
- **THEN** the scene renders via the staged wavefront passes, byte-identically to
  the behavior before this change

#### Scenario: Same staged loop on both backends

- **WHEN** the same scene is rendered with the wavefront path integrator on Metal
  and on Vulkan at an equal sample budget
- **THEN** both run the identical stage order and per-bounce memory bound, and the
  structural outputs agree exactly across backends
