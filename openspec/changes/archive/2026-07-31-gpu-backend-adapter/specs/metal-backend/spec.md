# metal-backend (delta)

## MODIFIED Requirements

### Requirement: Vulkan-only host paths degrade safely on Metal

Host paths with no Metal implementation SHALL degrade through **named
capability reads on the backend adapter interface**, not through vendor
identity checks or attribute-presence probes. Specifically: descriptor-set
writes, external-memory neural handoff, GPU skinning (`vk_skinning.py`, which
has no MSL counterpart and falls back to CPU), indirect wavefront dispatch,
and megakernel record sourcing SHALL each be gated on a declared capability.
The `descriptor_sets is None` sentinel MUST NOT be used as a backend test, and
`hasattr(ctx, "compute_queue")` MUST NOT be used as a Vulkan test — the latter
is unconditionally true because `MetalContext.compute_queue` is `None` rather
than absent, so three wavefront pass factories are currently protected only by
their caller. Where a path is genuinely unavailable, the refusal or fallback
SHALL name the missing capability rather than the backend.

#### Scenario: Degradation names the capability

- **WHEN** a host path unavailable on the active backend is reached — GPU
  skinning, external-memory handoff, or megakernel record sourcing
- **THEN** the fallback or refusal is selected by reading the declared
  capability, and any user-visible message names the missing capability

#### Scenario: Wavefront factories are guarded, not lucky

- **WHEN** a Metal context is passed to a Vulkan wavefront pass factory
- **THEN** the factory refuses on a declared capability read, rather than
  passing an attribute-presence check that is always true and relying on the
  caller having routed correctly

#### Scenario: Renderer carries no vendor branch for a capability question

- **WHEN** `renderer.py` is searched for `is_metal` branches
- **THEN** each remaining occurrence is either the adapter selection itself or
  a genuine two-implementation split, and no occurrence stands in for a
  question the capability record can answer
