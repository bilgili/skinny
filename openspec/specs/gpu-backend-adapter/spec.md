# gpu-backend-adapter Specification

## Purpose
TBD - created by archiving change gpu-backend-adapter. Update Purpose after archive.
## Requirements
### Requirement: One declared interface at the backend seam

The system SHALL declare one interface that both GPU backends satisfy,
covering resource construction, binding, dispatch, readback, and a capability
record. Each concept SHALL have exactly one name across the adapters — a
concept present on both targets MUST NOT be exposed under two names (as
`PreviewPipeline` / `PreviewPipelineMetal` are today), and a method present on
both MUST use the same parameter names and the same argument domains (one
address-mode and format vocabulary, not `VkEnum` integers on one target and
strings on the other). Members that genuinely exist on only one target SHALL
be listed in an explicit one-sided table declared alongside the interface.

#### Scenario: Adapters conform modulo the declared one-sided table

- **WHEN** the conformance test compares the public surface of every adapter
- **THEN** the surfaces are identical except for members named in the
  one-sided table, and any parameter shared by both has the same name and
  argument domain on both

#### Scenario: No consumer reaches past the seam

- **WHEN** the source tree outside the adapter modules is searched for imports
  of backend-module privates (for example `_make_sampler`,
  `_rgba_f32_to_rgba8`) or for `import` of one specific backend module by name
- **THEN** no such reach remains outside the adapters and `backend_select`

### Requirement: Backend divergence is expressed as named capabilities

Consumers SHALL branch on named capabilities, never on vendor identity or on
attribute presence. The capability record SHALL cover at least: descriptor-set
availability, external-memory interop, indirect dispatch, in-place shared
writes, GPU skinning availability, bindless texture capacity, and whether
dispatches require watchdog tiling. The `descriptor_sets is None` sentinel and
the `hasattr(ctx, "compute_queue")` probe SHALL be removed from every consumer.

#### Scenario: The always-true Vulkan probe is gone

- **WHEN** the source tree is searched for `hasattr(ctx, "compute_queue")` or
  equivalent attribute-presence backend probes
- **THEN** none remain, and the wavefront pass factories gate on a declared
  capability instead — noting that the removed probe was unconditionally true
  because `MetalContext.compute_queue` is `None` rather than absent

#### Scenario: Capacity constants agree with the shader

- **WHEN** the bindless texture capacity is read from the active adapter's
  capability record and compared with the value compiled into the shader for
  that target
- **THEN** they are equal, enforced by test rather than by a source comment

#### Scenario: Every capability replaces a real branch

- **WHEN** a capability is added to the record
- **THEN** at least one pre-existing backend branch is removed in the same
  change

### Requirement: A recording adapter makes dispatch hostlessly assertable

The system SHALL provide a third adapter that records the sequence of
allocations, bindings and dispatches without executing GPU work and without
requiring a device. Tests SHALL use it to assert dispatch ordering, binding
coverage and pass sequencing on any host. The recording adapter MUST NOT be
used to assert radiometric results — image correctness remains the parity
matrix's responsibility.

#### Scenario: Pass sequencing is asserted without a device

- **WHEN** a render is driven against the recording adapter on a host with no
  GPU device of either kind
- **THEN** the recorded dispatch sequence is available for assertion, and the
  test neither skips nor requires a guarded runner

#### Scenario: A missing binding is caught before the GPU

- **WHEN** a dispatch is recorded whose declared resource bindings do not
  cover the shader globals the pipeline reflects
- **THEN** the recording adapter reports the gap, rather than the omission
  surfacing as a device-only failure or a black image

### Requirement: The recording adapter records against real sources

A registered pass's declared shader globals SHALL come from the compiler's own
reflection — generated offline into a checked-in golden — so the
binding-coverage report describes production code rather than values the caller
supplied. The adapter itself SHALL keep only the registry and the device-free
scene bind map it compares against.

The adapter SHALL keep recording, never simulating: it observes declarations,
allocations, bindings, and dispatch order, and produces no pixels and no
radiometric result. Image correctness stays the parity matrix's job.

A parameter added to a shared adapter member SHALL be added to **all three**
adapters with the same name and the same argument domain, including the recording
one — a device-only addition leaves the third adapter behind and is what the
declared-surface fixture exists to catch.

#### Scenario: The declared globals come from the compiler

- **WHEN** the coverage gate needs a registered pass's declared globals
- **THEN** they come from the compiler's reflection golden for that pass, not a
  hand parser and not a literal set at the call site

#### Scenario: The recorder still produces no pixels

- **WHEN** a recorded image is read back
- **THEN** it is zero-filled, and no radiometric claim is derived from it

#### Scenario: A shared parameter reaches all three adapters

- **WHEN** a parameter is added to a member the three adapters share
- **THEN** the surface fixture and the argument-domain check both pass only once
  every adapter carries it with the same domain

