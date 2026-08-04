## ADDED Requirements

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
