# renderer-module-structure (delta)

## ADDED Requirements

### Requirement: The renderer's device-free core is importable without a GPU package

The device-free code that sits at module scope in `renderer.py` SHALL live in
modules that import no GPU package, split by subject rather than gathered into
one container: material and std-surface packing with their stride constants,
camera math and the camera classes, film and image writers, the SPPM photon
budget math, the texture pool, and the small shared helpers. Each module MUST
be importable in a process where the `vulkan` package is unavailable, enforced
by a subprocess import gate rather than by convention. Signatures, constant
values and packed bytes MUST be unchanged by the move. Tests that exercise
these symbols SHALL import them from their new modules, not by way of
`skinny.renderer` — a re-export keeps source call sites working, but a test
importing `skinny.renderer` still drags in the GPU package and so does not
demonstrate hostlessness.

#### Scenario: Pure modules import with no GPU package present

- **WHEN** each extracted module is imported in a subprocess in which the
  `vulkan` package cannot be imported
- **THEN** the import succeeds

#### Scenario: Packers are testable on a Metal-only host

- **WHEN** the material packing tests run on a host with no Vulkan SDK
- **THEN** they execute rather than skip, closing the silent-skip failure mode
  in which a stripped dynamic-library path turns a missing SDK into a green
  run

#### Scenario: The move changes nothing observable

- **WHEN** the extracted functions and constants are compared with their
  pre-move counterparts — signatures, constant values, and bytes emitted for
  identical inputs
- **THEN** they are identical
