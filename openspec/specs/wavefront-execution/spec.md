# wavefront-execution Specification

## Purpose

Add a `wavefront` execution mode as an axis orthogonal to the integrator
(`path` / `bdpt`): a Vulkan-backend renderer that produces each frame through
staged compute dispatches communicating via GPU-resident path-state buffers and
ray queues, reuses the shared scene front-end, and bounds memory by tiled
streaming rather than per-pixel allocation. `megakernel` remains the default and
preserves current behavior; Metal pins to `megakernel`.

## Requirements

### Requirement: Execution-mode axis orthogonal to the integrator

The renderer SHALL expose an execution mode — `megakernel` or `wavefront` — as a
selection independent of the integrator (`path` / `bdpt`). The execution mode SHALL
be selectable at runtime, included in the render state hash so that switching it
resets progressive accumulation, persisted in `settings.json` across sessions, and
surfaced in `ALL_PARAMS` and every render-surface front-end (GLFW, Qt, debug
viewport) for parity. The `megakernel` mode SHALL remain the default and SHALL
preserve current behavior exactly.

#### Scenario: Switching execution mode resets accumulation

- **WHEN** the user switches the execution mode while a scene is accumulating
- **THEN** progressive accumulation resets to frame 0 and resumes under the newly
  selected mode

#### Scenario: Execution mode persists across sessions

- **WHEN** the user sets an execution mode and restarts the application
- **THEN** the restored execution mode matches the last selected mode

#### Scenario: Megakernel default is unchanged

- **WHEN** the execution mode is `megakernel`
- **THEN** the rendered output and the per-frame dispatch behavior are identical to
  the renderer before this change

### Requirement: Wavefront backend renders via staged dispatches

In `wavefront` mode the renderer SHALL produce each frame through staged compute
dispatches — ray generation, intersection, integration logic, and per-material
shading — that communicate through GPU-resident path-state buffers and ray queues,
rather than a single in-kernel path loop. The wavefront backend SHALL reuse the
shared front-end (BVH and geometry buffers, instance/TLAS records, lights,
environment importance sampling, camera, accumulation image, and uniforms) without
duplicating it.

#### Scenario: Wavefront produces an equivalent image to the megakernel

- **WHEN** the same scene, camera, and sample count are rendered in `megakernel`
  and in `wavefront` mode under a supported integrator
- **THEN** the two accumulated images are equivalent within a documented tolerance

#### Scenario: Wavefront shares the scene front-end

- **WHEN** a scene is loaded and the execution mode is `wavefront`
- **THEN** the same BVH/geometry buffers, instance records, lights, environment,
  and accumulation image used by the megakernel feed the wavefront stages

### Requirement: Tiled streaming bounds wavefront memory

The wavefront backend SHALL process paths in fixed-size streams rather than
allocating one path-state slot per pixel, refilling terminated lanes so that
path-state and bdpt subpath-vertex memory are bounded by the configured stream
size and maximum path depth, not by output resolution. The stream size SHALL be
configurable with a conservative default.

#### Scenario: Memory does not scale with resolution

- **WHEN** the output resolution increases at a fixed stream size
- **THEN** wavefront path-state allocation does not grow proportionally to the
  pixel count

### Requirement: Wavefront supports both path and bidirectional integrators

The wavefront backend SHALL support the unidirectional `path` integrator and the
`bdpt` integrator. The `bdpt` wavefront SHALL match the existing bdpt scope (flat
first-hit only) and SHALL build a camera subpath and a light subpath, store their
vertices, and connect prefix pairs with a visibility test and MIS weighting. Until
wavefront `bdpt` reaches verified parity, selecting `bdpt` together with
`wavefront` SHALL fall back to the megakernel bdpt and SHALL surface that fallback
to the user.

#### Scenario: Wavefront path tracing matches the megakernel path tracer

- **WHEN** a scene is rendered with the `path` integrator in both modes
- **THEN** the accumulated images are equivalent within tolerance

#### Scenario: Bidirectional fallback before wavefront bdpt parity

- **WHEN** the user selects `bdpt` and `wavefront` before wavefront bdpt is
  verified
- **THEN** the renderer uses megakernel bdpt and indicates that wavefront is not
  active for this combination

### Requirement: Wavefront is a Vulkan-backend feature

The wavefront execution mode SHALL be available on the Vulkan backend. On the
Metal backend the execution mode SHALL be pinned to `megakernel` and the wavefront
selection SHALL be unavailable, consistent with other Vulkan-only GPU compute
features.

#### Scenario: Metal pins to megakernel

- **WHEN** the application runs on the Metal backend
- **THEN** the execution mode is `megakernel` and the wavefront option is not
  selectable
