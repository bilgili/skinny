# wavefront-execution Specification

## Purpose

Add a `wavefront` execution mode as an axis orthogonal to the integrator
(`path` / `bdpt`): a renderer — on both the Vulkan and native Metal backends —
that produces each frame through staged compute dispatches communicating via
GPU-resident path-state buffers and ray queues, reuses the shared scene
front-end, and bounds memory by tiled streaming rather than per-pixel
allocation. `megakernel` remains the default on both backends and preserves
current behavior.
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

### Requirement: Wavefront BDPT offers selectable subpath-build modes

In `wavefront` mode the `bdpt` integrator SHALL offer a selectable subpath-build
strategy, fixed for the session, with at least: a single-kernel walk that builds
both subpaths at once (the default, named `fused`), and a per-bounce staged walk
in which the eye subpath — and optionally the light subpath — is built through a
generate stage followed, for each bounce, by an active-lane compaction and an
extend dispatch over only the still-live lanes (subpath vertices residing in GPU
buffers between stages rather than per-thread registers for the walk's
duration). The single-kernel mode SHALL be named `fused`, distinct from the
execution-mode `megakernel`, which names the monolithic `main_pass` dispatch and
a different codepath; the prior name `megakernel` for this walk SHALL continue
to be accepted as a deprecated alias resolving to `fused`. Every mode SHALL
produce an accumulated image equivalent to the megakernel `bdpt` within the
documented tolerance; the mode SHALL affect only the `wavefront` + `bdpt`
combination.

#### Scenario: Every walk mode matches the megakernel image

- **WHEN** the same scene, camera, and sample count are rendered with the `bdpt`
  integrator in `megakernel` execution mode and in `wavefront` mode under any
  offered subpath-build mode (`fused`, `eye`, `eye_light`)
- **THEN** the accumulated images are equivalent within the documented tolerance

#### Scenario: Staged walk processes only live lanes per bounce

- **WHEN** a staged subpath walk advances and some lanes have terminated (miss,
  Russian roulette, non-flat hit, or maximum depth)
- **THEN** the subsequent bounce's extend stage is dispatched over only the
  still-active lanes, not over the full stream

#### Scenario: Deprecated walk alias resolves to fused

- **WHEN** the walk is selected as `megakernel` (via `--bdpt-walk megakernel`,
  `SKINNY_BDPT_WALK=megakernel`, or a persisted value)
- **THEN** the renderer uses the `fused` single-kernel walk and produces the
  same image as selecting `fused` directly, with no error

### Requirement: Wavefront BDPT compacts and strategy-splits the connection stage

In `wavefront` mode the `bdpt` connection stage SHALL exclude lanes that have no
connectable eye subpath, and SHALL route the remaining lanes by subpath shape
into separate connection kernels — a next-event/emissive kernel and a
generic-connection (multi-vertex light subpath) kernel — each dispatched over
only its own lanes. The generic-connection kernel SHALL NOT be dispatched over
lanes whose light subpath cannot form a generic connection. The accumulated image
SHALL remain equivalent to the megakernel `bdpt` within the documented tolerance.

#### Scenario: Directional-only scene skips the generic-connection kernel

- **WHEN** a scene is rendered with `wavefront` `bdpt` and the only lights are
  directional (delta) lights, so no light subpath has two or more vertices
- **THEN** the generic-connection kernel is dispatched with zero work groups and
  the rendered image still matches the megakernel `bdpt`

#### Scenario: Lanes without a connectable eye subpath are excluded

- **WHEN** a lane's camera ray misses, hits a non-flat material, or otherwise
  produces no surface eye vertex
- **THEN** that lane is not dispatched into any connection kernel

### Requirement: Execution-mode and walk exposed by the headless front-end

The headless front-end (`skinny-render`) SHALL expose the execution-mode
selection (`--execution-mode`) and the wavefront-bdpt walk selection
(`--bdpt-walk`) in addition to the interactive front-ends, threaded into the
renderer construction so a headless render can run the wavefront backend and
choose its bdpt walk. (See the `render-cli` capability for the unified flag
surface.)

#### Scenario: Headless selects wavefront and a walk

- **WHEN** `skinny-render` is run with `--execution-mode wavefront --bdpt-walk eye`
  on a Vulkan backend with the `bdpt` integrator
- **THEN** the headless render uses the wavefront backend with the staged `eye`
  walk and produces an image equivalent to the megakernel `bdpt` within the
  documented tolerance

