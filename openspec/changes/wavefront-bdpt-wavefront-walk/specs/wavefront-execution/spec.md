## ADDED Requirements

### Requirement: Wavefront BDPT builds subpaths through staged, compacted bounces

In `wavefront` mode the `bdpt` integrator SHALL construct both the camera (eye)
subpath and the light subpath through per-bounce staged compute dispatches — a
generate stage followed, for each bounce, by a material-free intersect stage and
a material shading/extend stage — rather than a single in-kernel walk that builds
the whole subpath at once. Each bounce SHALL compact the still-active lanes
(counting sort) and dispatch the extend stage over only those lanes, so the work
of later bounces scales with the number of paths still alive rather than the full
stream size. Subpath vertices SHALL reside in GPU buffers between stages, not in
per-thread registers for the duration of the walk. The accumulated image SHALL
remain equivalent to the megakernel `bdpt` within the documented tolerance.

#### Scenario: Later bounces process only live lanes

- **WHEN** a BDPT subpath walk advances and some lanes have terminated (miss,
  Russian roulette, non-flat hit, or maximum depth)
- **THEN** the subsequent bounce's extend stage is dispatched over only the
  still-active lanes, not over the full stream

#### Scenario: Staged BDPT matches the megakernel image

- **WHEN** the same scene, camera, and sample count are rendered with the `bdpt`
  integrator in `megakernel` mode and in `wavefront` mode
- **THEN** the two accumulated images are equivalent within the documented
  tolerance

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
