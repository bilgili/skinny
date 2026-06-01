## ADDED Requirements

### Requirement: Wavefront BDPT offers selectable subpath-build modes

In `wavefront` mode the `bdpt` integrator SHALL offer a selectable subpath-build
strategy, fixed for the session, with at least: a single-kernel walk that builds
both subpaths at once (the default), and a per-bounce staged walk in which the
eye subpath — and optionally the light subpath — is built through a generate
stage followed, for each bounce, by an active-lane compaction and an extend
dispatch over only the still-live lanes (subpath vertices residing in GPU
buffers between stages rather than per-thread registers for the walk's
duration). Every mode SHALL produce an accumulated image equivalent to the
megakernel `bdpt` within the documented tolerance; the mode SHALL affect only
the `wavefront` + `bdpt` combination.

#### Scenario: Every walk mode matches the megakernel image

- **WHEN** the same scene, camera, and sample count are rendered with the `bdpt`
  integrator in `megakernel` mode and in `wavefront` mode under any offered
  subpath-build mode
- **THEN** the accumulated images are equivalent within the documented tolerance

#### Scenario: Staged walk processes only live lanes per bounce

- **WHEN** a staged subpath walk advances and some lanes have terminated (miss,
  Russian roulette, non-flat hit, or maximum depth)
- **THEN** the subsequent bounce's extend stage is dispatched over only the
  still-active lanes, not over the full stream

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
