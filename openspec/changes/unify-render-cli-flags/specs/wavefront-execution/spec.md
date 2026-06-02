## MODIFIED Requirements

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

## ADDED Requirements

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
