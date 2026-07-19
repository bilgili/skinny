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

The renderer SHALL expose an execution mode — `megakernel` or `wavefront` — as
a selection independent of the integrator (`path` / `bdpt`). The execution mode
SHALL be selected on the **command line** (`--execution-mode`, with a
`SKINNY_EXECUTION_MODE` environment fallback), mirroring `--backend`, and SHALL
be **fixed for the session** — it is a constructor argument of the renderer, not
a runtime-switchable, GUI-surfaced, or persisted parameter. When no explicit
mode is given (the default, `auto`), the execution mode SHALL be **derived from
the startup integrator**: `path` → `megakernel`, `bdpt` → `megakernel`, `sppm`
→ `wavefront`, `mlt` → `wavefront`; an explicit `megakernel` or `wavefront`
SHALL override that derivation. The `megakernel` mode SHALL remain the derived
default for `path` and `bdpt` and SHALL preserve current behavior exactly. Both
execution modes SHALL run on both backends — the native Metal backend runs the
wavefront mode at parity with Vulkan (`metal-wavefront-parity`), so a
wavefront-only integrator (`sppm`, `mlt`) resolves and runs on Metal. (This
supersedes the pre-`metal-wavefront-parity` clause that pinned Metal to the
megakernel.)

#### Scenario: Execution mode is selected on the command line

- **WHEN** the application is launched with `--execution-mode wavefront` (or
  `SKINNY_EXECUTION_MODE=wavefront`)
- **THEN** the renderer runs in wavefront mode for the whole session, and the
  mode is not offered as a runtime toggle in any front-end

#### Scenario: Default is derived from the integrator

- **WHEN** no explicit execution mode is specified (`auto`)
- **THEN** the execution mode is `megakernel` for `path` and `bdpt` (behavior
  identical to the renderer before this change) and `wavefront` for `sppm` and
  `mlt`

#### Scenario: Megakernel default is unchanged

- **WHEN** the execution mode resolves to `megakernel`
- **THEN** the rendered output and the per-frame dispatch behavior are identical
  to the renderer before this change

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

### Requirement: Only the selected backend is compiled

The renderer SHALL compile the GPU pipelines for the selected execution mode
only. In `megakernel` mode it SHALL build the megakernel compute pipeline and
SHALL NOT build the wavefront stage pipelines. In `wavefront` mode it SHALL NOT
compile `main_pass.slang` or the megakernel compute pipeline at all, and SHALL
build only the wavefront stage pipelines. The scene plumbing the wavefront
shares with the megakernel — the set-0 descriptor-set layout, per-frame
descriptor sets, `generated_materials` emission, and the per-graph binding map —
SHALL be built independently of the megakernel pipeline so the wavefront
backend stands alone.

#### Scenario: Wavefront mode does not compile the megakernel

- **WHEN** the renderer is constructed in `wavefront` mode and a scene is loaded
- **THEN** the megakernel `main_pass` pipeline is never compiled (no
  `slangc` invocation for `main_pass`, and the renderer exposes no megakernel
  pipeline), and the scene renders via the wavefront stage pipelines

#### Scenario: Megakernel mode does not build wavefront pipelines

- **WHEN** the renderer is constructed in `megakernel` mode and a scene is
  loaded
- **THEN** no wavefront stage pipeline is built, and the scene renders via the
  megakernel pipeline

#### Scenario: Adding a material in wavefront mode never touches the megakernel

- **WHEN** a model introducing a new material graph is added at runtime in
  `wavefront` mode
- **THEN** only that material's wavefront shade pipeline is compiled, and no
  megakernel pipeline is (re)built

### Requirement: Wavefront BDPT and SPPM shade terminal non-flat first hits via the path tracer

The wavefront BDPT and SPPM integrators SHALL shade a camera (eye) first hit on
**any** non-flat material — `MATERIAL_TYPE_SUBSURFACE`, `MATERIAL_TYPE_SKIN`,
`MATERIAL_TYPE_VOLUME`, `MATERIAL_TYPE_PYTHON` — by falling back to the path
integrator for that lane, producing the same radiance the megakernel produces for
that pixel. They SHALL NOT leave such a lane's radiance at zero (rendering the
object black). This mirrors the megakernel, where `main_pass` gates BDPT on
`MATERIAL_TYPE_FLAT` and routes every non-flat first hit to
`PathTracer.estimateRadiance`.

The fallback's `PathTracer.estimateRadiance` is a full multi-bounce path. On the
Metal backend the renderer SHALL bound it so no single committed command buffer
runs the multi-bounce fallback over more than one watchdog-safe band: when the
scene contains a non-terminal non-flat material (`MATERIAL_TYPE_VOLUME` or
`MATERIAL_TYPE_PYTHON`), (a) the wavefront BDPT/SPPM eye stage SHALL submit and
drain per eye tile (the row-band discipline of
`metal-megakernel-watchdog-tiling`), **and** (b) the eye `stream_size` (tile lane
count) SHALL be capped to the megakernel BDPT band budget so the tile itself — not
only tile accumulation — stays within the macOS GPU watchdog (a full-frame
`1<<20`-lane SPPM eye tile of multi-bounce volume walks would otherwise trip the
watchdog even with per-tile submit). Scenes with no non-terminal non-flat material
SHALL keep the single-submit path and full `stream_size` unchanged, and the Vulkan
backend (no watchdog) SHALL be behaviourally and byte-for-byte unchanged.

For wavefront BDPT the fallback lane SHALL build no eye subpath and no light
subpath of its own (matching a megakernel non-flat pixel, which runs no BDPT), and
SHALL still receive s=1 light-tracer splats contributed by other lanes. For
wavefront SPPM the fallback lane SHALL store no photon visible point and SHALL add
the path-traced radiance (weighted by the accumulated specular-chain throughput)
to the pixel. Flat-material lanes SHALL be unchanged, so scenes with no non-flat
material are byte-identical.

A `MATERIAL_TYPE_VOLUME` first hit under BDPT/SPPM shades via this path fallback,
but the bidirectional connection and photon strategies remain volume-blind (the
recorded medium-transport exclusion is unchanged).

#### Scenario: python-material object renders under wavefront BDPT and SPPM

- **WHEN** a scene with a `MATERIAL_TYPE_PYTHON` material
  (`cornell_box_python_material.usda`) is rendered with `--execution-mode
  wavefront` and `--integrator bdpt` or `--integrator sppm`
- **THEN** the python-material object shades (not black), within tolerance of the
  wavefront path anchor for the same scene

#### Scenario: volume first hit renders under wavefront BDPT/SPPM

- **WHEN** a scene with a `MATERIAL_TYPE_VOLUME` first hit is rendered under
  wavefront BDPT or SPPM
- **THEN** the eye-visible volume pixels shade via the path fallback (not black),
  and the render completes without wedging the GPU

#### Scenario: heavy-fallback frames stay within the GPU watchdog

- **WHEN** a non-terminal non-flat scene is rendered under wavefront BDPT/SPPM on
  Metal and the process is killed mid-render
- **THEN** the GPU remains usable afterwards (kill harness), because each committed
  command buffer bounded the multi-bounce fallback to one eye tile

#### Scenario: flat-only scenes are unchanged

- **WHEN** a scene containing only `MATERIAL_TYPE_FLAT` materials is rendered under
  wavefront BDPT or SPPM
- **THEN** every lane takes the flat path exactly as before this change, no
  per-tile flush is inserted, and the output is byte-identical

### Requirement: Wavefront drives the MLT chain sequence

The wavefront driver SHALL provide a fourth per-frame staged integrator
sequence for MLT alongside path, BDPT, and SPPM: chain mutation (advance each
chain's primary-sample state), subpath walks and connections through the
existing staged BDPT kernels consuming chain samples, acceptance + splat, and a
b-normalized splat resolve into the accumulation image. Chain and bootstrap
state SHALL live in persistent GPU buffers owned by the driver (allocated
through the existing suballocation path) and survive across accumulation
frames; the sequence SHALL run on both backends through the existing
backend-neutral recorder seam.

#### Scenario: MLT frame executes the staged sequence

- **WHEN** a frame is rendered with `(mlt, wavefront)`
- **THEN** the driver records mutation, walk/connection, acceptance/splat, and
  resolve stages in order, and the accumulation image advances by one
  b-normalized MLT pass

#### Scenario: Chain buffers persist across frames

- **WHEN** two consecutive frames render with no state-hash change
- **THEN** frame two's chains continue from frame one's accepted states (no
  re-bootstrap, no chain reset)

### Requirement: Wavefront path-state carries a spectrum under the spectral define

The wavefront's GPU-resident path-state and subpath carriers SHALL, under the
`SKINNY_SPECTRAL` compile-time variant, transport radiance/throughput as a hero-wavelength
spectrum and carry the path's sampled wavelengths, so that transport across the staged
dispatches is per-wavelength. The color-carrying fields SHALL use the shared `Spectrum` typealias (which is
`float3` in the RGB build and `float4` in the spectral build) so that the non-spectral stream
record layout, buffer sizing, and compiled SPIR-V are unchanged; the added per-lane sampled
wavelengths SHALL appear only under the spectral define. The Python stream-layout mirrors and
both the scalar and Metal (MSL) stride checks SHALL match the shader structs in both variants.

#### Scenario: RGB stream layout unchanged

- **WHEN** the wavefront kernels and stream buffers are built without the spectral define after
  the change lands
- **THEN** the path-state record stride, queue buffer sizes, and each non-spectral wavefront
  `.spv` are identical to their pre-change form

#### Scenario: spectral stream layout matches host and both backends

- **WHEN** the wavefront kernels are built with the spectral define
- **THEN** the widened path-state stride (Spectrum carriers plus sampled wavelengths) matches
  the `wavefront_layout` mirrors and both the scalar and MSL stride asserts, on Vulkan and Metal

#### Scenario: film resolves per wavelength at the resolve stage

- **WHEN** a spectral wavefront path folds a completed lane into the accumulation image
- **THEN** the lane's spectral radiance is resolved to linear sRGB via the shared CIE film
  resolve before the accumulation write, using the wavelengths drawn at that path's start

