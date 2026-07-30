# renderer-module-structure Specification

## Purpose
TBD - created by archiving change renderer-module-carveout. Update Purpose after archive.
## Requirements
### Requirement: MLT chain-state orchestration lives in a dedicated module with a hostless-testable pure core

The renderer's MLT host orchestration SHALL live in a dedicated module
outside `renderer.py`, alongside the existing pure `mlt_bootstrap.py` core —
covering the per-reset replay seed derivation, the
mutation-iterations-per-frame budget, the uniform-tail-active predicate, and
the bootstrap round-trip sequencing (bootstrap dispatch → weight readback →
chain-seed resample → seed upload → chain-init dispatch → `b` publication).
The seed derivation, iterations budget, and tail predicate MUST be pure
functions importable and testable without a GPU device or a constructed
renderer, and the seed formula MUST remain numerically identical to the
current implementation (its cross-process reproducibility underpins the
parity gate's MLT determinism). Both backends SHALL drive the bootstrap
round-trip through the single shared sequence, supplying only their own
submit/upload primitives.

#### Scenario: Pure MLT core is hostless-testable

- **WHEN** the MLT chain-state module's seed, iterations-per-frame, and
  uniform-tail-predicate functions are imported in a test process with no GPU
  device and no `Renderer` instance
- **THEN** they evaluate successfully, and the seed function returns exactly
  the same integers as the pre-carve-out `Renderer._next_mlt_seed` for the
  same `frame_index` values

#### Scenario: MLT rendering is bit-identical after the extraction

- **WHEN** an MLT suite scene is rendered at equal budget on either backend
  before and after the MLT chain-state extraction
- **THEN** the two images are bit-identical (same seed, same chains, same
  splats), and the parity matrix MLT combos pass with unchanged measured
  values and unchanged tolerances

### Requirement: Frame-constant derivation is pure and separate from byte packing

Every derived value in the frame-constant packing path SHALL be computed by
pure, device-free functions in a dedicated module — camera view/projection
inverses, the lens FOV-framing ratio and sensor half-height, the detail-flag
bitfield, the exposure/imaging-ratio fold, the emissive total power, and the
proposal-mask/reuse capability folding (including the megakernel neural-bit
strip and mixture renormalisation) — with the packing method consuming their
results as plain values. The packing method MUST NOT gain or
lose side effects, and the packed byte stream MUST remain byte-identical for
every state combination. Serialization itself (append order, offsets,
reflection-driven MSL relocation) remains the scope of the
`reflection-owned-byte-layouts` change and is out of scope here.

#### Scenario: Derivation is hostless-testable

- **WHEN** the derivation module's functions (detail flags, lens framing,
  exposure fold, sampling-capability folding) are exercised in a test process
  with no GPU device
- **THEN** they return correct values for representative inputs — including
  the lens-active framing ratio, the missing-map detail-flag masking, and the
  neural-on-megakernel strip-and-renormalise case — without constructing a
  `Renderer`

#### Scenario: Packed uniforms are byte-identical

- **WHEN** the frame-constant blob is packed before and after the derivation
  extraction for a state matrix covering lens on/off, detail maps on/off,
  each integrator, both execution modes, and the neural-on-megakernel case
- **THEN** the packed bytes are equal in every case, on both the Vulkan and
  the Metal (reflected MSL) packing paths

### Requirement: Wavefront integrator pass construction and dispatch route through the pass-object seam

Construction of the staged wavefront passes (path, BDPT, SPPM, MLT) SHALL be
supplied by per-backend factories in the backend modules
(`vk_wavefront` / `metal_wavefront`), and the renderer SHALL hold one
ensure/cache path per integrator — keyed by the existing rebuild keys, whose
values MUST NOT change — instead of per-backend `_ensure_*` /
`_ensure_*_metal` method pairs. Per-frame dispatch SHALL go through the
existing pass-object surfaces (`record_dispatch`/`record_frame` on Vulkan,
`dispatch_frame` on Metal) with backend divergence confined to the backend
adapters, mirroring the `wavefront_driver.WavefrontRecorder` precedent. The
ensure path MUST preserve every existing unbuildable-pass None fallback
(e.g. MLT/SPPM in a megakernel-mode session fall back to the path tracer,
never crash) exactly as before the move. This
requirement reduces renderer-resident `is_metal`/`_metal` sites; it MUST NOT
alter the metal-backend capability's mandate that Vulkan-only paths
short-circuit safely on Metal, and MUST NOT introduce a new backend
abstraction layer beyond the existing duck-typed surfaces.

#### Scenario: One ensure path per integrator

- **WHEN** the renderer builds or rebuilds a staged wavefront pass for any
  integrator on either backend
- **THEN** it does so through a single backend-agnostic ensure path calling
  the active backend's pass factory, and the rebuild key computed for the
  pass equals the pre-carve-out key value for the same renderer state

#### Scenario: Wavefront rendering is bit-identical across the seam move

- **WHEN** the parity matrix wavefront combos (path, BDPT, SPPM, MLT; RGB and
  spectral) run on both backends after the pass-seam extraction
- **THEN** every combo passes its pbrt-truth and self-consistency gates with
  unchanged measured values, unchanged baselines, and unchanged tolerances

### Requirement: Every carve-out stage lands independently with bit-identity gates

Each extraction stage of the renderer module carve-out SHALL be an
independently landable unit (one PR-able task group) that leaves the renderer
fully functional, and SHALL be gated before merge by: no modification to any
file under `src/skinny/shaders/` (hence RGB `.spv` byte-unchanged), a green
parity matrix with no baseline or tolerance edits, and the stage's specific
bit-identity check (golden byte equality or bit-identical images). Follow-on
clusters (USD live-edit, gizmo overlay, detail maps) SHALL be extracted in
subsequent OpenSpec changes following the documented pattern, not in this
change.

#### Scenario: A stage merges under the bit-identity gate

- **WHEN** a carve-out stage is proposed for merge
- **THEN** its diff touches no shader source, the parity matrix passes with
  unchanged recorded values, and the stage's golden-byte or bit-identical
  image check passes on both backends

#### Scenario: Follow-on clusters are deferred, not smuggled

- **WHEN** the final carve-out stage of this change is complete
- **THEN** the USD live-edit, gizmo overlay, and detail-map clusters remain
  in `renderer.py`, and the documented extraction pattern names them, their
  ordering, and the gate each future change must carry

### Requirement: GPU resource allocation, binding and destruction live in one module with paired declarations

The renderer's GPU resource inventory SHALL live in a dedicated module outside
`renderer.py`, in which each resource is declared once and that single
declaration carries its allocation inputs, its binding identity on both
backends (Vulkan descriptor binding number, Metal shader-global name, either
optionally absent), and its destruction. The module SHALL absorb `_init_gpu`,
`_create_descriptors`, the five `_rebind_*_descriptors` methods,
`_rewrite_size_dependent_descriptors`, `_ensure_mesh_buffer_capacity`,
`_build_metal_binds`, and the resource-destroy body of `cleanup`. Backend
divergence SHALL be confined to one binding step consuming the shared
declaration list — the per-method `is_metal` / `descriptor_sets is None`
early-returns MUST NOT survive the move. The resulting GPU state MUST be
identical to the pre-change renderer: same resources, same sizes and formats,
same binding numbers, and the same descriptor-write order.

#### Scenario: Every allocated resource is destroyed

- **WHEN** the resource set is constructed against a recording context and
  then closed
- **THEN** the set of resources destroyed equals the set allocated, with no
  resource allocated twice and none left undestroyed

#### Scenario: One declaration feeds both backends

- **WHEN** the set is bound for the Vulkan target and for the Metal target
- **THEN** both bindings are derived from the same declaration list, the
  Vulkan binding numbers are unique, the Metal global names are unique, and
  the two cover the same declarations modulo declarations explicitly marked as
  absent on one target

#### Scenario: Inventory matches the pre-change renderer

- **WHEN** the resource set's declarations are compared against the recorded
  pre-change inventory fixture — name, kind, size inputs, format, binding
  number and descriptor-write order captured from `_init_gpu` and
  `_create_descriptors` before the move
- **THEN** they match entry for entry, including order

#### Scenario: Growth reflows bindings through the set

- **WHEN** a resource is reallocated because its capacity grew — mesh buffers,
  the volume grid, the bindless texture pool, or a size-dependent image after
  a viewport resize
- **THEN** the rebind that follows is performed by the resource set from the
  same declaration, and no call site outside the set rewrites a descriptor

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

### Requirement: The per-frame path is scene sync, a pure frame plan, and execution

The renderer's per-frame path SHALL be split into three stages: scene sync
(the state-advancing work currently in `update`), a **pure** frame plan that
derives the frame's decisions as an inspectable value, and execution of that
plan against a target. The plan SHALL name the execution mode, the pass
sequence, the accumulation state and reset decision, any dispatch banding or
tiling, and which optional per-frame work is performed — and MUST hold no
device handles, so it can be derived and asserted with no GPU present. The
windowed and headless paths SHALL share one execution body and differ only in
their target; the barrier, execution-mode gate and dispatch block that are
currently duplicated between them MUST NOT remain duplicated. The plan SHALL
consume the accumulation reset decision from the parameter-registry owner
rather than re-deriving it. Dispatch sequence and rendered images MUST be
unchanged.

#### Scenario: The frame plan is derived without a device

- **WHEN** a frame plan is derived from renderer state in a process with no
  GPU device
- **THEN** it is produced and its pass sequence, execution mode and
  accumulation decision can be asserted

#### Scenario: Windowed and headless share one dispatch body

- **WHEN** the same frame plan is executed against a windowed target and an
  offscreen target
- **THEN** the recorded dispatch sequence is identical, and the two paths
  differ only in output destination, swapchain acquisition and presentation,
  and readback

#### Scenario: Ordering constraints are asserted, not implied

- **WHEN** the plan's step order is inspected
- **THEN** the constraints that are currently implicit in line order — notably
  that the pick-result drain precedes uniform packing — are expressed in the
  plan and asserted by test

#### Scenario: Images are unchanged by the split

- **WHEN** the parity matrix's pbrt-truth and self-consistency gates run
  before and after the split
- **THEN** the results are identical, not merely within tolerance

