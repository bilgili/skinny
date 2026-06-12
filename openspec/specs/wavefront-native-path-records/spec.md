# wavefront-native-path-records Specification

## Purpose
TBD - created by archiving change wavefront-native-path-records. Update Purpose after archive.
## Requirements
### Requirement: Wavefront-native path-record emission
The wavefront integrator SHALL emit per-vertex neural training records in the
shipped `PathRecord` layout (the same 64-byte record the megakernel record entry
and the `.nrec` dump use) into the existing append buffer and counter bindings, so
that online neural training can drain records from the wavefront backend without
dispatching the `mainImageRecord` megakernel.

#### Scenario: Guideable wavefront bounces produce records
- **WHEN** record emission is enabled and a wavefront path lane takes a guideable
  bounce (a flat/graph material, reflective, with the sampled direction in the
  upper hemisphere and a positive pdf)
- **THEN** a record carrying that vertex's position, normal, outgoing direction,
  sampled flow-local direction, and training-weight contribution is appended,
  byte-for-byte compatible with the shipped record reader and the offline dump

#### Scenario: The record stream matches the megakernel dump
- **WHEN** the same scene and proposal set are recorded once via the wavefront path
  and once via the megakernel `.nrec` dump on hardware where the megakernel runs
- **THEN** the two record sets are equivalent (the same vertices and contributions,
  independent of emission order)

### Requirement: Terminate-time backward attribution in wavefront
The wavefront integrator SHALL carry a bounded per-lane vertex stack and SHALL
compute each record's contribution weight at lane termination as
max((L_final - L_k) / beta_in_k, 0), matching the megakernel's per-thread
attribution exactly, and SHALL drop non-finite contributions. This is required
because the wavefront path is split across per-bounce dispatches, so the tail
radiance is not resident at the bounce that sampled the recorded direction.

#### Scenario: Contribution equals the megakernel attribution
- **WHEN** a lane terminates with final radiance L_final and a recorded vertex k
  snapshot (L_k, beta_in_k)
- **THEN** the emitted contribution is max((L_final - L_k) / beta_in_k, 0), the same
  value estimateRadianceRecord computes for that vertex

#### Scenario: The per-lane stack is bounded
- **WHEN** a lane records guideable bounces
- **THEN** at most `REC_MAX_BOUNCES` vertices are stacked per lane and the append is
  bounds-checked against the buffer capacity, never writing past the buffer

### Requirement: Record emission is gated and default-off
Record emission SHALL be enabled only while online neural training is active and
SHALL be off by default, so a wavefront render that is not training is unchanged —
no per-lane stack writes, no record appends, and no measurable record overhead on
the hot path.

#### Scenario: Default render is unchanged
- **WHEN** record emission is disabled (no online training)
- **THEN** the wavefront image is identical to the current wavefront backend and the
  record buffers are untouched

#### Scenario: Enabling record mode does not bias the render
- **WHEN** record emission is enabled during an online-training render
- **THEN** the rendered image remains unbiased — recording observes the path without
  altering the transport estimate

### Requirement: Online drain sources wavefront records
The renderer's live record drain SHALL be source-selectable, and SHALL default to
reading the wavefront-produced records (without dispatching the `mainImageRecord`
megakernel) when running the wavefront path integrator. The megakernel record
entry SHALL remain available both for the offline `.nrec` dump and as an
explicitly-selected live drain source on hardware without the megakernel watchdog
limitation.

#### Scenario: Online training drains without the megakernel
- **WHEN** online training is active on the wavefront backend with the path
  integrator and the default record source
- **THEN** the per-frame drain reads the records the wavefront render produced and
  appends them to the replay buffer, and the megakernel record pipeline is not built
  or dispatched on the per-frame path

#### Scenario: Megakernel remains a selectable live drain source
- **WHEN** the record source is explicitly set to the megakernel (e.g. on a box
  without the GPU watchdog limitation)
- **THEN** the live drain dispatches the `mainImageRecord` entry and reads its
  records, exactly as before this change

### Requirement: Record emission and drain run on the native Metal backend
The wavefront path-record emission and the live record drain SHALL run on the
native Metal backend with the same contracts as Vulkan: the shipped 64-byte
`PathRecord` layout into the existing append-buffer and counter bindings, the
bounded per-lane vertex stack with terminate-time backward attribution, and the
default-off gating (records exist only while online training is active). The
records-enabled build SHALL fit Metal's per-kernel buffer-slot cap by compiling
out wavefront-dead globals, and a build whose kernels exceed the cap SHALL fail
with a clear error naming the kernel and its slot count rather than a raw Metal
compile failure. With records disabled the Metal wavefront render SHALL be
byte-identical to the render before this capability.

#### Scenario: Metal wavefront bounces produce records
- **WHEN** record emission is enabled on the native Metal backend and a wavefront
  path lane takes a guideable bounce
- **THEN** a record is appended that parses with the shipped record reader and
  carries the same fields as a Vulkan-emitted record

#### Scenario: Metal and Vulkan record streams are equivalent
- **WHEN** the same scene, proposal set, and sample budget are recorded once on
  the Metal wavefront path and once on the Vulkan wavefront path
- **THEN** the two record sets are equivalent independent of emission order —
  identical record counts per bounce depth and matching value distributions
  within cross-backend float tolerance (MSL vs SPIR-V contraction drifts
  individual values in the ~1e-4 range, the same reason shaded wavefront
  parity is perceptual rather than bit-exact)

#### Scenario: Records-off Metal render is unchanged
- **WHEN** the Metal wavefront render runs without online training
- **THEN** the image is byte-identical to the pre-records Metal wavefront render
  and the record buffers are untouched

#### Scenario: Slot-cap overflow fails clearly
- **WHEN** a records-enabled Metal build produces a kernel whose buffer bindings
  exceed the argument-table cap
- **THEN** the build fails with an error naming the kernel and its slot count,
  not a downstream Metal pipeline error

#### Scenario: Live drain reads Metal records without the megakernel
- **WHEN** online training is active on the Metal wavefront path integrator
- **THEN** the per-frame drain reads the counter and records the wavefront render
  produced and appends them to the replay buffer, with no megakernel record
  pipeline built or dispatched
