## ADDED Requirements

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
The renderer's live record drain SHALL read the wavefront-produced records when
record mode is enabled, without dispatching the `mainImageRecord` megakernel, while
the megakernel record entry and the file dump remain available for the offline
`.nrec` path.

#### Scenario: Online training drains without the megakernel
- **WHEN** online training is active on the wavefront backend
- **THEN** the per-frame drain reads the records the wavefront render produced and
  appends them to the replay buffer, and the megakernel record pipeline is not built
  or dispatched on the per-frame path
