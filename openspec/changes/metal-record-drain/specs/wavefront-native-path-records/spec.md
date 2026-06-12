# wavefront-native-path-records — Delta (metal-record-drain)

## ADDED Requirements

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
- **THEN** the two record sets are equivalent — the same vertices and
  contributions, independent of emission order

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
