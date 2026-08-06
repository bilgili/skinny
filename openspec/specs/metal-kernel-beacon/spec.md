# metal-kernel-beacon Specification

## Purpose
TBD - created by archiving change metal-kernel-beacon. Update Purpose after archive.
## Requirements
### Requirement: A shared beacon buffer records the running kernel identity

The Metal backend SHALL allocate one 256-byte shared-storage beacon buffer and
bind it to every compute kernel it dispatches: the megakernel, every wavefront
stage kernel, the preview kernel, and the debug-raster kernel. The buffer SHALL
use Apple-Silicon UMA shared storage, so the host reads it with no readback copy.
The buffer SHALL carry a fixed byte layout: a magic word, a monotonic sequence
number, a kernel-id word, a reserved phase word, a reserved trip word, and a
mirror of the sequence number for torn-read detection.

The beacon buffer SHALL be bound at a fixed Metal shader-global name. The Vulkan
backend SHALL NOT allocate or bind the beacon buffer. The beacon buffer SHALL
occupy a Metal argument-table slot ONLY when the `SKINNY_METAL_TRACE` build gate
is on, so the production Metal argument-table budget is unchanged.

#### Scenario: the beacon buffer binds to every compute kernel under the trace gate
- **WHEN** the Metal backend builds its compute kernels with `SKINNY_METAL_TRACE`
  on
- **THEN** the megakernel, every wavefront stage kernel, the preview kernel, and
  the debug-raster kernel each bind the same 256-byte beacon buffer at the fixed
  shader-global name

#### Scenario: production builds do not carry the beacon buffer
- **WHEN** the Metal backend builds its compute kernels with `SKINNY_METAL_TRACE`
  off (the default)
- **THEN** no kernel binds the beacon buffer, the Metal argument-table slot count
  is unchanged, and the Metal shader binaries are byte-identical to the current
  production binaries

### Requirement: The trace gate keeps production shaders byte-identical

The gate SHALL compile the shader beacon helper, every `beacon.store` call, and
the `gKernelBeacon` buffer declaration in `bindings.slang` only under
`#if defined(SKINNY_METAL_TRACE)`. The host-side allocation and bind of the
beacon buffer SHALL be gated the same way. The declaration itself — not only the
store calls — SHALL sit behind the gate, because an ungated-but-unused Metal
global still consumes an argument-table slot. The gate SHALL default off. The
Vulkan SPIR-V SHALL never carry the beacon helper, because the gate is a
Metal-only axis. When the gate is off, the emitted Metal binaries and the emitted
Vulkan SPIR-V SHALL be byte-identical to the current production output.

#### Scenario: Vulkan SPIR-V is byte-unchanged
- **WHEN** the beacon helper and its gate are added to the shared shader sources
- **THEN** every Vulkan SPIR-V binary the build produces is byte-identical to the
  binary produced before this change

#### Scenario: default Metal binary is byte-unchanged
- **WHEN** the Metal backend builds with the default gate state (off)
- **THEN** every Metal kernel binary is byte-identical to the binary produced
  before this change

### Requirement: The shader helper stores the kernel id at kernel entry

Under `SKINNY_METAL_TRACE`, every compute entry point SHALL call
`beacon.store(KERNEL_ID)` as its first statement, where `KERNEL_ID` is the
kernel's static integer id. The store SHALL write the kernel id into the
kernel-id word of the beacon buffer. In this change the store SHALL write 0 into
the reserved phase word and 0 into the reserved trip word. The reserved words
SHALL stay in the layout so a later change can write per-loop trip counts without
an interface change.

#### Scenario: a running kernel writes its id to the beacon
- **WHEN** a traced kernel begins executing on the device
- **THEN** the beacon buffer's kernel-id word holds that kernel's static id, and
  its phase and trip words hold 0

### Requirement: The child stamps the kernel id before each dispatch

The child SHALL stamp the kernel's id into a memory-mapped beacon cell and flush
it before it submits the command buffer that runs that kernel. GPU work runs in a
child process, per `metal-dispatch-hygiene`. The stamp SHALL write the kernel-id
word and increment the sequence number, and SHALL bracket the write with the
sequence-mirror word so a concurrent reader detects a torn read.

The stamp SHALL bracket exactly one in-flight kernel per committed command
buffer, so the last stamp before a wedge names the kernel that hung. The dispatch
paths meet this rule as follows:

- The synchronous single-shot wrappers and the megakernel stamp the kernel id,
  then submit that one kernel, then drain. One kernel runs per command buffer, so
  the stamp always names the in-flight kernel.
- The wavefront `MetalFrameEncoder` batches many kernels into one command buffer
  and submits once on the production path (trace off). It stamps at ENCODE time,
  so a batch of kernels overwrites the cell before the single submit. Under the
  `SKINNY_METAL_TRACE` gate the encoder degrades to per-kernel submit (see the
  requirement below), so the stamp again brackets exactly one in-flight kernel.

The child SHALL stamp BEFORE it submits, because `wait_for_idle()` never returns
for an infinite kernel and the child cannot flush after the wedge.

#### Scenario: the stamp lands before a hanging dispatch
- **WHEN** the child stamps kernel N and then submits a command buffer whose only
  kernel is N and that never completes
- **THEN** the memory-mapped beacon cell holds kernel N's id and a valid
  sequence pair, even though the child is now blocked in `wait_for_idle()`

### Requirement: The parent polls the beacon and reports the hung kernel

The parent SHALL poll the memory-mapped beacon cell on a wall-clock loop. The
poll SHALL return a report only when the cell's magic word is valid and its two
sequence words are equal; otherwise the poll SHALL return "no valid beacon" and
SHALL NOT raise. On a wall-clock timeout the parent SHALL send SIGTERM to the
child, wait the grace period, read the last valid beacon cell, and report the
kernel id together with its name from the kernel-identity table. An id absent
from the table SHALL report as an "unknown id" name and SHALL NOT raise.

The reported kernel id SHALL name the in-flight kernel ONLY when one kernel ran
per committed command buffer. On the megakernel path and the synchronous
single-shot path this always holds. On the wavefront path this holds ONLY under
the `SKINNY_METAL_TRACE` per-kernel-submit rule; on the production batched path
(trace off) the report names the last-encoded kernel of the in-flight batch,
which need not be the kernel that hung. The report SHALL NOT be described as
naming the in-flight wavefront kernel unless trace is on.

#### Scenario: a stalled kernel is reported by name
- **WHEN** the child hangs in a command buffer whose only kernel is N and the
  parent's wall-clock timeout expires
- **THEN** the parent sends SIGTERM, then reads the beacon cell and reports
  `kernel_id=N` with kernel N's entry-point name

#### Scenario: a torn or uninitialized cell is not reported as a kernel
- **WHEN** the parent reads a beacon cell whose magic is wrong or whose two
  sequence words differ
- **THEN** the read returns "no valid beacon" instead of a kernel id, and the
  parent does not raise

### Requirement: The parent never reports a kernel the child never reached

The parent's reported kernel id SHALL always be a kernel the child actually
reached. The beacon cell SHALL hold a kernel id only after the child stamps that
kernel before its dispatch. The parent SHALL treat a torn read or a magic
mismatch as "no valid beacon", so it never reports a partial, stale-composite, or
never-written value as a kernel id. The parent SHALL never report a kernel id the
child has not yet stamped.

#### Scenario: no phantom report
- **WHEN** the parent produces a kernel-id report
- **THEN** that kernel id equals a kernel the child stamped before a dispatch —
  never an id the child never wrote and never a torn composite of two ids

### Requirement: A static table maps kernel ids to entry-point names

The change SHALL define one static kernel-identity table that maps each integer
kernel id to its entry-point name. The table SHALL cover the megakernel
(`mainImage`), every wavefront stage kernel named in `vk_wavefront.py`, the
preview kernel, and the debug-raster kernel. Id 0 SHALL be reserved to mean "no
kernel". Ids SHALL be append-only: a new kernel SHALL take the next free id, and
an existing id SHALL NOT be renumbered or reused. A pinned test SHALL enforce the
id-to-name mapping.

#### Scenario: every dispatched kernel has a table entry
- **WHEN** the Metal backend can dispatch a compute kernel
- **THEN** that kernel's entry-point name has an id in the kernel-identity table,
  and id 0 maps to the "no kernel" name

#### Scenario: ids are append-only
- **WHEN** a new kernel is added to the table
- **THEN** it takes the next free id, and no existing id changes its name

### Requirement: The traced wavefront encoder submits one kernel per command buffer

Under the `SKINNY_METAL_TRACE` gate, the wavefront `MetalFrameEncoder` SHALL
submit and drain each dispatch as its own command buffer. For each encoded
kernel the encoder SHALL stamp the mmap cell, submit that one kernel, and drain
(`wait_for_idle`) before it encodes the next kernel. At most one wavefront kernel
SHALL be in flight per committed command buffer while trace is on. This is the
invariant that makes the mmap cell name the in-flight — possibly hung — kernel.

With the gate off (the production default), the encoder SHALL keep its batched
single-submit behavior unchanged. The encoder SHALL accumulate the frame's
dispatches into one command buffer and submit once, exactly as before this
change. The emitted MSL and the emitted SPIR-V SHALL be byte-identical to the
production output, because this change adds no shader code and no new build gate.

The wavefront recorders and the wavefront driver SHALL be unchanged. They drive
every dispatch through the encoder surface, so the trace-gated branch sits below
them in `MetalFrameEncoder`.

#### Scenario: trace on submits one wavefront kernel at a time
- **WHEN** the wavefront driver records a bounce loop while `SKINNY_METAL_TRACE`
  is on
- **THEN** each stage kernel is stamped, submitted as its own command buffer, and
  drained before the next stage is encoded, so at most one wavefront kernel is in
  flight per command buffer

#### Scenario: trace off keeps the batched single submit
- **WHEN** the wavefront driver records a bounce loop while `SKINNY_METAL_TRACE`
  is off
- **THEN** the encoder batches the stages into one command buffer and submits
  once, and the emitted MSL and SPIR-V are byte-identical to the production output

#### Scenario: a wedged wavefront kernel is named under trace
- **WHEN** a gpu-marked test drives the actual wavefront per-kernel-submit path
  with `SKINNY_METAL_TRACE` on, and one wavefront stage kernel hangs
- **THEN** the parent reports that in-flight stage kernel by name, not a
  later-encoded stage of the same frame

### Requirement: Beacon accuracy is stated per dispatch path

The beacon documentation and spec text SHALL state the report accuracy per
dispatch path. The megakernel and the synchronous single-shot wrappers name the
EXACT in-flight kernel, because one kernel runs per command buffer. The wavefront
path names the exact in-flight kernel ONLY under `SKINNY_METAL_TRACE`, through
per-kernel submit. No spec, doc, or code comment SHALL claim an un-qualified
"names the stuck kernel" for the batched wavefront path.

#### Scenario: no un-qualified wavefront accuracy claim
- **WHEN** a doc or spec describes the beacon report on the wavefront path
- **THEN** it states that the report names the in-flight kernel only under trace,
  and that the batched path names the last-encoded kernel of the in-flight batch

### Requirement: Context teardown releases the beacon buffer

`MetalContext.destroy()` SHALL release the lazily-created `_beacon_buffer` when
it releases the beacon writer. The release SHALL be idempotent and safe when the
buffer was never created (trace off, or trace on but no dispatch ran). Teardown
SHALL be complete per `metal-dispatch-hygiene`, so no beacon resource outlives
the context.

#### Scenario: destroy clears the beacon buffer
- **WHEN** a traced `MetalContext` that created its beacon buffer is destroyed
- **THEN** `destroy()` releases the beacon buffer and clears its handle, and a
  second `destroy()` call is a safe no-op

### Requirement: The seqlock write documents its quiescence assumption

The seqlock write in the beacon writer SHALL carry a code comment that states the
guard rejects gross tearing only, not a payload straddle, and that correctness
rests on writer quiescence at read time. The parent reads the cell only after it
SIGTERMs the wedged child, at which point the child is quiescent inside
`wait_for_idle` or already terminated, so no partial payload can straddle the
read. Behavior SHALL be unchanged; this is a documentation requirement only.

#### Scenario: the seqlock comment records the straddle limit
- **WHEN** a reader inspects the seqlock write in the beacon writer
- **THEN** a comment states that the guard catches gross tearing, not a payload
  straddle, and that correctness rests on writer quiescence at read time

