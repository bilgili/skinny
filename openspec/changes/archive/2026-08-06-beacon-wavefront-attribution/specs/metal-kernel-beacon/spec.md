# Metal kernel beacon — wavefront attribution delta

## MODIFIED Requirements

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

## ADDED Requirements

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
