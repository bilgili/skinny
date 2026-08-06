# Design — beacon-wavefront-attribution

## Problem and constraint

The `metal-kernel-beacon` change reports the hung kernel by name. It is correct
on the synchronous single-shot path and the megakernel path, where one kernel
runs per committed command buffer. It is WRONG on the wavefront path.

The wavefront path records many kernels into one `MetalFrameEncoder` command
buffer and submits once (`MetalFrameEncoder.dispatch` in `metal_compute.py`; the
recorders in `metal_wavefront.py`; the flush points in `wavefront_driver.py`).
The host `beacon_stamp` fires at ENCODE time
(`MetalFrameEncoder.dispatch` calls `self.ctx.beacon_stamp(...)` before
`begin_compute_pass`). Each kernel in the batch overwrites the mmap cell before
the single submit. When the GPU wedges on kernel `k` mid-batch, the mmap holds
the LAST-ENCODED kernel of the batch, not `k`.

The GPU `gKernelBeacon` buffer writes per kernel, but it lives in the child's UMA
address space. The parent reads only the mmap FILE. Nothing copies the GPU buffer
to the file, and the child is wedged in `wait_for_idle` or already SIGTERM'd, so
the per-kernel GPU write never reaches the parent.

The motivating hang — MLT-spectral — is WAVEFRONT-ONLY, so the wrong path is the
one that matters most.

## Ownership and seam

**Owner.** `MetalFrameEncoder` in `metal_compute.py` owns the submit granularity.
The trace-gated branch lives in `MetalFrameEncoder.dispatch` and
`MetalFrameEncoder.dispatch_indirect`. `MetalContext` owns the beacon-buffer
release in `destroy()`. `BeaconWriter` in `metal_beacon.py` owns the seqlock
comment.

**Seam.** One trace-gated branch inside the encoder decides submit granularity.
When `ctx.trace` is off the encoder batches and submits once (production, today).
When `ctx.trace` is on the encoder submits and drains after each dispatch.

**Why the seam is inside the encoder.** The wavefront recorders
(`_MetalWavefrontRecorder`, `_MetalSppmRecorder`, and the MLT recorder) and the
driver `record_path_loop` / `record_sppm_loop` drive EVERY dispatch through the
encoder surface (`enc.dispatch`, `enc.barrier`, `enc.flush`, `enc.submit`). One
branch in `MetalFrameEncoder.dispatch` covers all of them. The recorders and the
driver need NO change. This is the single-owner fix: change the shared dispatch
method, not every recorder.

## Data flow (trace on)

For each wavefront stage kernel `k`:

1. The recorder calls `enc.dispatch(pipe_k, ...)`.
2. `dispatch` stamps the mmap cell with `k` (unchanged, already before encode).
3. `dispatch` encodes `k` into the encoder, then — because `ctx.trace` is on —
   submits that one command buffer and drains (`wait_for_idle`), then reopens a
   fresh encoder.
4. A healthy `k` returns; the recorder encodes the next stage. A wedged `k` never
   returns; the child hangs with the mmap cell holding exactly `k`.
5. The parent times out, SIGTERMs the child (the chained handler runs
   `destroy()`), waits the grace period, reads the cell, and reports `k`.

Because at most one kernel is in flight per command buffer under trace, the mmap
cell always names the in-flight — possibly hung — kernel. This reuses the proven
synchronous mechanism the megakernel and the single-shot wrappers already use.

## Data flow (trace off, production)

`enc.dispatch` stamps and encodes as today, and does NOT submit. The frame's
stages accumulate in one encoder; `barrier()` inserts the compute-memory barrier
between stages; `submit()` submits and drains once at frame end. This path is
byte-identical to today. The stamp still fires, but the cell holds the
last-encoded kernel, which the spec now states plainly is not the in-flight
kernel on the batched path.

## Frozen interfaces

The following interfaces are FROZEN once the design gate approves.

### `MetalFrameEncoder.dispatch` — trace-gated submit granularity

Signature is UNCHANGED:

```python
def dispatch(self, pipe: ComputePipeline, groups, *, bindings=None,
             uniform_blob=None, uniforms=None, bindless=None) -> None: ...
```

Behavior:

- The method stamps the mmap cell (`self.ctx.beacon_stamp(_pipe_entry(pipe))`),
  builds the root object, and encodes one compute pass — UNCHANGED.
- **When `self.ctx.trace` is False**: the method returns after `cpass.end()`
  WITHOUT submitting. The frame accumulates into one encoder and `submit()`
  submits once. This is byte-identical to today.
- **When `self.ctx.trace` is True**: after `cpass.end()` the method submits the
  current encoder and drains, then reopens a fresh encoder — the body of the
  existing `flush()`. Exactly one kernel is in flight per committed command
  buffer.

### `MetalFrameEncoder.dispatch_indirect` — same trace-gated rule

Signature is UNCHANGED:

```python
def dispatch_indirect(self, pipe: ComputePipeline, args_buffer, offset: int = 0,
                      *, bindings=None, uniform_blob=None, uniforms=None,
                      bindless=None) -> None: ...
```

Behavior: identical trace-gated rule. Trace off encodes only; trace on submits +
drains + reopens after the encoded indirect dispatch.

### `MetalFrameEncoder.barrier`, `.flush`, `.submit` — unchanged

- `barrier()` still inserts a global compute-memory barrier. Under trace the
  encoder is fresh and empty when `barrier()` runs (the prior `dispatch` already
  submitted), so the barrier is harmless — the drain already ordered the stages.
- `flush()` is unchanged (submit + drain + reopen). Under trace it is a no-op on
  an already-empty encoder.
- `submit()` is unchanged. Under trace it finishes and submits a possibly-empty
  encoder, then marks `_submitted`. An empty submit is safe.

The recorders and the driver call these methods verbatim, so they are unchanged.

### `MetalContext.destroy` — beacon-buffer release contract

`destroy()` SHALL release `_beacon_buffer` when it releases `_beacon_writer`,
inside the same idempotent guard. The release calls the buffer's `destroy()` (or
drops the reference so SlangPy refcount frees it) and sets `self._beacon_buffer =
None`. The release SHALL be safe when `_beacon_buffer` is `None` (trace off, or
trace on but no dispatch ran, so `beacon_native` was never touched). Repeated
`destroy()` calls stay safe no-ops. No signature change; `destroy()` stays
`def destroy(self) -> None`.

### `BeaconWriter` seqlock comment — documentation contract

The seqlock write in `BeaconWriter.stamp` SHALL carry a code comment that states:
the sequence-mirror guard catches gross tearing, not a payload straddle, and
correctness rests on writer quiescence at read time. The parent reads only after
it SIGTERMs the wedged child, so the writer is quiescent (blocked in
`wait_for_idle` or terminated) and no partial payload can straddle the read.
Behavior is UNCHANGED.

## Why not design A (discarded alternative)

**Design A — the shared-memory GPU beacon.** Back `gKernelBeacon` with POSIX
shared memory that the parent also maps. The GPU writes its per-kernel id into
the shared buffer, the write survives the wedge, and the parent reads the
GPU-written bytes directly. Wavefront batching stays on, so no per-kernel submit
and no trace-build slowdown.

Rejected for three reasons:

1. It depends on slang-rhi creating a Metal buffer over a host pointer
   (`MTLDevice.makeBuffer(bytesNoCopy:...)`). This surface is unverified and
   likely absent in the SlangPy / slang-rhi Python API. The whole beacon design
   exists because slang-rhi hides the raw Metal objects.
2. It adds a NEW cross-process shared-memory data-flow contract, and it makes the
   parent read GPU-written memory. That is a heavier, less proven path than the
   mmap file the child already writes and the parent already reads.
3. It is heavier than reusing the proven mmap path. Design B changes one branch
   in one method and touches no shader code.

Design A is worth revisiting only if the per-kernel-submit slowdown under trace
ever becomes a real constraint. It is not — trace builds run only for
hang-hunting, where throughput does not matter, and per-kernel bounded command
buffers align with `metal-dispatch-hygiene`.

## Safety and liveness argument (modeled in TLA+)

The safety property is `NoMisattribution`: after the parent reads the cell, the
reported kernel equals the in-flight (hung) kernel. The TLA+ module
`specs/tla/BeaconWavefrontAttribution.tla` models both submit modes through a
`Trace` flag:

- **Batched mode (`Trace = FALSE`)**: the encoder stamps at encode, accumulates a
  batch, and submits once. A ghost `inflight` picks ANY kernel of the in-flight
  batch to hang. The cell holds the LAST-ENCODED kernel. The checker finds a
  counterexample where a non-last kernel hangs, so `reported ≠ inflight`.
  `NoMisattribution` is VIOLATED. The weaker `NoPhantomReport` (reported is some
  reached kernel) still HOLDS, which is why the bug is a misattribution among
  reached kernels, not a phantom.
- **Per-kernel-submit mode (`Trace = TRUE`)**: the encoder may not encode a second
  kernel while one is in flight, so the batch holds one kernel and the cell holds
  exactly the in-flight kernel. `reported = inflight` always. `NoMisattribution`
  is SATISFIED.

The liveness property `HungIsReported` states a hung kernel is eventually
reported, through the SIGTERM → read chain, under weak fairness on the parent's
timeout, SIGTERM, and read actions. The invariant `WriterQuiescentWhenHung`
formalizes the seqlock comment: while a kernel is hung the seqlock is closed
(`cellTorn = FALSE`), so the parent reads a committed cell.

The file uses the TLA-valid name `BeaconWavefrontAttribution.tla`, because SANY
requires the filename to match the module identifier and TLA+ identifiers forbid
the hyphen in the change id. The design gate owns the `.cfg` beside it and runs
the module under `Trace = FALSE` (to see the violation) and `Trace = TRUE` (to
see the fix).

## Bounds and assumptions (TLA+ model)

- The kernel batch is a fixed short sequence (`KernelSeqDef == <<1, 2, 3>>`), so a
  batch of 3 wavefront kernels.
- The parent's wall clock is longer than any healthy kernel, so it times out only
  on a genuine hang. This is modeled by gating the timeout on `phase = "hung"`.
- Kernel ids are >= 1; id 0 (`NONE`) means "no kernel".
