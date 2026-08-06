# Beacon wavefront attribution (name the stuck wavefront kernel)

## Why

The shipped `metal-kernel-beacon` change reports the hung kernel by name. A
pre-merge review found the report is correct on ONE path only. On the
synchronous single-shot path and the megakernel path the beacon names the exact
stuck kernel. On the WAVEFRONT path it names the WRONG kernel.

The root cause is the batched encoder. Wavefront mode records many kernels into
one `MetalFrameEncoder` command buffer and submits ONCE
(`src/skinny/metal_compute.py` `MetalFrameEncoder.dispatch`;
`src/skinny/metal_wavefront.py` recorders; `src/skinny/wavefront_driver.py` flush
points). The host `beacon_stamp` fires at ENCODE time. Each kernel in the batch
overwrites the mmap cell before the single submit. When the GPU wedges on kernel
`k` mid-batch, the mmap holds the LAST-ENCODED kernel of the batch, not `k`.

The GPU `gKernelBeacon` buffer does write per kernel, but it lives in the child's
UMA address space. The parent reads only the mmap FILE, and nothing copies the
GPU buffer to that file. The child is wedged in `wait_for_idle` or already
SIGTERM'd, so the per-kernel GPU signal never reaches the parent.

This matters because the motivating hang — MLT-spectral — is WAVEFRONT-ONLY. The
one path where the operator most needs the kernel name is the one path that names
the wrong kernel.

## What Changes

Under the `SKINNY_METAL_TRACE` gate ONLY, `MetalFrameEncoder` degrades to
SYNCHRONOUS PER-KERNEL SUBMIT. For each encoded kernel the encoder stamps the
mmap cell, submits that ONE kernel as its own command buffer, and drains
(`wait_for_idle`) before it encodes the next. The existing mmap host stamp thus
becomes per-kernel-accurate on the wavefront path. This reuses the proven
synchronous mechanism. It adds no dependency and no new cross-process contract.

- The seam is one trace-gated branch inside `MetalFrameEncoder.dispatch` and
  `MetalFrameEncoder.dispatch_indirect`. When `ctx.trace` is off the encoder
  batches and submits once, exactly as today. When `ctx.trace` is on the encoder
  submits and drains after each dispatch.
- The wavefront recorders in `metal_wavefront.py` and the driver in
  `wavefront_driver.py` are UNCHANGED. They already drive every dispatch through
  the encoder surface, so the seam sits below them.
- Production (gate off) behavior is byte-identical. The MSL and the SPIR-V are
  unchanged. The batched single submit and its performance are unchanged. Only
  trace builds pay the per-kernel-submit slowdown, which is irrelevant for
  hang-hunting.
- The per-kernel bounded command buffer ALIGNS with `metal-dispatch-hygiene`,
  which wants each dispatch to finish or hang alone.

Three review findings fold in with this change:

- **Doc correction.** State the beacon accuracy PER PATH. The megakernel and the
  synchronous single-shot wrappers name the exact kernel. The wavefront path
  names the exact kernel ONLY under trace, through per-kernel submit. No spec
  text claims an un-qualified "names the stuck kernel" for the batched path.
- **Test gap.** The current gpu test drives a wavefront NAME through the
  synchronous harness, which implies wavefront coverage it lacks. A new
  gpu-marked scenario drives the ACTUAL wavefront per-kernel-submit path and
  asserts the reported kernel is the in-flight one.
- **Teardown leak.** `MetalContext.destroy()` releases the lazily-created
  `_beacon_buffer`, so trace teardown is complete per `metal-dispatch-hygiene`.
- **Seqlock comment.** The seqlock write carries a one-line code comment that the
  guard rejects gross tearing only, not a payload straddle, and that correctness
  rests on writer quiescence at read time. Behavior is unchanged.

## Scope

- The trace-gated per-kernel submit inside `MetalFrameEncoder.dispatch` and
  `MetalFrameEncoder.dispatch_indirect`.
- The per-path accuracy statement in the `metal-kernel-beacon` spec.
- The `MetalContext.destroy()` release of `_beacon_buffer`.
- The seqlock code comment.
- A new gpu-marked wavefront attribution scenario.

## Non-goals

- **Design A — the shared-memory GPU beacon.** Backing `gKernelBeacon` with POSIX
  shared memory that the parent also maps, so the GPU per-kernel write survives
  the wedge and the parent reads GPU-written bytes while wavefront batching stays
  on. This is a non-goal. See the design for why it is discarded.
- **Any change to production (gate-off) behavior.** The batched single-submit
  encoder is unchanged. The MSL and the SPIR-V stay byte-identical. Trace-off
  performance is unchanged.
- **The native `MTLFunctionLog` path.** slang-rhi hides the `MTLCommandBuffer`,
  so this path stays unreachable, as in the parent change.

## Capabilities

### Modified Capabilities

- `metal-kernel-beacon`: the report accuracy is stated per dispatch path. The
  traced wavefront encoder submits one kernel per command buffer, so the mmap
  cell always names the in-flight kernel. The production batched path is
  unchanged and byte-identical. `MetalContext.destroy()` releases the beacon
  buffer. The seqlock write documents its quiescence assumption.
- `metal-dispatch-hygiene`: under `SKINNY_METAL_TRACE` the wavefront dispatch
  submits one kernel per bounded command buffer, so a wedge isolates the one
  kernel that hung.

## Impact

- **Host**: `MetalFrameEncoder.dispatch` and `MetalFrameEncoder.dispatch_indirect`
  gain a trace-gated per-kernel submit branch; `MetalContext.destroy()` releases
  `_beacon_buffer`; the seqlock write gains a comment. No signature changes.
- **Shaders**: none. The MSL and the SPIR-V are byte-unchanged.
- **Tests**: a new gpu-marked wavefront attribution scenario; a hostless
  assertion that `destroy()` clears `_beacon_buffer`.
- **No new Python or C++ dependencies.**
