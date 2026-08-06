# Tasks — beacon-wavefront-attribution

Each task traces to a frozen requirement. The interfaces in `design.md` are
FROZEN: `MetalFrameEncoder.dispatch` / `.dispatch_indirect` signatures, the
`MetalContext.destroy()` shape, and the `BeaconWriter.stamp` behavior stay
UNCHANGED. Plan touches host submit granularity and documentation only. No
shader changes. Groups 1–4 are the implementer's. Group 5 is the verifier's.

## 1. Trace-gated per-kernel submit in `MetalFrameEncoder`

Satisfies: metal-kernel-beacon "The traced wavefront encoder submits one kernel
per command buffer"; metal-dispatch-hygiene "Traced wavefront dispatch submits
one kernel per command buffer".

- [x] In `MetalFrameEncoder.dispatch` (`src/skinny/metal_compute.py:1372`), after
  `cpass.end()`, add one branch on `self.ctx.trace`. When `trace` is False,
  return without submitting (unchanged batched path). When `trace` is True, run
  the existing `flush()` body: submit the encoder, `wait_for_idle()`, reopen a
  fresh encoder.
- [x] Reuse the `flush()` mechanism, not `submit()`. The trace branch SHALL NOT
  set `self._submitted`, so a later dispatch in the same frame still encodes and
  the frame-end `submit()` stays valid.
- [x] In `MetalFrameEncoder.dispatch_indirect`
  (`src/skinny/metal_compute.py:1386`), apply the identical trace-gated rule
  after its `cpass.end()`. Trace off encodes only; trace on submits + drains +
  reopens.
- [x] Keep the stamp order unchanged: `self.ctx.beacon_stamp(_pipe_entry(pipe))`
  fires before the compute pass, on both paths.
- [x] Update the batched-loop code comment inside `dispatch` so it states the
  trace-off vs trace-on submit granularity, not only the batched case.
- [x] Leave `barrier()`, `flush()`, and `submit()` bodies UNCHANGED. Confirm no
  recorder or driver call site in `metal_wavefront.py` or `wavefront_driver.py`
  changes.

## 2. `MetalContext.destroy()` releases `_beacon_buffer`

Satisfies: metal-kernel-beacon "Context teardown releases the beacon buffer".

- [x] In `MetalContext.destroy()` (`src/skinny/metal_context.py:547`), release
  `_beacon_buffer` inside the existing `_destroyed` idempotent guard, beside the
  `_beacon_writer` release.
- [x] Read the handle with `getattr(self, "_beacon_buffer", None)`. When it is
  not None, call its `destroy()` (or drop the reference), wrap in a best-effort
  `try/except`, then set `self._beacon_buffer = None`.
- [x] Confirm the release is safe when `_beacon_buffer` is None (trace off, or
  trace on with no dispatch) and on a second `destroy()` call.

## 3. `BeaconWriter.stamp` quiescence comment

Satisfies: metal-kernel-beacon "The seqlock write documents its quiescence
assumption".

- [x] In `BeaconWriter.stamp` (`src/skinny/metal_beacon.py:174`), extend the
  seqlock comment. State that the sequence-mirror guard catches gross tearing
  only, not a payload straddle, and that correctness rests on writer quiescence
  at read time (the parent reads only after it SIGTERMs the wedged child).
- [x] Change no code in `stamp`. Behavior stays UNCHANGED.

## 4. Per-path accuracy documentation

Satisfies: metal-kernel-beacon "Beacon accuracy is stated per dispatch path".

- [x] Find the user-facing beacon documentation. Search the docs tree and
  `README.md` for beacon accuracy prose (`Grep` for "beacon" and "stuck
  kernel").
- [x] Correct any un-qualified "names the stuck kernel" claim for the wavefront
  path. State that the megakernel and synchronous single-shot wrappers name the
  exact in-flight kernel, and the wavefront path names it only under
  `SKINNY_METAL_TRACE`, through per-kernel submit.
- [x] State that the batched wavefront path (trace off) names the last-encoded
  kernel of the in-flight batch, which need not be the kernel that hung.
- [x] Route the doc edit to the owner document per `CLAUDE.md` (a Metal/backend
  or beacon-owning doc). Register a new document in `README.md` § Documentation
  only if one is created; do not create a second index.

## 5. Verification (verifier-owned)

Satisfies the scenarios in both spec deltas. The verifier derives these from the
spec, not from the implementation, and writes tests only.

- [x] Add a gpu-marked wavefront attribution scenario that drives the ACTUAL
  per-kernel-submit path with `SKINNY_METAL_TRACE` on. One wavefront stage kernel
  hangs; assert the parent reports that in-flight stage kernel by name, not a
  later-encoded stage of the same frame. This closes the shipped sync-harness
  gap (metal-kernel-beacon scenario "a wedged wavefront kernel is named under
  trace"). The child helper (`tests/metal_cleanup_child.py`) likely needs a new
  wavefront-hang mode — that support code is the verifier's to add.
- [x] Add a hostless assertion that `MetalContext.destroy()` clears
  `_beacon_buffer` and that a second `destroy()` is a safe no-op (scenario
  "destroy clears the beacon buffer"). Prefer a stub context over a device.
- [x] Add a hostless assertion that the `BeaconWriter.stamp` seqlock comment
  records the straddle limit and the quiescence assumption (scenario "the seqlock
  comment records the straddle limit").
- [x] Re-confirm gate-off byte-identity. No shader changed, so re-assert the
  existing megakernel gate-off SPIR-V golden
  (`test_gate_off_megakernel_spirv_carries_no_beacon`) still passes; do not add a
  new compile.
- [x] Register every new test file with `git add -f` (worktree `.gitignore` is
  `*`). Run hostless tests with `PYTHONPATH=src`; run the gpu-marked scenario
  under the guarded Metal runner, one process at a time.

## Sequencing risk

- **Reopen, do not close, the frame under trace.** The trace branch must reuse
  the `flush()` body (submit + drain + reopen a fresh encoder) and must NOT set
  `_submitted`. If it set `_submitted`, the frame-end `submit()` would skip and a
  same-frame later dispatch could target a stale encoder. Pin this in task 1.
- **Barrier / flush / submit on an empty encoder.** Under trace the recorder's
  `barrier()` runs on the fresh empty encoder (the prior dispatch already
  submitted). `flush()` and `submit()` then act on an empty or possibly-empty
  encoder. Each is safe by design, but the gpu test in group 5 must exercise a
  multi-stage bounce loop so a barrier lands between two per-kernel submits and
  the frame-end submit hits an empty encoder — otherwise the empty-encoder path
  is untested.
- **Indirect fallback under trace.** `dispatch_indirect`'s CPU-readback fallback
  already calls `flush()` mid-frame. Under trace the per-kernel submit precedes
  that `flush()`, so `flush()` is a no-op on the empty encoder. Confirm the
  recorder still reads the count correctly; the gpu scenario should cover an
  indirect-dispatching wavefront stage if the runner allows it.
