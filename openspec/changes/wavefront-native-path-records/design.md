# Design — wavefront-native-path-records

## Context

`neural-online-training` shipped the online loop (replay → trainer → publisher →
frame-end swap) and a live record drain, `Renderer.drain_path_records_to_replay`,
that runs the `mainImageRecord` **megakernel** entry and reads bindings 36/37. On
NVIDIA/Windows (RTX 4090, driver 32.0.15.x) that megakernel:

- compiles a driver pipeline in **~200–400 s** (8+ MB SPIR-V), and
- **device-losts** on the first dispatch — the single megakernel dispatch exceeds
  the default 2 s Windows TDR watchdog (`TdrDelay` unset), so the OS resets the GPU.

Wavefront rendering is unaffected (small per-stage kernels) and is the only backend
that runs neural inference. This change moves record emission onto the wavefront
path so online training never dispatches the megakernel.

## The attribution problem

The training weight for a guideable vertex `k` is

    contrib_k = max((L_final − L_k) / beta_in_k, 0)

where `L_k` is the radiance accumulated up to and including vertex `k`'s local
terms and `beta_in_k` is the throughput entering `k`. In the megakernel one thread
owns the whole path, so `L_final` is a local register at loop end and the
per-vertex `(L_k, beta_in_k)` snapshots live on the thread's stack
(`estimateRadianceRecord`). In wavefront the path is spread across per-bounce
dispatches, so neither `L_final` nor the snapshots are co-resident at the bounce
that sampled `wi`.

## Approach: per-lane vertex stack + terminate-time splat

Carry the snapshot stack in per-lane VRAM, not registers:

```
struct RecVertex { float3 pos, normal, wo, wiLocal; float3 L_k, beta_in; uint depth; }  // 76 B
// up to REC_MAX_BOUNCES (6) per lane
```

**Storage (revised from the original proposal).** The stack lives in SEPARATE
set-1 buffers — `wfRecStack[stream·REC_MAX_BOUNCES]` (binding 9) + a per-lane
`wfRecCount` (binding 10), owned by `WavefrontPathPass` — NOT inside
`WavefrontPathState`. That struct is copied by value in every wavefront kernel
(`s = wfState[i]; … wfState[i] = s`); inlining a 6-deep stack there would force a
dynamically-indexed local array into scratch and ~8× the path-state bandwidth in
*every* kernel even when not training — re-introducing the register/bandwidth
pressure the wavefront split exists to avoid. The buffers are full-size only
while recording and 1-element dummies otherwise (the `wfNeural`/36-37 pattern),
so the default render stays byte-identical and pays no extra VRAM or bandwidth.

- **On a guideable bounce** (flat/python material, reflective, `wiLocal.y > 1e-4`,
  `pdf > 0`) in `wfFinishShade`: push `RecVertex` with the pre-throughput-update
  `beta_in` and the running `L_k` — the same guard and snapshot order as the
  megakernel (`wfPushRecord`).
- **At lane termination** (miss / RR / no-valid-bsdf / sphere-light hit via
  `wfTerminate`, and max-depth survivors via `wfPathResolve`): `L_final` is the
  lane's accumulated radiance; iterate the lane's stack and `emitRecord` each
  `recordContrib(L_final, L_k, beta_in)`, dropping non-finite — identical to the
  megakernel's tail loop. Emission uses the bounds-safe `emitRecord` (counter < cap).

The attribution arithmetic is shared via `integrators/path_record_common.slang`
(`recordContrib`); only the *storage* of the snapshots moves from registers to
the per-lane VRAM buffers.

## Record-mode gate

Emission is gated by `FrameConstants.recordMode`, set only while the wavefront
record drain is active (`Renderer._wf_record_active`, resolved from the record
source at `enable_online_training`). When off, the wavefront kernels take the
shipped path with no stack writes and no emit — the default render stays
byte-identical and pays no record overhead (verified: recordMode off vs on
produces a bit-identical image, diff = 0).

## Renderer rewire (dual-source drain)

`drain_path_records_to_replay` is **source-selectable** rather than a wholesale
replacement, so megakernel training stays available. `_record_source` is
`auto` | `megakernel` | `wavefront`; `auto` resolves to `wavefront` for the
wavefront *path* integrator (and `megakernel` for bdpt — out of scope — or
megakernel execution).

- **wavefront source:** the normal wavefront render already filled bindings 36/37
  (recordMode on), so the drain just reads the counter + buffer
  (`records_from_buffer` → `replay.add`) and resets the counter for the next
  frame — **no** `mainImageRecord` dispatch, so no ~400 s-compile and no 2 s-TDR
  device-loss. Records come from the same paths that produced the displayed frame.
- **megakernel source:** the existing path — dispatch `mainImageRecord`, read
  back — unchanged, for non-TDR boxes that want it.

The megakernel `mainImageRecord` entry and `dump_path_records` are also kept for
the **offline** `.nrec` dump (not on the per-frame path).

## Why not just raise the TDR

`TdrDelay`/`TdrLevel` are machine-wide registry settings needing admin + reboot,
and a ~400 s-compile, multi-second-dispatch megakernel is the wrong cost for a
per-frame online feed even if it didn't device-lost. The wavefront emission is the
correct fix; the TDR change is at best a workaround for the offline dump.

## Verification

- GPU end to end on the motivating NVIDIA/Windows box: `{bsdf,neural}` online loop
  drains real wavefront records, trains, swaps, stays unbiased — the test that
  could not run against the megakernel drain.
- Stream parity: the wavefront record stream matches the megakernel `.nrec` dump
  (same records, modulo ordering) on a box where the megakernel still runs.
- Default-render invariance: with record-mode off, the wavefront image is
  unchanged vs the current backend.
