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

Carry the snapshot stack in the per-lane path state (VRAM), not registers:

```
struct RecVertex { float3 pos, normal, wo, wiLocal; float3 L_k, beta_in_k; uint depth; }
// up to REC_MAX_BOUNCES (6) per lane, in the path-state buffer
```

- **On a guideable bounce** (flat/graph material, reflective, `wiLocal.y > 1e-4`,
  `pdf > 0`): push `RecVertex` with the pre-throughput-update `beta_in_k` and the
  running `L_k` — the same guard and snapshot order as the megakernel.
- **At lane termination** (miss / absorb / max depth / light hit): `L_final` is the
  lane's accumulated radiance; iterate the lane's stack and `emitRecord` each
  `contrib_k`, dropping non-finite samples — identical to the megakernel's tail
  loop. Emission uses the existing bounds-safe `emitRecord` (atomic counter < cap).

This reuses the exact arithmetic in `path_record.slang`; only the *storage* of the
snapshots moves from registers to the per-lane state buffer.

## Record-mode gate

Emission is gated by a frame-constants flag set only while online training is
active (`Renderer._online_training`). When off, the wavefront kernels take the
shipped path with no stack writes and no emit — the default render stays
byte-identical and pays no record overhead.

## Renderer rewire

`drain_path_records_to_replay` stops dispatching `mainImageRecord`. Instead, with
record-mode enabled the normal wavefront render already filled bindings 36/37, so
the drain just reads the counter + buffer and `records_from_buffer` → `replay.add`.
The megakernel `mainImageRecord` entry and `dump_path_records` are kept for the
**offline** `.nrec` dump (not on the per-frame path, so its compile/TDR cost is
irrelevant there; it can also run on a non-TDR box).

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
