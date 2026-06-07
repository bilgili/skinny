## Why

Online neural training (`neural-online-training`) feeds its replay buffer from
per-vertex path records. Those records are produced by `mainImageRecord`, a
**second megakernel entry** (`integrators/path_record.slang`) the renderer
dispatches each frame and drains from bindings 36/37
(`Renderer.drain_path_records_to_replay`). The megakernel was chosen because one
thread owns the whole path, so the tail radiance `Li` is known at loop end and
attributed back from a local register stack (the shader's own rationale).

That choice is **incompatible with online training on NVIDIA/Windows**, found
while bringing Stage 2 up on an RTX 4090 (sm_89):

- the 8 MB megakernel takes **~200–400 s** for the driver to compile a pipeline, and
- a single dispatch runs longer than the **2 s Windows TDR** GPU watchdog (or
  exceeds a driver/hardware limit), so the device is **lost** (`VkErrorDeviceLost`)
  on the first dispatch.

The wavefront path does **not** hit this — it splits the work into small per-stage
kernels — and neural inference is already wavefront-only. So the renderer the user
actually runs for neural guiding is wavefront, but the record drain it depends on
forces a megakernel that cannot run on the target box. The live GPU drain is
therefore currently a seam exercised only where there is no such watchdog (the
`neural-online-training` task 7.3 NVIDIA-box note), and the Mac/wavefront suite
validates the reader contract off-GPU instead of the real drain.

This change removes the megakernel dependency by emitting the **same**
`PathRecord` stream from the **wavefront** integrator, so online training drains
records on the same backend it renders with — no second megakernel, no 2 s-TDR
dispatch, no 400 s compile.

## What Changes

- **Wavefront record emission.** The wavefront path pass emits one `PathRecord`
  per guideable (flat/graph, reflective, upper-hemisphere) bounce into the
  existing append buffer (binding 36) + counter (binding 37), byte-for-byte the
  shipped `PathRecord` layout (`path_records.py`, 64 B), so `records_from_buffer`
  and the offline `.nrec` tooling consume it unchanged.
- **Terminate-time backward attribution in wavefront.** Because the wavefront path
  is smeared across per-bounce dispatches, the tail radiance `Li` is not known at
  the bounce that sampled `wi`. Carry a small per-lane vertex stack
  (`pos, normal, wo, wiLocal, L_k, beta_in_k, depth`, ≤ `REC_MAX_BOUNCES`) in the
  path-state buffer and splat `contrib_k = max((L_final − L_k)/beta_in_k, 0)` when
  the lane terminates — the identical arithmetic `estimateRadianceRecord` already
  does per-thread in the megakernel.
- **Record-mode gate.** A frame-constants/preset flag enables record emission only
  while online training is active, so the default wavefront render stays
  byte-identical when not training (no new work, no register pressure on the hot
  path).
- **Renderer drain rewire.** `Renderer.drain_path_records_to_replay` reads the
  wavefront-produced records (no separate `mainImageRecord` dispatch); the
  megakernel record entry remains only for the offline `.nrec` dump
  (`dump_path_records`), which is not on the per-frame online path.

- **Not in scope:** the megakernel `mainImageRecord` entry itself (kept for the
  offline dump); the online loop / handoff / swap (shipped in
  `neural-online-training`); CUDA training internals or interop; raising the
  Windows TDR (a system setting, not a code fix); a wavefront record path for BDPT.

## Capabilities

### New Capabilities
- `wavefront-native-path-records`: the wavefront integrator emits the shipped
  `PathRecord` training stream directly (with a per-lane vertex stack and
  terminate-time backward attribution), gated to online-training frames, so the
  live record drain needs no megakernel dispatch and runs unbiased on the same
  backend used for neural inference — removing the 2 s-TDR / ~400 s-compile
  megakernel dependency that blocks online training on NVIDIA/Windows.

### Modified Capabilities
- `neural-online-training`: the live record drain SHALL source records from the
  wavefront path rather than the `mainImageRecord` megakernel; the recorded stream
  and the unbiasedness contract are unchanged.

## Impact

- **Code:** `shaders/wavefront/*` (per-lane record stack in path state + emit on
  guideable bounce + terminate-time splat), `shaders/integrators/path_record.slang`
  (share the attribution math), `vk_wavefront.py` (path-state stride for the stack,
  record-mode push/flag), `renderer.py` (`drain_path_records_to_replay` reads the
  wavefront records; record-mode gate threads through online training).
- **Backends:** wavefront only; megakernel `mainImageRecord` unchanged (offline
  dump). BDPT record emission out of scope.
- **Tests:** GPU drain validated end to end **on this NVIDIA/Windows box** (the
  motivating environment) — `{bsdf,neural}` online loop drains real wavefront
  records, trains, swaps, stays unbiased; parity of the wavefront record stream
  against the megakernel `.nrec` dump on a box where the megakernel still runs.
- **Docs:** `docs/Wavefront.md` (record-emission stage + per-lane stack),
  `docs/Architecture.md` (bindings 36/37 now wavefront-fed on the online path).
- **Risk:** per-lane vertex stack grows the wavefront path-state stride; gated off
  when not training so the default render is unaffected.
