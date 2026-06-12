# Design — Metal Record Drain

## Context

The wavefront record system (change `wavefront-native-path-records`) has three
parts: shader emission (`wavefront/wf_records.slang`: per-lane vertex stack +
`wfResetRecords`/`wfPushRecord`/`wfEmitRecords`, appending 64-byte `PathRecord`s
to bindings 36/37 with terminate-time backward attribution), a gating uniform
(`fc.recordMode`, packed from `_wf_record_active`), and the renderer's live
drain (`_ensure_wf_record_drain` allocates the drain target and seeds the
counter; `_drain_wavefront_records` reads counter + records into the
`ReplayBuffer`).

On Metal all three are blocked:

- `wf_records.slang:38-51` stubs the emitters to no-ops under `SKINNY_METAL`
  unconditionally.
- `path_record_common.slang:36` compiles `recordBuf`/`recordCounter` (36/37)
  out of the neural-active Metal build (`SKINNY_METAL && SKINNY_METAL_NEURAL`)
  — that gate is how the neural weight buffers 33/34/35 fit Metal's
  31-buffer-slot argument table (`bindings.slang:79` does the same to
  `toolBuffer`).
- `renderer.py:7605` gates the drain on `descriptor_sets is not None`
  (always `None` on Metal), so `_wf_record_active` never arms and `recordMode`
  stays 0.

Meanwhile the Metal path pass already allocates and binds (by name) the
per-lane stack buffers `wfRecStack`/`wfRecCount`
(`metal_wavefront.py:509-510,526-527`) — today dead-stripped. The Metal
argument-table count is per-kernel; the investigator's estimate for the
neural-active build is ~27 of 31 slots on the heaviest kernel, so +2 for
36/37 plausibly fits, but the number must be proven empirically per kernel.

The weights half of the loop already works on Metal
(`metal-neural-interop`: UMA interop publisher + frame-end swap on the Metal
render paths). This change supplies the data half.

## Goals / Non-Goals

**Goals:**

- Wavefront path-record emission on the native Metal backend: same
  `PathRecord` layout, same per-lane stack + terminate-time attribution, same
  gating (off by default, on only while online training is active).
- Backend-neutral live drain: per-frame counter read + record readback into
  the `ReplayBuffer` on Metal, no megakernel dispatch.
- Fully-on-Metal online loop: drain → trainer → `--neural-handoff interop` →
  frame-end swap, end to end on one Metal device.
- Default Metal wavefront render byte-identical with records off.
- Clear slot-count failure if the neural+records build exceeds 31 slots on
  any kernel — never a cryptic Metal compile error.

**Non-Goals:**

- BDPT record emission (out of scope on Vulkan too — drain falls back to the
  megakernel source there, which stays unsupported on Metal).
- The megakernel record source on Metal (the ~400 s compile / watchdog seam is
  exactly what wavefront-native emission avoids).
- Zero-copy drain via `StorageBuffer(shared=True)` — a follow-up optimization;
  this change uses the existing `download_sync` readback (read_back staging).
- fp16 record storage or record-layout changes.

## Decisions

**D1 — Records are a build flavor: `SKINNY_METAL_RECORDS`, mirroring
`SKINNY_METAL_NEURAL`.**
The Metal wavefront pass gains a `records_active` constructor flag; when set,
the Slang session defines `SKINNY_METAL_RECORDS=1`, which (a) un-stubs the
`wf_records.slang` emitters on Metal and (b) keeps `recordBuf`/`recordCounter`
declared in the neural build. Enabling online training rebuilds the pass with
the flag (same mechanism as selecting the neural proposal; accumulation reset
via the existing `_last_state_hash = None` path). Disabling rebuilds without
it, restoring the bit-identical default render.
*Alternative:* always compile records in on Metal. Rejected: pays 2 argument
slots + per-lane stack register pressure on every wavefront render, and breaks
the "default render unchanged" requirement headroom for future bindings.

**D2 — Slot budget: keep the existing compile-outs, add an explicit reflected
slot check.**
Under `SKINNY_METAL && SKINNY_METAL_NEURAL && SKINNY_METAL_RECORDS`:
`toolBuffer` stays compiled out (already dead in every wavefront kernel),
36/37 come back, 33/34/35 stay. Estimated heaviest-kernel usage ≈29 of 31. If
a kernel still exceeds the cap, compile out further wavefront-dead globals
(`gizmoSegments` and `lightSplatBuffer` are dead in the path pass — candidates
behind the same define) rather than touching live bindings. After session
build, reflect each entry point's buffer count and raise a `RuntimeError`
naming the kernel and its count when >31 — the empirical-probe discipline the
backend already uses for indirect dispatch.
*Alternative:* Metal argument buffers (tier-2) to escape the cap. Rejected:
slang-rhi 0.42 does not expose them; the cap workaround is established
project practice.

**D3 — Drain goes backend-neutral, not Metal-forked.**
Replace the `descriptor_sets is not None` gate with a backend split inside the
same methods: on Vulkan, the existing descriptor-write path; on Metal,
allocate the drain target through `self._gpu.StorageBuffer`, expose it to
`_build_metal_binds()` under the existing `recordBuf`/`recordCounter` names
(already in the bind dict), reset the counter with `upload_sync([0, capacity])`
and read it back with `download_sync` (the Metal `StorageBuffer` readback path,
including the shared-mode staging bounce from `metal-neural-interop`, already
works). `records_from_buffer` and the `ReplayBuffer` are host-side and shared.
*Alternative:* a separate `_drain_wavefront_records_metal`. Rejected: the
logic differs only in buffer plumbing; two copies would drift.

**D4 — `recordMode` flows through the MSL uniform pack.**
`_wf_record_active` arms on Metal once the drain is backend-neutral; verify
`_pack_uniforms_msl` carries `fc.recordMode` at the MSL-correct offset (the
Vulkan pack already does; the Metal pack must not silently drop it — add the
field if absent and cover it in the uniform-offset test).

**D5 — Verification mirrors the Vulkan record gates.**
(a) Structural validity on Metal: records drain with count > 0, finite
contributions, stack bounded, layout parses via `records_from_buffer`.
(b) Cross-backend equivalence: same scene/config recorded on Metal and Vulkan
→ equivalent record sets (same vertices and contributions, independent of
emission order — the same criterion the spec uses for wavefront-vs-megakernel).
(c) Records-off bit-identity: Metal wavefront render with the records build
disabled is byte-identical to before the change.
(d) End-to-end: fully-on-Metal online loop — drain feeds the numpy trainer,
weights publish through the Metal interop handoff, `networkVersion` advances,
and the converged mixture estimate stays unbiased vs `{bsdf}`.

## Risks / Trade-offs

- [A kernel exceeds 31 slots with neural+records] → D2's reflected slot check
  fails loudly with the kernel name; mitigation ladder: compile out
  `gizmoSegments`/`lightSplatBuffer` (wavefront-dead) behind the records
  define before touching anything live.
- [Per-lane stack (REC_VERTEX_STRIDE 76 B × REC_MAX_BOUNCES) inflates Metal
  threadgroup/register pressure and slows the path pass] → stack lives in
  device memory buffers (`wfRecStack`/`wfRecCount`, already allocated); cost
  is bandwidth on training frames only — records are off outside training
  (D1). Measure path-pass frame time records-on vs records-off and note it.
- [Drain readback stalls the render thread each frame] → same cost structure
  the Vulkan drain already accepts (counter read + record bytes); on UMA the
  copy is cheap. The shared-storage zero-copy drain is the named follow-up if
  it shows up in profiles.
- [Record-stream equivalence across backends is not bit-exact] → the spec
  criterion is set-equivalence (vertices + contributions), not byte order;
  RNG streams match per lane/bounce on both backends (same Slang code, same
  seeds) — the Metal↔Vulkan wavefront A/B precedent applies.
- [Thermal/compile discipline] — every records build is a Metal wavefront
  pipeline compile: one guarded process at a time, GPU tests marked `gpu`,
  sweeps run `-m 'not gpu'`.

## Open Questions

- Exact per-kernel slot counts for the neural+records build (resolved by D2's
  reflection check during task 1; determines whether the dead-global
  compile-out ladder is needed at all).
- Does `_pack_uniforms_msl` already include `recordMode`? (Check during D4;
  the MSL uniform-offset test pins the answer either way.)
