# Metal Record Drain

## Why

Online neural training needs two GPU↔trainer flows: trained weights into the
renderer, and live path records out of it. The weights side now works on the
native Metal backend (change `metal-neural-interop`), but the records side is
still Vulkan-only: `wf_records.slang` stubs every emitter to a no-op under
`SKINNY_METAL`, the record bindings 36/37 are compiled out of the neural-active
Metal build to fit the 31-buffer-slot argument-table cap, and the renderer's
live drain is gated on Vulkan descriptor sets. The result is that a
fully-on-Metal online training loop has no training data — the one remaining
gap to feature parity for online neural guiding on Apple Silicon.

## What Changes

- Un-stub wavefront record emission on Metal behind a new records-active build
  (mirroring the `SKINNY_METAL_NEURAL` pattern): when online training arms, the
  Metal wavefront path pass rebuilds with record emission compiled in —
  per-lane vertex stack, terminate-time backward attribution, and appends to
  bindings 36/37 — using the same `wf_records.slang` code Vulkan runs.
- Fit the neural+records Metal build under the 31-slot cap: keep the already
  compiled-out `toolBuffer` out, compile out further wavefront-dead globals if
  needed, and fail the build with a clear slot-count error rather than a
  silent Metal compile failure.
- Make the renderer's live drain backend-neutral: `_ensure_wf_record_drain` /
  `_drain_wavefront_records` get a Metal branch (bind-by-name buffers,
  counter reset by upload, readback via `download_sync`) instead of the
  `descriptor_sets is not None` Vulkan gate; `recordMode` reaches the Metal
  uniform pack.
- Default Metal wavefront render stays byte-identical when records are off
  (same gating contract as Vulkan: no stack writes, no appends, no record
  overhead).
- Tests: guarded GPU tests for record emission validity on Metal, record-stream
  equivalence vs Vulkan, default-render bit-identity with records off, and a
  fully-on-Metal online loop (drain → trainer → `interop` weight handoff →
  version bump).
- Docs: `docs/NeuralGuiding.md` (drop the "record drain remains Vulkan-only"
  caveat), `docs/Wavefront.md` Metal section, `docs/Architecture.md` slot-cap
  notes, `README.md`, `CHANGELOG.md`.

## Capabilities

### New Capabilities

(none — this extends existing capabilities)

### Modified Capabilities

- `wavefront-native-path-records`: new requirement — record emission and the
  live drain SHALL run on the native Metal backend (under the argument-table
  cap), with the same gating, attribution, and record-layout contracts as
  Vulkan.
- `metal-backend`: the wavefront-parity requirement extends to the online
  training loop — record emission, live drain, and the existing interop weight
  handoff SHALL together support fully-on-Metal online neural training.

## Impact

- Shaders: `wavefront/wf_records.slang` (stub gate), `integrators/
  path_record_common.slang` + `bindings.slang` (slot-cap compile-out
  conditions). No new bindings; 36/37 reuse.
- `src/skinny/metal_wavefront.py` — records-active define + pass rebuild,
  slot-budget probe/error.
- `src/skinny/renderer.py` — backend-neutral drain (`_ensure_wf_record_drain`,
  `_drain_wavefront_records`, `_wf_record_active` gating), `recordMode` in the
  MSL uniform pack, pass-rebuild trigger on online-training enable.
- `tests/` — new guarded Metal record tests + fully-on-Metal online-loop test.
- Docs: `docs/NeuralGuiding.md`, `docs/Wavefront.md`, `docs/Architecture.md`,
  `README.md`, `CHANGELOG.md`.
- Out of scope: BDPT record emission (out of scope on Vulkan too), megakernel
  record source on Metal, zero-copy drain via shared storage (records could
  later reuse `StorageBuffer(shared=True)` from `metal-neural-interop` to skip
  the readback staging copy — follow-up optimization).
