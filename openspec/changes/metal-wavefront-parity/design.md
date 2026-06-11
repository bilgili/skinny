## Context

The archived `metal-megakernel-parity` change brought `main_pass.slang`
(`mainImage`) to structural + perceptual parity on a `MetalContext` and flipped
`auto`→Metal on Apple Silicon. It proved that an in-process Slang→Metal compile
dispatches bit-identically to Vulkan for a **single** kernel, *provided* uniforms
go through `set_data` byte blobs (the D4 fence-hang discipline) and `float3` is
padded to 16 B in MSL (the `_pack_uniforms_msl` lesson).

Everything on the **wavefront axis** is still Vulkan-only:

- `vk_wavefront.py` is "Vulkan only" by its own docstring. Its real per-frame
  work is `WavefrontPathPass.record_dispatch(cmd, scene_set)` — a tiled,
  counting-sorted bounce loop recorded into a `VkCommandBuffer`: per tile
  `generate → for each bounce { intersect → build_args → scatter → per-material
  shade dispatched via vkCmdDispatchIndirect → } → resolve`. Buffer sizes come
  from `wavefront_layout.queue_buffer_sizes` (the GPU-free source of truth).
- `WavefrontBdptPass` adds the subpath-walk + connection stages (multiple walk
  modes), also indirect-dispatched and compacted.
- ReSTIR DI (`restir/*`) attaches as a reuse plugin scheduled at bounce 0 through
  the scene-sampling seam; neural (`neural_proposal_pass`) attaches as a
  per-bounce pre-pass. Both are wavefront-only and fall back to their analytic
  subset on the megakernel.

The Metal compute surface (`metal_compute.py`) currently exposes single-shot
`dispatch` (megakernel) and `dispatch_kernel` (generic), the latter ending each
call with `dev.wait_for_idle()`. There is **no** indirect dispatch, **no**
multi-pass command encoding with inter-stage barriers, and capability flags
(`supports_fp16_storage/compute`, external memory) are hard-`false`.

## Goals / Non-Goals

**Goals:**

- The staged wavefront path and BDPT integrators render on a `MetalContext`,
  sharing one backend-neutral bounce-loop driver with the Vulkan path, and reach
  A/B image equivalence with the Vulkan wavefront output (structural-exact +
  perceptual tolerance on converged color, per the megakernel precedent).
- ReSTIR DI reuse and the neural directional proposal build and run on the Metal
  wavefront path through the unchanged scene-sampling seam.
- `metal_compute` gains indirect compute dispatch and a multi-pass,
  barrier-separated frame encoding good enough that a 6-bounce loop is not
  serialized by a per-stage `wait_for_idle`.
- `supports_fp16_storage/compute` report the real Metal device; neural weights
  load as Metal GPU state via a buffer-upload handoff, with an fp32 fallback when
  fp16 is unavailable.
- `auto`→Metal keeps the full feature set; `--backend vulkan` stays
  byte-identical.

**Non-Goals:**

- No change to wavefront/ReSTIR/neural **algorithms** or `.slang` math — this is
  a backend port, not a shader redesign.
- No new integrator, reuse regime, or training mode (the in-flight
  `wavefront-adaptive-shade-kernels` and neural-training work are separate).
- No GPU↔GPU external-memory/semaphore interop on Metal (the file/buffer handoff
  is sufficient for frozen weights); `supports_external_*` stay `false`.
- No removal of `vk_wavefront.py`'s Vulkan behavior — Vulkan output must not
  shift by a single bit.

## Decisions

### D1 — Extract a backend-neutral wavefront driver instead of forking a `metal_wavefront.py`

The bounce-loop *orchestration* (tile sizing, bounce count, stage order, buffer
lifetimes, ReSTIR/neural scheduling) is identical across backends; only the
*dispatch + sync primitives* differ. Model the loop as an ordered list of **stage
descriptors** (`entry`, binding spec, dispatch-size source: fixed | per-tile |
indirect-args-buffer) executed by a thin backend adapter:

- Vulkan adapter records each stage into the frame `VkCommandBuffer` (today's
  `record_dispatch`), preserving byte-identical behavior.
- Metal adapter encodes each stage into one slang-rhi command encoder per frame
  with a global compute barrier between stages.

This keeps a single source of truth for the loop and confines backend specifics
to ~two adapters. *Alternative rejected:* a parallel `metal_wavefront.py` mirror
— doubles the loop logic and invites drift (the very thing the megakernel port
avoided by giving `metal_compute` API parity with `vk_compute`).

### D2 — Native indirect dispatch via slang-rhi, CPU-readback fallback behind a capability probe

The per-material shade and tiled loop dispatch sizes are GPU-written
(`build_args` → `VkDispatchIndirectCommand` triples). Add
`ComputePipeline.dispatch_indirect(args_buffer, offset, *, bindings)` to
`metal_compute` using the slang-rhi command encoder's indirect-dispatch entry if
exposed by SlangPy. If the installed SlangPy does not surface indirect dispatch,
fall back to reading the slot-count buffer back to the host and issuing a direct
dispatch — correct but with a per-bounce round-trip. A one-time capability probe
(`ctx.supports_indirect_dispatch`) selects the path and is logged. *Alternative
rejected:* always CPU-readback — simpler but serializes every bounce on a
host↔GPU sync; unacceptable for equal-time comparison.

### D3 — One frame command encoder with inter-stage barriers, not per-stage `wait_for_idle`

`dispatch_kernel`'s trailing `wait_for_idle` is fine for one shot but would
serialize a multi-stage, 6-bounce loop on the GPU idle each stage. The Metal
adapter opens a single command encoder for the whole loop, inserts a global
compute-memory barrier between stages (matching the Vulkan
`COMPUTE→COMPUTE (+ indirect read)` pipeline barriers), and submits once. Uniform
and small constant blocks are still bound via `set_data` byte blobs only (D4
fence-hang discipline carries over). *Alternative rejected:* keep per-stage
`wait_for_idle` for v1 — would make the equal-time ReSTIR/neural parity tests
meaningless and risk masking real ordering bugs.

### D4 — Scalar-layout parity for the wavefront record structs

The Vulkan kernels compile with `-fvk-use-scalar-layout`; `wf_records.slang` /
`wavefront_state.slang` byte layouts must match what the Metal in-process
Slang→Metal compile produces. Slang's Metal target pads `float3` to 16 B (the
megakernel lesson), so record structs that round-trip through queue buffers must
either avoid bare `float3` or be verified field-by-field. Extend the existing
GPU-free `wavefront_layout` tests to assert the reflected **MSL** offsets equal
the scalar SPIR-V offsets, so a layout drift fails CI before any GPU run.
*Alternative rejected:* trust the compilers agree — the float3 padding gap is
exactly the kind of silent corruption that produced garbage in the megakernel
port.

### D5 — Persistent reservoir + weight buffers as Metal `StorageBuffer`s through the unchanged seam

ReSTIR reservoirs and neural weights are frame-persistent GPU state. Allocate
them once as Metal `StorageBuffer`s, survive across accumulation frames, and
reset only on the reuse/neural config-hash change (today's behavior). The
scene-sampling seam already abstracts "build proposal + reuse passes"; the Metal
wavefront driver calls the same builder, so ReSTIR/neural need no seam change —
only the pass objects they construct must be backend-neutral.

### D6 — Neural weight handoff by buffer upload; fp16 gated on a device probe

Frozen offline weights load by uploading the weight blob into a Metal
`StorageBuffer` via `set_data` (no external-memory interop needed). Probe the
Metal device for fp16; set `supports_fp16_storage/compute` accordingly. When
present, store/infer weights in fp16 (e4m3/half per the neural-precision study);
otherwise `_effective_neural_config()` falls back to fp32 exactly as on a
Vulkan device without fp16. Per-sample `neuralNetworkVersion` stamping is
unchanged.

## Risks / Trade-offs

- **slang-rhi may not expose indirect dispatch through SlangPy** → D2's CPU
  readback fallback keeps it correct; a probe + log makes the slow path visible,
  and the indirect path lands the moment the binding is available.
- **`float3` / scalar-layout mismatch corrupts queue records** → D4 extends the
  GPU-free layout test to MSL offsets so drift fails before a GPU run; mirror the
  megakernel `_pack_uniforms_msl` relocation table for any packed record.
- **Per-stage barrier handling wrong → fence hang or read-before-write** → reuse
  the proven Vulkan barrier points (the stage list already encodes them) and the
  D4 `set_data`-only discipline; bring up path-only first (fewest stages) before
  BDPT.
- **fp16 numerics differ Metal vs Vulkan** → require fp32 A/B parity first, then
  allow fp16 within the perceptual tolerance; the study's fp32 fallback is the
  safety net.
- **Metal argument-buffer / binding limits** → the wavefront passes reuse the
  megakernel descriptor-binding map that already fits Metal; new queue buffers
  are plain storage buffers, not bindless.
- **Perf regression vs megakernel on Metal** → equal-time A/B is a gate, not just
  image parity; if wavefront is slower than megakernel on Metal for a scene, that
  is reported, not hidden (no silent fallback).

## Migration Plan

Phased, each phase independently verifiable and behind the existing backend
gate so partial progress never regresses Vulkan or the Metal megakernel:

1. **Capability + primitives** — fp16 probe + flags in `metal_context.py`;
   `dispatch_indirect` + single-encoder multi-pass loop + indirect probe in
   `metal_compute.py`; extend `wavefront_layout` tests to MSL offsets.
2. **Backend-neutral driver** — extract the stage-descriptor loop from
   `vk_wavefront.py`; Vulkan adapter wraps today's `record_dispatch`
   byte-for-byte; prove Vulkan output unchanged.
3. **Wavefront path on Metal** — Metal adapter runs generate→…→resolve; headless
   A/B path parity Metal vs Vulkan.
4. **Wavefront BDPT on Metal** — subpath walk + compacted connection stages;
   A/B parity across every walk mode.
5. **ReSTIR DI on Metal** — persistent reservoirs through the seam; unbiased
   converges to stock-NEE reference; A/B vs Vulkan ReSTIR.
6. **Neural proposal on Metal** — weight upload + fp16/fp32; pdf parity vs
   reference; default-selection byte-identical; A/B vs Vulkan neural.
7. **Docs** — `docs/Wavefront.md`, `docs/ReSTIR.md`, `docs/NeuralGuiding.md`,
   `docs/Architecture.md`, `README.md`/`CLAUDE.md` backend tables.

**Rollback:** `--backend vulkan` is untouched at every phase. If a Metal
wavefront phase regresses, gate Metal wavefront behind a capability check that
pins Metal to the megakernel (today's behavior) — no user-visible change beyond
the lost feature.

## Open Questions

- Does the vendored SlangPy/slang-rhi expose indirect compute dispatch and an
  explicit multi-pass command encoder with compute barriers, or must D2/D3 use
  the CPU-readback + per-stage path for v1? (Resolve in phase 1 by probing the
  installed binding.)
- Exact perceptual tolerance for the neural fp16 A/B — reuse the megakernel
  color tolerance, or tighten given the proposal's narrower output?
- Should `vk_wavefront.py` keep its name as the Vulkan adapter, or move the
  Vulkan adapter under the new driver module? (Naming only; no behavior impact.)
