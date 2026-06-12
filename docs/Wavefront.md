# Skinny — Wavefront Execution Mode

The **wavefront** mode renders the *same* light-transport integral as the
[megakernel](Megakernel.md), but tears the per-pixel bounce loop apart across
**many small compute dispatches** connected by GPU-resident queues. Rays in
flight are kept as a stream of state records in VRAM; each bounce runs as a
sequence of stages (generate → intersect → build-args → scatter → shade →
resolve), and lanes are bucketed by material so each shade dispatch hits one
coherent material branch.

It is one of two execution modes (`EXECUTION_WAVEFRONT = 1`, `params.py:65`),
selected once at startup with `--execution-mode wavefront`
(`cli_common.py:85`) and **fixed for the session**. Both modes run on Vulkan
(MoltenVK on macOS) — "wavefront" is an execution strategy, not a separate
graphics API. The implementation lives in `vk_wavefront.py`,
`wavefront_layout.py`, and `shaders/wavefront/`.

For the high-level contrast with the megakernel see
[Megakernel vs wavefront](#megakernel-vs-wavefront) below.

---

## 1. Orchestration (`vk_wavefront.py`)

Three Python pass classes own their GPU resources and record their entire
staged dispatch loop into the frame command buffer (modelled on
`vk_skinning.SkinningPasses`). The render gate replaces the megakernel's single
dispatch:

- **Gate** (`renderer.py:7147-7158`, headless `7367-7377`): when
  `effective_execution_mode_index == EXECUTION_WAVEFRONT`, call
  `staged.record_dispatch(cmd, descriptor_sets[f])` instead of `vkCmdDispatch`.
- **Selection:** `integrator_index == 1` → `_ensure_wavefront_bdpt_pass()`; else
  `_ensure_wavefront_path_pass()`; if no scene yet → `WavefrontEnvPass`
  (env-only fallback). `WAVEFRONT_BDPT_SUPPORTED = True` (`renderer.py:915`), so
  wavefront BDPT is live and no longer falls back to the megakernel.
- **Lifecycle:** passes build lazily and cache on dims —
  `_ensure_wavefront_path_pass` keys on `(width, height, build_catchall)`
  (`renderer.py:1424-1462`), `_ensure_wavefront_bdpt_pass` on `(width, height)`
  (`1475-1507`). Both reuse the megakernel's set-0 layout via
  `self._scene_set0_layout` (`renderer.py:1351-1356`) — wavefront binds the
  **same scene descriptor set 0**, so the UBO, materials, lights, textures, and
  env CDFs are shared verbatim.

> Vulkan-only: gated on `hasattr(ctx, "compute_queue")` (`renderer.py:1090`,
> `1399`). There is no CPU fallback for the wavefront path.

### Path stages (`WavefrontPathPass`, `vk_wavefront.py:464-679`)

Constants: `_GROUP = 64` (`[numthreads(64,1,1)]`), `MAX_BOUNCES = 6` (lockstep
`WF_MAX_BOUNCES`, `wf_shade_common.slang:17`), `STREAM_CAP = 1<<20`,
`NUM_SLOTS = 2`, `HIT_STRIDE = 96`. Six/seven kernels are compiled from
`wavefront_path.slang` to per-entry `.spv` (`_wfpath_*`). The `record_dispatch`
loop (`vk_wavefront.py:603-667`), tiled by `stream_base` until `num_pixels`:

![Wavefront path stages: wfPathGenerate seeds the stream, then a CPU-driven bounce loop runs clear_counts → wfPathIntersect → wfBuildArgs → wfScatter → indirect shade(0)/shade(1), looping back per bounce, then wfPathResolve accumulates and tonemaps.](diagrams/wavefront_path_stages.svg)

Stage detail:

| Stage | Work |
|-------|------|
| `wfPathGenerate` | seed camera ray + path state (full stream) |
| `clear_counts()` | `vkCmdFillBuffer` zero `slot_count` + `slot_cursor` (+ barrier) |
| `wfPathIntersect` | trace, cutout-skip, miss/env terminate, classify by material slot, `InterlockedAdd(wfSlotCount[slot])` |
| `wfBuildArgs` | `vkCmdDispatch(1,1,1)`: prefix-sum counts → offsets, write `VkDispatchIndirectCommand` per slot |
| `wfScatter` | counting-sort alive lanes into per-slot queue slices |
| `shade(0, wfPathShadeFlat)` | `vkCmdDispatchIndirect(indirect, slot*12)` — flat + MaterialX |
| `shade(1, wfPathShade)` | `vkCmdDispatchIndirect(indirect, 1*12)` — skin/python/debug (only if `build_catchall`; push `shadeSlot` at offset 4) |
| `wfPathResolve` | running-mean accumulation + tonemap display (full stream) |

The bounce loop (`clear_counts` → … → `shade`) repeats `MAX_BOUNCES` times,
CPU-driven with barriers between stages.

The display tail is shared by every wavefront resolve kernel (path, BDPT, and
ReSTIR-via-path) through `wfWriteDisplay` (`wavefront/wf_display.slang`), the
sole writer of the output image (binding 1). Mirroring `main_pass.slang`'s
post-accumulation path, it composites the s=1 light splat, applies exposure +
tonemap + sRGB, then overlays the **HUD** (binding 3) and the **transform
gizmo** (binding 22). Because the gizmo segment list and `numGizmoSegments` are
rebuilt + uploaded every frame regardless of execution mode, the editor gizmo
draws in wavefront mode exactly as in megakernel. The focus-plane + furnace
over-energy overlays remain megakernel-only (they need the primary ray / scene
hit, which the wavefront does not keep per-pixel).

`mem_barrier()` (COMPUTE→COMPUTE | DRAW_INDIRECT) separates each stage. Push
constant `WfTilePC {streamBase, shadeSlot, streamSize}` (12 B,
`wf_shade_common.slang:31`). Pipeline layout = `[set0 scene, set1 path-state]`
+ 12 B push (`vk_wavefront.py:542-547`).

### BDPT stages (`WavefrontBdptPass`, `vk_wavefront.py:781-1068`)

Constants: `BDPT_MAX_VERTS = 7`, `VERTEX_STRIDE = 128`, `AUX_STRIDE = 128`,
`NUM_SLOTS = 2` (`SLOT_NEE=0`, `SLOT_FULL=1`), `STREAM_CAP = 1<<18`,
`EYE_BOUNCES = 5`, `LIGHT_BOUNCES = 6`. Three `walk_mode`s
(`vk_wavefront.py:810`, picked per session) control only subpath construction;
the connect+resolve tail is shared:

- **`fused`** (default): one `wfBdptWalk` (eye + light + s=1 splat).
- **`eye`**: staged eye walk (`wfBdptGenEye` + loop `wfBdptBounceEye`) +
  `wfBdptLightTail` (fused light + splat).
- **`eye_light`**: fully staged eye + light walks (`wfBdptGenLight`/
  `wfBdptBounceLight`) + standalone `wfBdptSplat`.

`record_dispatch` (`vk_wavefront.py:952-1056`) per tile: `build_subpaths()` →
`compact("wfBdptClassify")` → `indirect(SLOT_NEE, wfBdptConnectNee)` →
`indirect(SLOT_FULL, wfBdptConnectFull)` → `wfBdptResolve`. The staged
eye/light walks reuse the same counting-sort machinery
(`wfBdptWalkClassify` → buildargs → scatter → indirect `wfBdptBounceEye/Light`)
per bounce. `clear_counts()` here adds an extra COMPUTE→TRANSFER WAR barrier
(`vk_wavefront.py:981-986`) absent in the path pass, guarding prior
indirect-dispatch reads of `slot_count` against the fill.

---

## 2. Stream state & material bucketing

**Path-state record** `WavefrontPathState` (`wavefront_state.slang:15-26`,
Python mirror `wavefront_layout.py:28-46`): **AoS**, scalar layout, **68 B
stride** — `rayOrigin/rayDir/throughput/radiance` (4× float3) +
`pixelIndex/rngState/depth/flags` (uint) + `bsdfPdf` (float). `flags` bits:
`PATH_FLAG_ALIVE=1`, `PATH_FLAG_SPECULAR=2`. `tests/test_wavefront_state.py`
locks the Slang struct ↔ Python table.

> The record is intentionally AoS, **not** SoA (`wavefront_state.slang:8-9`):
> convert hot fields to SoA only if profiling shows a bandwidth bottleneck. The
> *queues* are SoA `uint` arrays; the per-lane path record is AoS.

**Set-1 buffers** (`wf_shade_common.slang:34-42`): `wfState`(0),
`wfHits`(1, `HitInfo`), then the counting-sort queue buffers — `wfLaneSlot`(2,
slot per lane), `wfSlotCount`(3, `[NUM_SLOTS]`), `wfSlotOffset`(4, queue base
per slot), `wfSlotQueue`(5, grouped lane indices), `wfSlotCursor`(6, scatter
cursor), `wfIndirectArgs`(7, `[NUM_SLOTS*3]`). Allocated in
`WavefrontPathPass.__init__` (`vk_wavefront.py:520-527`); `indirect` buffer
created with `INDIRECT_BUFFER` usage.

**Material bucketing = 2 slots, not per-material** — `wfSlotForType`
(`wf_shade_common.slang:47`): `MATERIAL_TYPE_FLAT → WF_SLOT_FLAT(0)`, everything
else → `WF_SLOT_OTHER(1)`. Slot 0 = `wfPathShadeFlat` (flat + MaterialX graph),
slot 1 = `wfPathShade` (heavy catch-all: skin / BSSRDF / python / debug).
`wfQueueLane(slot, qpos)` resolves the stream slot or `0xFFFFFFFF` past the
slice (`wf_shade_common.slang:52-56`).

> The standalone `compaction.slang`, `scatter.slang`, `build_args.slang`,
> `shade_Marble_3D/Tiled_Brass/Tiled_Wood.slang`, and `indirect_paint.slang`
> are **de-risk / verification carriers** and an *alternate* per-material
> partition (`ShadePassGroup`, `renderer.py:1602`), **not** the live path-tracer
> hot loop. The live loop uses the 2-slot in-shader counting sort embedded in
> `wavefront_path.slang`. The per-material `shade_*` kernels are albedo-only
> debug visualisers (`accumBuffer[pixel] = base_color`).

---

## 3. How a bounce is split (vs the megakernel loop)

The megakernel runs generate→trace→shade→bounce in registers, one thread per
pixel ([Megakernel.md §3](Megakernel.md)). The wavefront relocates the **same
loop body verbatim** across dispatches, round-tripping `WavefrontPathState`
through VRAM between bounces. Each kernel calls the same
`evaluateBounce` / `evaluateFlatBounce` and the same MIS, so it is an unbiased
estimator of the same integral (A/B-verified).

Key differences in the mechanism:

- **No ping-pong double-buffer.** Lanes stay in fixed stream slots
  `[0, stream_size)`; liveness is the `PATH_FLAG_ALIVE` bit. Dead lanes are
  skipped by the alive check in each kernel and excluded from the per-bounce
  counting sort (scatter enqueues only alive lanes, `wavefront_path.slang:180`).
- **The "queue" is per-bounce material routing, not inter-bounce compaction.**
  There is no cross-bounce stream compaction in the live loop (standalone
  `compaction.slang` is unused there).
- **Bounce control flow lives on the CPU** — `MAX_BOUNCES` fixed iterations with
  barriers, recorded into the command buffer.
- `wfFinishShade` (`wf_shade_common.slang:70-136`) is the material-independent
  bounce tail (accumulate, RR, spawn next ray, sphere-light MIS, store/terminate)
  shared by both shade kernels.

---

## 4. BDPT wavefront stages (`wavefront_bdpt.slang`)

Aux record `WfBdptAux` carries `eyeLen/lightLen/escaped/radiance/pixel/rngState`
plus transient eye-walk state. Set 1: `wfEye`(0), `wfLight`(1), `wfAux`(2),
queue buffers 3-8. Kernels:

| Kernel | Role | Line |
|--------|------|------|
| `wfBdptWalk` | fused eye + light walk + s=1 splat | `:109` |
| `wfBdptGenEye` / `wfBdptBounceEye` | staged eye walk (verbatim `randomWalk` body), indirect over slot 0 | `:335` / `:489` |
| `wfBdptWalkClassify` | flag live (`ewFlags & ALIVE`) lanes for next bounce | `:468` |
| `wfBdptGenLight` / `wfBdptBounceLight` | symmetric light walk | `:676` / `:715` |
| `wfBdptLightTail` | fused light + splat (`eye` mode) | `:273` |
| `wfBdptSplat` | standalone s=1 splat (`eyeLen≥2 && lightLen≥2`) | `:851` |
| `wfBdptClassify` | route by subpath shape → finalize escaped / SLOT_FULL / SLOT_NEE | `:879` |
| `wfBdptBuildArgs` / `wfBdptScatter` | counting sort | `:907` / `:925` |
| `wfBdptConnectNee` | emissive + `connectT1` only | `:1004` |
| `wfBdptConnectFull` | adds t≥2 `connectGeneric` + `misWeight` | `:1017` |
| `wfBdptResolve` | running mean of the **eye-side** estimate into `accumBuffer` | `:1027` |

The connect estimator is reused verbatim from `integrators/bdpt.slang`
(`randomWalk`, `connectT1`, `connectGeneric`, `misWeight`, `splatLightWalk`,
`bdptEnvNEE`). The NEE/FULL split skips the heavy O(s·t) generic+MIS double loop
for NEE-only lanes. The s=1 light splat accumulates separately in
`lightSplatBuffer` (composited only on display, never into `accumBuffer`), so
headless A/B compares the eye-side estimate only (`:1042-1047`).

---

## 5. Material-sorted shading & codegen

The **live path uses no per-material codegen** — just 2 fixed slots (flat vs
catch-all). The per-material codegen is the *alternate* `ShadePassGroup`
partition: `emit_wavefront_shade_module` (`vk_compute.py:302-345`) emits one
`shade_<name>.slang` per `GraphFragment`, importing only `generated.<name>_graph`
so each is an independent compilation unit; `compile_shade_module_cached`
(`vk_wavefront.py:79-129`) is a per-module content-hash SPIR-V LRU cache keyed
on entry + flags + module source + graph deps + `shared_shader_hash` (which
excludes `generated/` and `wavefront/shade_*`, so adding one material misses
only that key). This is the **MoltenVK compile win**: each shade kernel is
small instead of one monolithic switch.

`flat_bounce.slang::evaluateFlatBounce` is the `MATERIAL_TYPE_FLAT` branch of
`evaluateBounce` standalone (imports only flat material + NEE +
`sampling.proposal`), keeping `wfPathShadeFlat` small.
`scatter.slang`/`build_args.slang`/`compaction.slang` are reusable
counting-sort/compaction primitives (used by the de-risk `IndirectPaintPass`,
mirrored inline in the live kernels).

---

## 6. Indirect args & queue counters

- **Persistent stream, not persistent threads:** fixed `stream_size` slots,
  tiled over pixels; no atomic work-stealing.
- **Counters:** `wfSlotCount` (atomic-incremented in intersect/classify),
  `wfSlotCursor` (atomic write cursor in scatter), `wfSlotOffset` (prefix-sum
  bases). `slot_count` + `slot_cursor` zeroed each bounce via `vkCmdFillBuffer`
  (`vk_wavefront.py:622-632`).
- **Indirect dispatch:** `wfBuildArgs` writes a `VkDispatchIndirectCommand` per
  slot; shade/connect kernels run via `vkCmdDispatchIndirect(indirect, slot*12)`
  so each covers exactly `ceil(count/64)` groups (empty slots dispatch 0
  groups). Barriers carry `DRAW_INDIRECT_BIT` + `INDIRECT_COMMAND_READ_BIT` so
  build-args writes are visible to the dispatch (`vk_wavefront.py:609-620`).
  `indirect_paint.slang` + `IndirectPaintPass` verify indirect == direct
  equivalence.

---

## 7. Proposal / scene-sampling seam

The seam (`sampling/proposal.slang`: `sampleBounceDirection` /
`mixtureProposalPdf`) is driven by `fc.proposalMask` + `fc.proposalAlpha` packed
into the **shared scene UBO (set 0, binding 0)** at `renderer.py:6839-6840`.
Because the wavefront passes bind set 0 verbatim, they read the **same `fc`
UBO** — the seam is active in wavefront mode with **no wavefront-specific
plumbing**. `flat_bounce.slang:44-50` calls `sampleBounceDirection` through the
seam; the catch-all routes through `integrators.path::evaluateBounce` which also
uses it (commit `5e17c8f`: "route wavefront flat-shade through the proposal seam
+ wavefront parity"). The default BSDF-only mask collapses to the material's
native `sample()` for bit-exact parity. Shipped proposal bits: `0x1` BSDF
(always on), `0x2` environment importance, `0x4` **neural** (the learned
spline-flow proposal below). `proposalWeights()` renormalises the active bits'
selection weights to Σ = 1 per lane, so any subset stays unbiased.

### Proposal seam: neural directional proposal (proposal bit2, wavefront-only)

The **neural directional proposal** (`--proposals bsdf,neural`, proposal bit
`0x4`) is a learned, position-conditioned **rational-quadratic neural spline
flow** that proposes the BSDF bounce direction toward the incident-light-aware
integrand and reports an **exact solid-angle pdf**. The net is **frozen,
offline-trained per scene** (in a standalone `spline_flow` PyTorch repo) and
**wavefront-only** — on both Vulkan and native Metal (see
[the Metal wavefront backend](#metal-wavefront-backend) section): the MLP is
infeasible inline in the megakernel under
MoltenVK's big-kernel limit, so the megakernel **strips the bit** and falls back
to its analytic proposal subset (renormalising `{bsdf, env}` to Σ = 1), mirroring
the [ReSTIR DI → identity](#reuse-seam-restir-di-reusemode--1-wavefront-only)
capability gate. Selecting it on the megakernel warns once rather than silently
dropping the request.

The seam stays **provably unbiased regardless of the net's quality** (one-sample
MIS divides throughput by the full mixture pdf), so an untrained / dummy net only
costs variance — never bias. That is exactly what the current Stage-1 milestone
verifies before any training.

**Option A — pre-pass + inline inverse.** Rather than evaluate the MLP inside the
hot bounce kernel, the forward draw is amortised across the live lanes by a
**compute pre-pass**:

- **`WavefrontNeuralProposalPass`** (`shaders/wavefront/neural_proposal_pass.slang`
  → `wfNeuralProposal`) runs once per live lane each bounce, scheduled **between
  scatter and shade** (the same slot the ReSTIR DI burst uses). It reads the
  lane's `HitInfo` + path state, builds the condition, draws **one forward** flow
  sample, and writes a per-lane `WfNeuralSample {wi, pdf, version, valid}` (32 B)
  to **set-1 binding 8** (`wfNeural`, owned by `WavefrontPathPass`). The base
  sample is hashed from the global pixel + frame + bounce — **independent of the
  shade kernel's RNG stream**, so enabling neural never perturbs the
  `{bsdf}`/`{bsdf,env}` draw paths.
- The **flat wavefront shade kernel** reads that record and mixes the precomputed
  forward direction into the bounce via one-sample MIS (`flat_bounce.slang` →
  `sampleBounceDirection`). Neural is scoped to the **lean flat** shade kernel
  only — the skin / catch-all kernels never run it.
- The **inverse** pdf — needed for *arbitrary* directions the flow did not draw
  (NEE light directions, BSDF-/env-selected bounce directions) — is evaluated
  **inline** in the flat shade (`neuralPdfWorld` in `sampling/neural_proposal.slang`,
  called from `sampling/proposal.slang` + `nee.slang`), because those directions
  are RNG-drawn in-kernel and cannot be precomputed. The inverse is the only
  inline MLP eval, and its three weight buffers (bindings 33/34/35) are always
  bound so the static reference resolves on every pipeline, including the
  megakernel (seeded with an all-zero dummy net until activation).

**Unbiasedness.** `proposalWeights()` (`sampling/proposal.slang`) gives the
per-lane effective one-sample-MIS weights: neural participates **only when its
bit is set AND the lane has a precomputed sample** (`neuralValid` — true only on
flat wavefront lanes); otherwise `{bsdf, env}` renormalise to Σ = 1. The **same**
effective weights + the inline inverse drive both the bounce-mixture pdf and the
NEE companion pdf (`mixtureProposalPdf`), so direct and indirect lighting stay
MIS-consistent. A `neuralActive` bool is threaded through
`sampling/reuse.slang` / `nee.slang` so the NEE companion includes the neural
term **exactly on neural lanes**.

**Condition encoding (canonical).** The condition handed to the flow is 9-dim,
raw-concatenated (no hashgrid) and **must match the offline trainer byte-for-byte**
(a mismatch raises variance silently, never bias):
`c = [ (pos − bboxMin)/bboxExtent·2 − 1  (∈ [−1,1]³), shadingNormal.xyz,
outgoingWorldDir.xyz ]`. The scene AABB (`fc.sceneBoundsMin` /
`sceneBoundsExtent`) rides in the UBO tail (read only when neural is active). The
flow hemisphere is y-up in the shading frame. The math itself lives in
`shaders/sampling/neural_flow.slang` (coupling layers, RQ-spline forward/inverse,
MLP, log-det); `neural_proposal.slang` is the thin renderer adapter (weight
buffers, world↔flow-local frame map). Default architecture: 6 layers, 24 spline
bins, hidden 96, condition 9 — the size + precision are now **build-time
configurable** (see [Neural size & precision](#neural-size--precision-tuning-neural-precision-size-study)).

**Weights file (NFW1).** Per-scene weights are baked to a little-endian `NFW1`
binary (`uint32 magic=0x4E465731, version=1; uint32 layers,bins,hidden,cond;`
then a `NfLayerHeader` table `(weightOffset, biasOffset, inDim, outDim)`, the
flat `f32` weights, and the flat `f32` biases), loaded host-side by
`src/skinny/sampling/neural_weights.py` and uploaded to bindings 33/34/35
(`NeuralProposal` plugin in `sampling/proposals.py`). `make_dummy_weights()`
bakes the all-zero bring-up net.

**Parity regression.** `tests/test_neural_parity.py` locks the Slang port
against the PyTorch reference: it re-implements `neural_flow.slang` in numpy off
the *same* flat `NFW1` layout (`headers/weights/biases`) and asserts the forward
`(wi, solid-angle pdf)` and the inverse pdf match committed PyTorch goldens
(`tests/data/neural_parity/`, baked once by `generate_goldens.py` with the
`spline_flow` torch venv). It runs in CI with **no torch and no GPU**; the proven
bar is `|Δwi| < 1e-4` / relative pdf `< 1e-3` (achieved `~4e-6` / `~5e-5`, with a
machine-precision forward↔inverse round-trip). An optional slangpy test dispatches
the real `sampleNeural`/`pdfNeural` for the true on-device gate and skips where the
typed header buffer cannot bind (then GPU parity rides on the headless bring-up).

**Offline pipeline + status.** Stage 1 (this change) is complete end-to-end. The
**record dump** is a *separate megakernel entry* `mainImageRecord`
(`integrators/path_record.slang`) — an RR-free path tracer that, per flat/graph
reflective bounce, appends `(position, normal, wo, wiLocal, contribution)` to
bindings 36/37 (`Renderer.dump_path_records` → a `.nrec` file). Megakernel hosts
it because one thread owns the whole path, so the tail radiance `Li` along each
sampled `wi` is known at loop end and attributed back from a local register stack
(`contribution = (L_final−L_k)/beta_in_k = f·cos·Li`). The dump is offline, so
megakernel-only costs nothing at render time. `mainImage` never references 36/37 →
byte-identical. The offline `spline_flow/render_records.py` fits the flow from
those records by contribution-weighted MLE (`q ∝ f·Li·cos`) under the **identical**
`neuralCondition` encoding and bakes `NFW1`.

#### Wavefront-native record emission (`wavefront/wf_records.slang`)

For the **live** online-training drain (Stage 2), dispatching that megakernel
each frame is fatal on NVIDIA/Windows: the 8 MB record entry ~400 s-compiles a
driver pipeline and a single dispatch exceeds the 2 s Windows TDR, losing the
device. The wavefront path integrator therefore emits the **same** `PathRecord`
stream directly. Because the path is smeared across per-bounce dispatches, the
tail radiance is not resident at the bounce that sampled `wi`, so the snapshots
live in a **per-lane VRAM vertex stack** instead of registers:

- **Storage.** `RecVertex[stream·REC_MAX_BOUNCES]` + a per-lane count, in two
  *separate* set-1 buffers (bindings 9/10) — **not** inside `WavefrontPathState`,
  which is copied by value in every kernel (inlining a 6-deep stack there would
  spill to scratch and ~8× its bandwidth in every kernel). Full-size only while
  recording; 1-element dummies otherwise. On the Metal records build the
  per-lane count is folded into the stack as a header element (slot cap — see
  the [Metal wavefront backend](#metal-wavefront-backend) records bullet).
- **Push** (`wfPushRecord`, in `wfFinishShade`): on a guideable bounce
  (flat/python, reflective, sampled dir in the HitInfo-normal upper hemisphere)
  snapshot `(pos, N, wo, wiLocal, L_k, beta_in, depth)` **before** the throughput
  update — the same guard and snapshot order as `estimateRadianceRecord`.
- **Splat** (`wfEmitRecords`): at lane termination — `wfTerminate` (miss / RR /
  no-valid-bsdf / sphere-light hit) and the max-depth survivors at
  `wfPathResolve` — `L_final` is the lane's radiance, so each stacked vertex emits
  `recordContrib(L_final, L_k, beta_in)` to bindings 36/37 via the shared
  `emitRecord`, dropping non-finite. `recordContrib` (`path_record_common.slang`)
  is the ONE source of truth shared with the megakernel dump.
- **Gate.** All of the above is behind `fc.recordMode`, set only while the
  wavefront record drain is active — so the default render takes no stack writes,
  no appends, and is **bit-identical** (verified: recordMode off vs on diff = 0).
- **Drain.** `Renderer.drain_path_records_to_replay` is source-selectable
  (`_record_source`: `auto` picks wavefront for the wavefront path integrator,
  else megakernel). In wavefront mode the normal render already filled 36/37, so
  the drain just reads the counter + buffer and resets it — no extra dispatch.
  The megakernel `mainImageRecord` stays available for the offline `.nrec` dump
  and as a forced (`megakernel`) drain source on non-TDR boxes.

**Landed + GPU-proven (Mac MPS/MoltenVK):** the full plumbing, a real scene-trained
net (4.36M Cornell records → trained flow, pdf ∫≈1), loaded + A/B'd in-renderer.
The equal-time gate (`test_neural_trained_equaltime_gate`) is **measured, not won
on Mac**: the net is unbiased (mixture-MIS) but the MLP pre-pass is ~28× a bsdf
bounce, and the flat ceiling-lit Cornell box is broad-indirect (cosine already
near-optimal) so the guide ≈ cosine and adds a firefly tail with no offsetting
win. The equal-time WIN is a follow-up — GPU-optimised inference (the deferred
CUDA-perf goal), guiding-iteration training (sample from learned `q`, retrain — the
one-shot `{bsdf}`-generated data caps net quality), a concentrated-indirect scene,
and a firefly-robustness measure (pdf floor / lower neural α). Online / dynamic
training (the per-sample `neuralNetworkVersion` hook) lands in Stage 2 — see
[Online neural training: frame-end weight swap](#online-neural-training-frame-end-weight-swap) below.

**Host wiring.** Weight buffers + `WavefrontNeuralProposalPass` are built lazily
in wavefront mode (mirroring `RestirDiPass`) in `renderer.py` / `vk_wavefront.py`.
The GUI preset is **"BSDF + Neural"** (`proposal_preset_index`, persisted; env
`SKINNY_PROPOSALS=bsdf,neural`); changing it resets progressive accumulation.

### Online neural training: frame-end weight swap

Stage 2 lets the neural proposal train **continuously** while the scene
animates, so the net adapts instead of staying frozen on a per-scene offline
bake. An async trainer publishes fresh weights at any time; the renderer
**never** touches the inference buffers mid-frame. The whole drain→train→
publish→swap loop is documented in
[Architecture.md § Online neural training](Architecture.md#online-neural-training);
this section covers only the wavefront-side commitment point.

**Render weights are frozen for the duration of a frame.** The
`WavefrontNeuralProposalPass` pre-pass and the inline inverse in the flat shade
both read bindings 33/34/35 and the active `neuralNetworkVersion`, which stay
fixed while the command buffer for a frame executes. The single point where new
weights are promoted is the **frame boundary**:

- `Renderer._online_frame_end_swap()` runs **after the fence wait** in
  `render_headless` (and **after present** in `render`) — once the GPU is done
  reading the current weights. It `publisher.swap()`s the trainer's pending
  weights into the render slot and increments the network version **in both places
  the per-sample density key is read**: `FrameConstants.neuralNetworkVersion` (the
  inline inverse) **and** the `WavefrontNeuralProposalPass` push-constant stamp
  (the forward pre-pass). The two are kept in lockstep so a forward draw and its
  inverse pdf always agree. How the pending weights reach bindings 33/34/35
  depends on the backend: the **file** backend re-uploads them via
  `_apply_render_weights`; the **interop** backend's `swap()` instead **host-waits
  the exported timeline semaphore** to the staged version (so the CUDA write is
  provably resident) and re-stamps the version with no re-upload —
  `acquire_for_render()` returns no host weights because the CUDA write already
  populated the bound buffers.

**Why a mid-flight swap is still unbiased.** Because the weights a frame draws
with are exactly the weights its per-sample density is evaluated against (the
version stamp guarantees it), an asynchronous swap can only change *which*
frozen net a given sample saw — never the consistency of its one-sample-MIS
weight (`β ·= f·cos / p_mix`). A stale or freshly-swapped net therefore raises
**variance only, never bias**; mixture-MIS unbiasedness is preserved across the
swap exactly as it is across an untrained net. The two weight-handoff backends
(`file` hot-reload vs CUDA↔Vulkan `interop`) differ only in *how* the pending
weights reach the buffer, not in this swap discipline.

### Neural size & precision tuning (`neural-precision-size-study`)

The network **size** (`NF_LAYERS`/`NF_BINS`/`NF_HIDDEN`) and **inference
precision** are build-time configurable, with the default (`fp32 @ 6/24/96`)
**byte-identical** to the shipped proposal. The single host source of truth is
`NeuralBuildConfig(layers, bins, hidden, precision)`
(`skinny.sampling.neural_weights`); the renderer takes it as `Renderer(...,
neural_config=…)` and threads it into every neural `.spv` compile (its `-D`
flags) + the weight upload (its dtype). The default config emits **no** `-D`
flags, so its cache key + SPIR-V are unchanged.

- **Size** — `NF_LAYERS/NF_BINS/NF_HIDDEN` are `#ifndef`-guarded defines
  (`neural_flow.slang`); a non-default size recompiles the flow modules (the
  wavefront pre-pass, the inline inverse in the flat shade, the record entry)
  via `slangc -D` and bakes/loads an NFW1 at those dims (the loader arch assert
  is the mismatch guard). The host (`export_weights.export_flow`) is already
  size-parametric.
- **Precision** — two compile-time aliases: **`NF_WT`** (weight *storage*, the
  `StructuredBuffer` element at 33/34) and **`NF_CT`** (the MLP GEMM
  *accumulate*). Three modes: **fp32** (`float`/`float`), **fp16-storage**
  (`half`/`float` — ½-byte weights, float GEMM) and **fp16-compute**
  (`half`/`half`). The **RQ-spline math + the returned solid-angle pdf stay
  `float` in every mode** (catastrophic-cancellation prone in fp16). NFW1 stays
  fp32 on disk; the host casts to half at upload. `vk_context` probes
  `shaderFloat16` + `uniformAndStorageBuffer16BitAccess` and enables them when
  present; an fp16 mode on a device without the capability **falls back to fp32**
  (logged) rather than failing.

**Study (it is a measurement, not a product).** A two-track harness maps quality
vs cost across the size×precision grid on MoltenVK: the *precision* track reports
fp16 pdf-parity drift vs the fp32/PyTorch reference
(`tests/test_neural_parity.py`, scene-independent — measured drift on the Cornell
net is `~4e-4` (storage) / `~1e-3` (compute), negligible); the *size* track
retrains per size in `spline_flow/bake_grid.py` and reports held-out NLL. Both
add an in-renderer unbiased + firefly check and the MoltenVK ms/frame +
weight-buffer bytes (`tests/study_size_precision.py` → a CSV + `RESULTS.md` with
the Pareto knee and a recommended ship config). Every fp16/size mode stays
**unbiased** (mixture-MIS; gated by `test_fp16_unbiased_gate`): the converged
Cornell image matches the fp32 reference within noise, and the fp16 weight buffer
is exactly **half** the bytes. The gates: `test_default_config_byte_identical`
(4.1, the default `-D`-free build) + `test_fp16_unbiased_gate` (4.2, skips on
devices without fp16).

**Study result (M5 Pro / MoltenVK, flat Cornell box — `docs/diagrams/neural_study/RESULTS.md`).**
On this broad-indirect scene held-out NLL is **flat across size** (~2% spread,
−0.276…−0.282) — a smaller net fits nearly as well — and **fp16-compute is the
fastest precision in 6/7 sizes** (half ALU + bandwidth; the cross-size ms/frame
is thermally noisy, so weight bytes is the clean cost axis). The recommended
**Pareto knee is `L6 B24 H48 @ fp16-compute`** — **75 KB, 18% of the baseline
fp32's 412 KB**, NLL within 2% of best, unbiased + firefly-bounded. The headline:
on Apple Silicon the net can shrink hard **and** drop to fp16 at negligible
quality cost, so the equal-time loser (Stage 1's MLP pre-pass cost) has real
headroom before the CUDA stage. *Shipping* a chosen config is a later change; the
study only recommends one.

### Reuse seam: ReSTIR DI (`reuseMode == 1`, wavefront-only)

The reuse hook (the other half of the scene-sampling seam) is realized by
**ReSTIR DI** — reservoir spatiotemporal resampling of **primary-hit** direct
lighting. `reuse=none` (identity) forwards to stock NEE; `reuse=ReSTIR DI` is
**wavefront-only** (multi-pass) and capability-gates to identity on the
megakernel of either device (`reuseMode` folds to 0 in `_pack_uniforms`); both
wavefront backends run it (Vulkan `RestirDiPass`, Metal
`metal_wavefront.MetalRestirDiPass`). This section is the
wavefront-integration summary; the **full reference** — equations, the
equation→shader map, design choices, and every GUI control — is in
**[ReSTIR.md](ReSTIR.md)**.

![ReSTIR DI fill → spatial → resolve pipeline](diagrams/restir_pipeline.svg)

- **GPU side** (`vk_wavefront.RestirDiPass`): three pipelines compiled from
  `restir/restir_primary.slang` — `restirFill` → `restirSpatial` → `restirResolve`
  — over **set 1** = the shared path-state (0) + hit (1) buffers plus ReSTIR-owned
  **ping-pong reservoirs A/B (2,3)** + a **G-buffer (4)** (pos+normal). A 36-B
  push constant (`RestirPC`) carries the config (flags: bit0 spatial / bit1
  temporal / bit2 biased, plus `mLight`/`mBsdf`/`spatialK`/`radius`/thresholds/`mCap`).
- **Schedule:** `WavefrontPathPass.record_dispatch` calls `record_primary_direct`
  at **bounce 0**, after the primary intersect and before shade (and restores the
  `wfTile` push constants — ReSTIR binds an incompatible pipeline layout). Shade's
  depth-0 `reuseDirect` is gated to 0 so ReSTIR is the sole primary-NEE source; the
  result is added into the path-state radiance, and `wfPathResolve` flushes it as
  usual.
- **Canonical integration (RIS owns primary direct):** `restirFill` runs an initial
  RIS over the **unified light set** with both LIGHT- and BSDF-sampled candidates.
  The target is the UNWEIGHTED `p̂ = luminance(f·Le)`; the technique MIS lives in the
  balance-heuristic mixture source pdf `p_mix = (M_light·p_light + M_bsdf·p_bsdf)/M`.
  Light candidates draw sphere / emissive-triangle / env ~ uniform over the active
  techniques; BSDF candidates trace the proposal direction to the sphere lights / env
  (sphere hits recover a reproducible uv, env stores an octahedral direction). The
  path tracer's own depth-0 BSDF-hits-sphere (`wf_shade_common`) and env-miss
  (`wavefront_path`) terms are **gated off** so they are not double-counted. Emissive
  triangles are NEE-only (the stock renderer has no BSDF-tri MIS) → light-technique
  only. Directional (delta) lights are plain NEE outside the RIS. `restirResolve`
  casts one shadow ray for the survivor → `f·V·W`. **Converges to `reuse=none`** on
  cornell_box_sphere/emissive + three_materials (glossy), A/B-verified against
  megakernel-PT / BDPT / wavefront-NEE (all agree).
- **Unbiased spatiotemporal combination (GRIS):** `restirSpatial` merges self +
  k domain-checked screen-neighbours + last frame's reservoir using the
  **generalized balance heuristic** — each source's MIS weight
  `m_s = M_s·p̂_s(z_s) / Σ_j M_j·p̂_j(z_s)` re-evaluates the survivor's target in
  every source's OWN domain (the source lane's material/frame re-loaded from
  `wfHits[j]` via `restirLoadLane`; the DI same-light-point reconnection has
  Jacobian 1). This bounds the spatial→temporal feedback that the old biased (ΣM)
  combination let explode on glossy surfaces. A **biased toggle** (`flags` bit2)
  selects the faster ΣM combination (skips the O(k²) re-eval; bounded on spatial,
  over-brightens with temporal on glossy).
- **Regime selector + tuning:** `restir_regime_index` → `_disc("ReSTIR regime", …)`
  (default **Spatial only** / Spatial+Temporal / Temporal only), plus live
  push-constant tuning (`_disc`/`_cont`: combine, M_light, M_bsdf, neighbours,
  radius, M_cap) refreshed each frame with no pass rebuild. All reset accumulation.
- **Progressive-regime note:** on skinny's progressive accumulator, temporal reuse
  **double-counts correlated history** (it fights the accumulator's own frame
  averaging) — bias scales with `M_cap` and shows on glossy surfaces — so the default
  is spatial-only. Proper deep temporal is the real-time **reprojected** regime (a
  P3 follow-on; reserved in the selector). Spatial reuse is unbiased (GRIS) and
  reduces variance on many-light scenes (`assets/restir_variance_demo.usda`,
  `tests/test_restir_variance.py`: ~30% lower RMSE than NEE at equal low spp).

---

<a id="metal-wavefront-backend"></a>

## Metal wavefront backend (`wavefront_driver.py` + `metal_wavefront.py`)

Since change `metal-wavefront-parity`, the wavefront execution mode runs on the
**native Metal backend** at parity with Vulkan. The loop *stage orders* live in
the backend-neutral **`wavefront_driver.py`** (`record_path_loop` /
`record_bdpt_loop` over the `WavefrontRecorder` protocol) — the single source
of truth both backends drive. `vk_wavefront.py` supplies the Vulkan recorder
(byte-identical to the historical inline command recording);
**`metal_wavefront.py`** supplies the Metal primitives:

- **`MetalWavefrontPathPass` / `MetalWavefrontBdptPass`** — per-entry
  in-process Slang→Metal pipelines (one session, `SKINNY_METAL=1`, no
  `slangc`/SPIR-V), with the path-state / hit / counting-sort queue buffers
  sized from the **reflected MSL strides** (Slang pads `float3` to 16 B on
  Metal — e.g. `WavefrontPathState` 96 B vs 68 B scalar, `BDPTVertex` 176 B vs
  the 128 B scalar headroom; host constants would undersize them).
- **`MetalRestirDiPass`** — the ReSTIR DI fill→spatial→resolve set; persistent
  ping-pong reservoirs + G-buffer at reflected strides; the 36 B `rpc` config
  rides the `set_data` uniform-blob mechanism.
- **`MetalNeuralProposalPass`** — the neural forward pre-pass. Selecting the
  neural proposal rebuilds the pass set with **`SKINNY_METAL_NEURAL=1`**, which
  un-stubs the frozen-weight buffers and the inline inverse pdf. Metal caps a
  kernel's argument table at **31 buffer slots**, and slots are assigned
  program-wide in declaration order — the un-stubbed program fits because the
  three buffer globals dead in every wavefront kernel (`toolBuffer`,
  `recordBuf`, `recordCounter`) are compiled out under that define. Weights
  upload via `set_data`, or in place through the UMA shared-storage interop
  publisher under `--neural-handoff interop` (change `metal-neural-interop`);
  fp32 on current slang-rhi (the Metal fp16 probe under-reports).
- **Records build (`SKINNY_METAL_RECORDS=1`, change `metal-record-drain`)** —
  arming the wavefront record drain (online training) rebuilds the path pass
  with the record emitters un-stubbed. The neural build already sits exactly
  at the 31-slot cap, so the records flavor frees two slots by compiling out
  the two resolve globals inert on a training render (`lightSplatBuffer` —
  records are path-only, where the splat is zero — and `gizmoSegments` — no
  gizmo overlay on training frames) and folds both counters into their data
  buffers: the per-lane count lives in a stack header element
  (stride `REC_MAX_BOUNCES + 1` per lane), and `recordBuf` becomes a
  `RWByteAddressBuffer` whose 64-byte header carries capacity (byte 0) and the
  atomic count (byte 60), with packed 64-byte records from byte 64 — the
  byte-address form also keeps the drained stream byte-for-byte the Vulkan
  layout despite MSL's float3 padding. Records-off renders are bit-identical
  through an arm → disarm round-trip; records-on costs ≈0.35 ms (~4.6 %) per
  128² Cornell frame. The megakernel record *source* stays Vulkan-only; the
  Metal drain always reads the wavefront-native stream
  (`tests/test_metal_record_drain_gpu.py`).
- **One `MetalFrameEncoder` per frame** — every stage encodes into a single
  command encoder with a global compute barrier between stages (no per-stage
  `wait_for_idle`), submitted once.
- **Indirect dispatch fallback** — slang-rhi 0.42's Metal backend silently
  no-ops `dispatchComputeIndirect`, so the logged capability probe resolves
  false and the per-material shade uses the CPU slot-count-readback fallback:
  flush the encoder, read the GPU-written `(x, y, z)` triple, issue an
  equal-count direct dispatch.

**Parity** (guarded A/B tests, `RUN_METAL_WAVEFRONT_COMPILE=1` under
`scripts/guarded_metal.sh`): path, all three BDPT walk modes, and ReSTIR DI are
**bit-identical** to the Vulkan wavefront render on this host (both backends
end on the same Metal GPU); the neural proposal matches at rel-MSE 0.00000 /
correlation 1.00000 with ~91 % of pixels exactly equal, and stays unbiased
(`tests/test_metal_{wavefront_path,wavefront_bdpt,restir,neural}_ab.py`).

**Equal-time** (`scripts/bench_metal_equal_time.py`, three-materials demo @
256², M5 Pro, steady-state): megakernel path **438.8 fps** (28.8 Mspp/s) vs
wavefront path **97.5 fps** (6.4 Mspp/s) — wavefront ≈ **0.22×** the
megakernel; wavefront BDPT 68.1 fps (the megakernel BDPT case is not benched —
its Metal kernel compile is impractically long, the original MoltenVK
big-kernel pain point in native form). The gap is dominated by the per-bounce
CPU-readback fallback's host syncs; it closes when slang-rhi ships a working
Metal indirect dispatch (the probe flips automatically). Mode selection stays
**user-driven** — there is no silent performance fallback from wavefront to
megakernel; the megakernel remains the default on both backends.

---

## Megakernel vs wavefront

| Axis | [Megakernel](Megakernel.md) | Wavefront |
|------|------------|-----------|
| Dispatches / frame | **1** (`vkCmdDispatch`) | ~5 stages × 6 bounces × tiles + resolve |
| Path loop | register-resident `for` loop, one thread/pixel | torn across kernels; state in VRAM |
| Path state | registers (per thread) | `WavefrontPathState` 68 B AoS in VRAM, ×stream |
| Material handling | runtime tag-switch per pixel (divergent) | counting-sort into 2 slots, indirect per-slot dispatch (coherent) |
| Inter-stage data | none | queue buffers + indirect args, round-tripped each bounce |
| Divergence | high (mixed materials serialise in a warp) | reduced (shade dispatch hits one material branch) |
| Register pressure | high (one fat kernel) | lower (small per-stage kernels) |
| Memory bandwidth | low (registers) | higher (VRAM state round-trip) |
| CPU overhead | minimal command recording | heavy command recording, per-bounce fills/barriers |
| Compile size | one large kernel — can trip MoltenVK Metal limit | many small kernels — sidesteps the limit |
| Accumulation | inline in `main_pass` | `wfPathResolve` / `wfBdptResolve` |
| VRAM vs resolution | output buffers scale with W×H | path stream capped + tiled, independent of resolution |

**Why both exist.** The megakernel is the simple reference path, but its single
fat kernel suffers warp divergence and register pressure, and on MoltenVK can
exceed the Metal compiler's kernel-size limit. The wavefront mode splits the
same estimator into small per-stage / per-material kernels connected by GPU
queues: same math, better material coherence, each kernel small enough to
compile. The cost is memory bandwidth (state round-trips through VRAM every
bounce) and CPU-side command-recording overhead. `build_catchall=False` skips
the heavy catch-all kernel entirely for flat-only scenes — the cheapest
MoltenVK-compile case.

Both are **unbiased estimators of the same integral** and A/B-verified to match:
the wavefront kernels are verbatim relocations of the megakernel's per-bounce
body calling the same `evaluateBounce` / MIS / proposal-seam code.

---

## Key files

| File | Role |
|------|------|
| `vk_wavefront.py` | orchestration, all 3 pass classes, per-stage dispatch |
| `wavefront_driver.py` | backend-neutral stage orders (`record_path_loop` / `record_bdpt_loop`, `WavefrontRecorder` protocol) — shared by Vulkan + Metal |
| `metal_wavefront.py` | Metal recorder + pass classes (path / BDPT / ReSTIR / neural) — in-process Slang→Metal, reflected-MSL buffer sizing |
| `wavefront_layout.py` | state stride + queue sizing (Python mirror; `msl=` strides for Metal) |
| `shaders/wavefront/wavefront_path.slang` | path kernels (`_wfpath_*`) |
| `shaders/wavefront/wavefront_bdpt.slang` | BDPT kernels (`_wfbdpt_*`) |
| `shaders/wavefront/wf_shade_common.slang` | set-1 bindings + `wfFinishShade` + slot routing |
| `shaders/wavefront/flat_bounce.slang` | flat-slot shade body |
| `shaders/wavefront/wavefront_state.slang` | `WavefrontPathState` record |
| `shaders/wavefront/{build_args,scatter,compaction,indirect_paint}.slang` | counting-sort / verification primitives |
| `shaders/sampling/proposal.slang` | proposal seam (shared with megakernel); `proposalWeights` / `sampleBounceDirection` / `mixtureProposalPdf` + the neural inline inverse |
| `shaders/sampling/reuse.slang` | reuse seam (`reuseDirect`: identity ⇒ NEE; ReSTIR gate; `neuralActive` thread to the NEE companion) |
| `shaders/sampling/{neural_flow,neural_proposal}.slang` | neural proposal: pure spline flow (fwd/inverse) / renderer adapter (weight buffers 33/34/35, condition + world map) |
| `shaders/wavefront/neural_proposal_pass.slang` (`wfNeuralProposal`) | neural pre-pass — one forward sample per live lane → `wfNeural` (set-1 binding 8) |
| `vk_wavefront.py` (`WavefrontNeuralProposalPass`) | neural pre-pass set (scene set 0 + own 3-binding set 1); scheduled scatter→shade |
| `sampling/{proposals.py,neural_weights.py}` | `NeuralProposal` plugin / NFW1 weight loader + dummy baker |
| `shaders/restir/{reservoir,light_ris,restir_primary}.slang` | ReSTIR DI: reservoir core / unified-light-domain RIS (light+BSDF candidates) + resolve / fill→spatial(GRIS)→resolve passes |
| `shaders/wavefront/{wf_shade_common,wavefront_path}.slang` | depth-0 BSDF-hits-sphere / env-miss gates (ReSTIR owns primary direct) |
| `vk_wavefront.py` (`RestirDiPass`) | ReSTIR DI pass set (reservoirs A/B + G-buffer, bounce-0 hook, 36-B RestirPC) |
| `vk_compute.py:302-345` | per-material wavefront shade codegen |
| `renderer.py:7147-7158` / `7367-7377` | wavefront render gate (windowed / headless) |
| `renderer.py:1424-1462` / `1475-1507` | path / BDPT pass lifecycle |
| `renderer.py:915` | `WAVEFRONT_BDPT_SUPPORTED` |
| `tests/test_wavefront_state.py` | struct-layout lock |
