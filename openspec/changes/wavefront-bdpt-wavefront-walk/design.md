## Context

The wavefront path tracer (`WavefrontPathPass`, `wavefront_path.slang`) is fast.
The wavefront BDPT pass (`WavefrontBdptPass`, `wavefront_bdpt.slang`) is slow.
Both run per accumulation sample; the cost gap is structural, not algorithmic
noise.

**Why PT is fast** — per bounce it: traces in a small material-free `intersect`
kernel, classifies live lanes into material slots with an atomic counting sort
(`wfBuildArgs` prefix-sums counts → offsets + `VkDispatchIndirectCommand`,
`wfScatter` places each lane in its slot's queue), then `vkCmdDispatchIndirect`
runs each shade kernel over **only its live, material-coherent lanes**. Dead
lanes (miss/RR) are never scheduled; group count shrinks each bounce.

**Why BDPT is slow** — three current stages (`wfBdptWalk` / `wfBdptConnect` /
`wfBdptResolve`) each dispatch `ceil(stream/64)` groups over the **full** stream:

1. No compaction. Background pixels, non-flat first hits, out-of-frame tail
   slots, and short paths all occupy warp slots in every stage.
2. `wfBdptWalk` is a megakernel: it runs the full eye walk (`randomWalk`, ≤6
   internal trace+sample bounces with per-vertex env-NEE), the full light walk,
   and the s=1 splat — holding `eye[7]`+`lightPath[7]` `BDPTVertex` (~1.7 KB) in
   registers the entire time, crushing occupancy.
3. `wfBdptConnect` is a megakernel running an O(s·t) double loop: up to ~25
   `connectGeneric` calls (each two `bdptSurface`→`loadFlatMaterial`
   MaterialX-graph reloads + a visibility ray + `misWeight`, which copies both
   7-vertex arrays every call), plus `connectT1` over all lights per eye vertex.
   It runs even on directional-only scenes where `lightLen==1` makes the entire
   t≥2 loop a no-op — yet still dispatched over the full stream.

Constraints: estimator math in `integrators/bdpt.slang` is A/B-verified and must
not change — correctness is by reuse. BDPT scope is flat first-hit, pinhole
camera. Vulkan-only. RNG draw order per lane must match the megakernel for
parity. The headless harness (`tests/test_headless.py`) compares accumulation
images in linear HDR and is the safety net.

## Goals / Non-Goals

**Goals:**

- Bring the path tracer's wavefront wins to BDPT: per-bounce live-lane compaction
  + indirect dispatch on both subpath walks, an intersect/extend kernel split,
  and a compacted + strategy-split connect stage.
- Eliminate the `eye[7]`+`lightPath[7]` register arrays in both the walk and the
  connect megakernels — vertices live in VRAM, touched one at a time.
- Preserve the accumulated image (A/B parity with megakernel BDPT, to the
  re-render noise floor) at every incremental step.
- Add per-stage GPU timestamps to baseline and prove the win.

**Non-Goals:**

- No change to the BDPT estimator math, scope (flat first-hit, pinhole), or the
  rendered image.
- No change to the megakernel BDPT, the wavefront path tracer, or the
  execution-mode/integrator selection.
- Not pursuing per-connection material-reload elimination (hoisting
  `bdptSurface(z)` / caching BSDF in the vertex) — that is a separate, orthogonal
  optimization (Tier 2) that stacks on top later.
- No Metal support (wavefront is Vulkan-only).
- Not chasing intra-warp (eyeLen,lightLen) divergence beyond the coarse
  NEE/FULL split; depth-bucket slotting is a possible future refinement.

## Decisions

### D1. Stage both random walks per-bounce, reusing the PT counting-sort

Each walk becomes `gen` → for `b in 0..BDPT_MAX_DEPTH`: `intersect` →
`build_args` → `scatter` → `extend`(indirect). The `intersect` kernel is
material-free (trace + termination classification only); `extend` is the heavy
kernel (build vertex, env-NEE for the eye walk, BSDF sample, RR, write back the
previous vertex's `pdfRev`). Vertices stream to the existing `wfEye`/`wfLight`
VRAM buffers as they are built.

The walk is **flat-only** (a non-flat hit ends the subpath), so — unlike PT's
flat/catch-all split — the walk needs only **one live slot**: alive vs dead. The
compaction win is dead-lane removal + occupancy, not material coherence.

_Alternative — leave the walk a megakernel (1a only):_ rejected. The 1.7 KB
register array and full-stream dispatch live in the walk too; staging it is where
the occupancy recovery comes from.

_Alternative — fuse eye+light into one lockstep dual walk:_ rejected. They die at
different rates and the eye walk has env-NEE/splat work the light walk lacks;
separate staged walks each compact independently and keep kernels small.

### D2. Per-lane walk state in dedicated buffers (model: `WavefrontPathState`)

Replace the register arrays with small per-lane state buffers carried across the
bounce loop:

```
EyeWalkState   { float3 rayOrigin, rayDir, throughput; float pdfFwdOmega,
                 misBsdfPdf; uint idx, rngState, flags; float3 escaped; uint pixel; }
LightWalkState { float3 rayOrigin, rayDir, throughput; float pdfFwdOmega;
                 uint idx, rngState, flags; }
```

`idx` is the current vertex count (the next write index into the lane's
`wfEye`/`wfLight` slice); `flags` carries ALIVE; `escaped` accumulates eye-walk
env-NEE + escaped-env radiance; `rngState` carries the RNG cursor bounce→bounce.
The `pdfRev` back-patch (vertex k writes vertex k-1) is a VRAM read-modify-write
ordered by the existing per-bounce barrier.

### D3. One shared counting-sort scratch set, used at every compaction point

The lane-slot / slot-count / slot-offset / slot-queue / slot-cursor / indirect
buffers (already owned by `WavefrontPathPass`) are added to `WavefrontBdptPass`
as a single set, reused sequentially: eye-walk bounce, then light-walk bounce,
then connect. Barriers already serialize the stages, so no aliasing hazard.
`wfBuildArgs` and `wfScatter` are reused verbatim (factor into a shared module if
include paths require it; otherwise import).

### D4. Splat is its own stage, ordered after the light walk

The s=1 light-tracer splat iterates light-subpath vertices, projects each onto
the camera, visibility-tests, and atomic-adds into `lightSplatBuffer`. Pinhole
`sampleWi` draws no RNG (verified), so the splat consumes no RNG and its
placement cannot perturb per-lane draw order. Keeping it a separate stage after
the light walk also matches the megakernel's draw order exactly.

### D5. Compact + strategy-split connect

After both walks, classify each live lane by subpath shape and route it to one
kernel, indirect-dispatched over its queue:

- `connect_nee`  — `eyeLen≥2`, `lightLen<2`: emissive (t=0) + `connectT1` (t=1).
- `connect_full` — `eyeLen≥2`, `lightLen≥2`: emissive + `connectT1` + generic
  (t≥2) + MIS.

Dead lanes route nowhere. On directional-only scenes every lane is NEE → the
heavy `connect_full` kernel dispatches **zero groups**. Splitting also keeps the
`misWeight` register footprint out of the NEE path. `resolve` stays a full-stream
kernel (cheap; reads `aux.radiance`, folds the running mean).

### D6. Incremental build, A/B at each step

```
S0  GPU timestamps (VkQueryPool) around every stage — baseline ms.
S1  connect compaction + strategy split (D5).            A/B + measure.
S2  stage the eye walk (D1/D2).                          A/B + measure.
S3  stage the light walk + standalone splat (D1/D4).     A/B + measure.
```

Hybrid intermediates work because vertices already live in VRAM: a staged half
hands off to a megakernel tail that reads the same buffers. Each step is
independently revertible.

## Risks / Trade-offs

- **RNG draw-order parity** → Carry the cursor in per-lane state; keep the
  per-lane draw sequence identical (eye-walk → light-origin → light-walk → splat
  → connect). Splat is RNG-free (pinhole). Guarded by the headless A/B harness at
  every step (S1–S3).
- **`pdfRev` back-patch hazard** (extend k writes vertex k-1 in VRAM) → ordered
  by the per-bounce memory barrier already present between stages; the read and
  write target distinct vertices within a lane's slice.
- **Dispatch/barrier overhead** (~50 dispatches/tile × ~8 tiles for 1080p ≈ 400
  `vkCmd`s/frame, plus a single-thread `build_args` + 2 barriers per bounce) →
  recorded once per frame, acceptable; each dispatch is far cheaper (compacted).
  If `build_args` overhead shows up in timestamps, batch the eye/light scratch to
  cut single-thread dispatches.
- **Residual intra-warp divergence** in `connect_full` (eyeLen,lightLen vary
  2..6) → accepted for v1; the coarse NEE/FULL split already removes the largest
  imbalance (no-light vs light). Depth-bucket slotting is a future option.
- **VRAM unchanged-to-slightly-up** → the new walk-state buffers are small
  (stream × ~64 B each); subpath-vertex buffers already exist. Tiled streaming
  keeps everything bounded by stream size, not resolution.
- **More moving parts than the megakernel** → mitigated by reusing the PT
  machinery verbatim and the estimator functions unchanged; the only new logic is
  bookkeeping (state structs, classification, the bounce loop).

## Migration Plan

Internal renderer change; no user-facing migration. `WavefrontBdptPass` is
swapped stage-for-stage behind the existing `execution_mode=wavefront` +
`integrator=bdpt` selection. Rollback is reverting the pass + shader module; the
megakernel BDPT and wavefront path tracer are untouched, so `bdpt` always has a
working path. Land per S0–S3 step, each green on the headless A/B harness before
the next.

## Open Questions

- Should `wfBuildArgs`/`wfScatter` be factored into a shared
  `wavefront/compaction` module imported by both passes, or duplicated? Lean
  shared to keep one source of truth — confirm slangc include paths cooperate.
- Is `resolve` worth compacting too, or is full-stream cheap enough? Decide from
  the S0 timestamp baseline.
- Final tile/stream-size tuning for BDPT once per-stage costs shrink — the
  current `STREAM_CAP = 1<<18` was chosen for the megakernel walk's VRAM; staged
  walks may allow a larger stream (fewer tiles, less per-tile barrier overhead).
