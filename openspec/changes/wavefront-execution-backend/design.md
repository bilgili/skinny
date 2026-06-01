## Context

The renderer is a megakernel. `main_pass.slang`'s `mainImage` compute entry
generates a camera ray, traverses the BVH, evaluates the hit material via
`evalSceneGraph(graphId, ...)` — a `switch` over all material graphs stitched
into the auto-generated `generated_materials.slang` — then bounces in-kernel
until the path terminates, all in registers. The integrator is a *runtime branch*
inside this one kernel: `fc.integratorType` (0 = path, 1 = bdpt) selects
`PathTracer` vs `BDPTIntegrator` per pixel (`main_pass.slang:125`). One compute
pipeline, one `vkCmdDispatch` per frame (`renderer.py:6466`).

Material graphs are generated per-graph as self-contained Slang
(`GraphFragment`, `materialx_runtime.py`); `_emit_generated_materials`
(`vk_compute.py:189`) stitches them into the switch. Adding a graph changes
`_graph_set_signature()` (`renderer.py:2129`) and triggers
`_build_pipeline_for_current_graphs()` — a full `slangc` recompile + pipeline
recreate. `_compile_slang` (`vk_compute.py:556`) caches SPIR-V by
blake2b(source+flags), but any new graph is a new superset → cache miss.

Geometry is monolithic: all meshes concatenated into one shared vertex/index/BVH
buffer (`_upload_meshes_concatenated`, `renderer.py:5572`); per-instance TLAS
records carry per-BLAS offsets (node/tri/vert, `INSTANCE_STRIDE=144`). Any
add/remove re-concatenates and re-uploads the whole set.

Multi-pipeline management already exists as precedent: `vk_skinning.py`
(`SkinningPasses`) owns two independent compute pipelines (skin + BVH refit), each
with its own shader module, descriptor set layout, and per-mesh descriptor sets,
dispatched in their own command buffer before the frame render. There is no
indirect dispatch anywhere yet.

This change adds a switchable wavefront execution backend (path + bdpt), a
per-material pipeline structure that makes adding a material a one-kernel compile,
and a geometry suballocator that makes add/remove touch only the changed mesh.

## Goals / Non-Goals

**Goals:**
- An execution-mode axis (`megakernel` | `wavefront`) orthogonal to the integrator
  axis, switchable at runtime, persisted, with front-end parity and accumulation
  reset on switch.
- Wavefront backend covering both `path` and `bdpt`.
- Adding a material under wavefront compiles exactly one small shade/eval kernel —
  never the megakernel.
- Incremental geometry add/remove: touch only the changed mesh's slab, stable
  offsets, no whole-scene re-upload.
- Tiled / streaming wavefront so path-state memory is bounded, not O(pixels).
- Mega-vs-wavefront image parity, verified headlessly per integrator.
- Reuse one material/integrator code body across both modes (shared Slang
  includes), so "switchable" is two emitters over one source, not two renderers.

**Non-Goals:**
- Removing the megakernel. It stays and is the default; it wins on simple scenes
  and is the only Metal path.
- General bidirectional bdpt. Wavefront bdpt matches the current "flat first-hit
  only" scope (`main_pass.slang:127`), not all-strategy bdpt.
- GPU-side material radix sort. Wavefront starts with atomic per-material append
  queues; a sort/compaction pass is a future optimization.
- Wavefront on Metal. Vulkan-first, like the GPU skinning/refit pipelines; Metal
  is pinned to megakernel.
- Reworking the integrators' math (MIS, light sampling, RR, BSSRDF). The same
  estimator code is relocated across dispatches, not rewritten.

## Decisions

**1. Execution mode is a new axis, not a third integrator.**
The integrator is *what* (algorithm: path/bdpt); execution mode is *how*
(megakernel vs wavefront). They are orthogonal. Add `self.execution_mode` to the
renderer, select it like `backend`, include it in `_current_state_hash()` (switch
resets accumulation), persist `execution_mode` in `settings.json`, and expose it
in `ALL_PARAMS` + GLFW + Qt + debug viewport. *Alternative:* fold wavefront in as
`integratorType == 2`. Rejected — wavefront is multi-dispatch with persistent VRAM
state; it is not expressible as a per-pixel branch inside the megakernel.

**2. Capability matrix gates the UI.**

```
            megakernel   wavefront
  path          ✓            ✓   (Phase 1)
  bdpt          ✓            ✓   (Phase 3)
```

Until Phase 3 lands, selecting `bdpt` + `wavefront` falls back to megakernel-bdpt
(and the UI surfaces that), rather than rendering wrong. On Metal, wavefront is
unavailable and the toggle is disabled/pinned to megakernel.

**3. Shared front-end; two codegen emitters over one `GraphFragment` source.**
Both modes share BVH + geometry buffers, TLAS/instances, lights, env CDF, camera,
accumulation image, and uniforms. The per-graph `GraphFragment` Slang is the
shared material source. `_emit_generated_materials` grows a second emitter:
- *megakernel emitter* (current): stitch fragments into the `evalSceneGraph`
  switch, one SPIR-V.
- *wavefront emitter* (new): wrap each fragment in its own compute entry
  `shadeMaterial_X` **and** an `evalBSDF_X` function (BSDF value, not sampling)
  for bdpt connection events. One module/pipeline per graph.
*Alternative:* duplicate material logic per mode. Rejected — the emitters must
draw from the same fragment so material features stay at parity for free.

**4. Wavefront path pipeline = staged dispatches over GPU path state + queues.**
Stages, each its own compute pipeline (managed by a new `vk_wavefront.py`
`WavefrontPasses`, modeled on `SkinningPasses`):
- *generate*: seed camera rays for the current path stream → ray queue.
- *intersect*: traverse the shared BVH for queued rays (reuse the megakernel's
  traversal include) → hit buffer; append each hit to its material's queue.
- *logic*: per hit, light sampling + MIS + Russian roulette + next-bounce ray
  generation; terminated lanes write radiance to the accumulation image and free
  their slot.
- *shade ×N*: one dispatch per material over that material's queue, evaluating
  `shadeMaterial_X` (BSDF sample + throughput update). This is the per-material
  pipeline set.
Path state is SoA in VRAM (~80 B/path: ray O/D, throughput, radiance, rng, depth,
pixelIdx, MIS pdf, flags). The bounce loop iterates intersect→logic→shade until
queues drain or max depth, then advances the stream.

**5. Tiled / streaming, fixed-size path streams with refill.**
Process a fixed number of paths at a time (a stream, e.g. `1<<20`) rather than one
slot per pixel (~160 MB at 1080p). As lanes terminate, refill the stream from the
remaining pixels/samples (stream compaction) to keep occupancy high. This bounds
VRAM and fits progressive accumulation naturally (one sample/pixel/frame, tiled
internally). *Alternative:* full-frame path-state buffers. Rejected — memory risk
on high resolution / shared GPUs.

**6. Atomic per-material append queues first; indirect dispatch sizes them.**
During intersect/logic, append each surviving hit to its material's queue via an
atomic counter (matId → queue). Each shade dispatch is sized to its queue count
via `vkCmdDispatchIndirect`, fed by a tiny *build-indirect-args* kernel that turns
queue counts into dispatch dims. *Alternative:* GPU radix sort + contiguous
slices (best coherence). Deferred — atomic queues are simpler and correct; sort is
a later optimization. De-risk indirect dispatch by first landing a conservative
direct-dispatch fallback (worst-case groups + early-out on empty lanes) and
A/B-ing the two.

**7. Vulkan compute layer grows multi-pipeline + indirect dispatch.**
`vk_compute.py` gains the ability to own N pipelines/descriptor sets and to record
`vkCmdDispatchIndirect`, following the `vk_skinning.py` pattern (separate modules,
layouts, sets, own command buffer). The wavefront passes run before/with the
frame render like skinning does.

**8. bdpt wavefront = two subpath walks + a connection stage, limited scope.**
Phase 3. Build a camera subpath wavefront and a light subpath wavefront, each a
generate→intersect→shade walk that **stores its vertices** (pos, normal,
throughput, fwd/rev pdf, matId) up to max depth. A connection+shadow stage
connects prefix pairs: visibility ray + `evalBSDF_X` at both endpoints + geometry
term + MIS recurrence. Vertex storage is bounded by `stream_size × maxdepth × 2`,
not pixels (consistent with Decision 5). Match the current flat-first-hit bdpt
scope, not general bdpt. *Alternative:* general all-strategy bdpt. Rejected for
this change — far larger; current megakernel bdpt is already limited.

**9. Geometry suballocator with stable offsets + free-list (mode-independent).**
Replace `_upload_meshes_concatenated` with a slab allocator over the shared
vertex/index/BVH buffers: each baked mesh gets a slab whose offsets are **stable**
for its lifetime, so the BLAS offsets in TLAS instance records stay valid without
reindexing. Add = bake (content-hash cached) + write one slab + add instance
record(s). Remove = free the slab (free-list) + drop instance record(s); no
re-concatenation. `_ensure_mesh_buffer_capacity` becomes a slab-aware grow (grow
preserves existing slab layout, so offsets survive). Fragmentation is handled by
the free-list with best-fit reuse and an occasional opt-in **compaction** that
moves slabs and rewrites the affected TLAS offsets. Both backends read these
buffers, so the suballocator is shared. *Alternative:* keep monolithic concat.
Rejected — it is the whole of cost (2).

**10. Headless A/B parity is the safety net for "switchable".**
Two execution modes plus two emitters risk drift. Gate with the existing
`tests/test_headless.py` harness: render the same scene in megakernel and
wavefront for each supported integrator and assert image equivalence within
tolerance. This makes the maintenance tax bounded and visible.

## Risks / Trade-offs

- **bdpt-in-wavefront is the hard core.** Variable-length subpath vertex storage,
  the connection MIS recurrence split across dispatches, and the dual-context use
  of per-material kernels (shade *and* connection eval) are the biggest risk. →
  Phase it last; match the limited flat-first-hit scope; fall back to
  megakernel-bdpt until it reaches A/B parity.
- **The path loop must be torn apart.** The megakernel's in-register
  generate→trace→shade→bounce loop becomes dispatches round-tripping path state
  through VRAM — more bandwidth, and MIS/light-sampling logic relocates into the
  logic stage. → Reuse the same estimator includes so it is relocation, not a
  rewrite; verify with A/B.
- **Memory under tiling.** Path state (~80 B/path) and bdpt vertex storage
  (×maxdepth ×2) are bounded by stream size, but a too-large stream still spikes
  VRAM. → Make stream size configurable with a conservative default; document the
  memory model.
- **Indirect dispatch is new to the codebase.** → Land the conservative
  direct-dispatch fallback first and A/B it against indirect.
- **Maintenance tax of two modes + two emitters.** → A/B parity CI gate
  (Decision 10); keep wavefront opt-in (megakernel default).
- **Metal gap.** Wavefront is Vulkan-only initially, so Metal users do not get the
  add-material-fast win for new graphs. → Acceptable and consistent with the
  Vulkan-only skinning/refit precedent; documented in the capability matrix.
- **Suballocator fragmentation / compaction correctness.** Moving slabs must
  rewrite every referencing TLAS offset atomically. → Compaction is opt-in and
  guarded by a unit test asserting offsets and rendered output are unchanged
  across a compaction.
- **Cost (2) and the wavefront work are orthogonal.** The suballocator is what
  actually makes geometry add/remove instant; wavefront only fixes the shader
  recompile. Both are bundled here so "add model = near-instant" is delivered
  end-to-end, but they are independent and can land/verify separately.

## Phase 1 — Detailed design (wavefront path tracer)

Pins the field-level contracts so §4–§6 (and the task-3 infra) are implementable
without guessing. All buffers use the project's scalar layout
(`-fvk-use-scalar-layout`: `float3` = 12 B, 4-byte aligned).

**P1-A. Path-state record — AoS first.** A `WavefrontPathState` StructuredBuffer,
one element per lane in the active stream. Fields (scalar layout):

| field | type | bytes | purpose |
|-------|------|-------|---------|
| `rayOrigin` | float3 | 0..12 | current bounce ray origin |
| `rayDir` | float3 | 12..24 | current bounce ray direction |
| `throughput` | float3 | 24..36 | path throughput β |
| `radiance` | float3 | 36..48 | accumulated L for this path's pixel |
| `pixelIndex` | uint | 48..52 | flattened target pixel (accumulation) |
| `rngState` | uint | 52..56 | PCG-32 state (matches `common.slang` `Rng.state`) |
| `depth` | uint | 56..60 | current bounce count |
| `flags` | uint | 60..64 | bit0 alive, bit1 specularBounce (skip NEE-MIS) |
| `bsdfPdf` | float | 64..68 | pdf of last BSDF sample (MIS vs NEE) |

Stride 68 B (alignment 4). *Deviation from Decision 5's "SoA":* start AoS — simpler
to get correct and matches the existing `pack()` discipline; convert hot fields to
SoA only if profiling shows the shade/intersect stages bandwidth-bound. The size is
locked by a `test_struct_layout`-style test against the Slang struct.

**P1-B. Queues + counting-sort by material.**
- `rayQueue`: `uint` index buffer (size = streamSize) + `rayCount` counter — lanes
  to intersect this bounce.
- `hitBuffer`: one `HitData` per lane (intersect output, `common.slang` struct).
- Material sort (counting sort, not a comparison sort): intersect atomically
  increments `materialCount[mid]` per hit; a `build_args` kernel prefix-sums
  `materialCount` → `materialOffset` and fills the indirect-args; a scatter writes
  each surviving lane into `materialQueue[materialOffset[mid] + local]`. *MVP
  fallback if the scatter is fiddly:* each `shade_<mid>` masks the hitBuffer by
  materialId (wastes lanes; `log()` the choice). Target is the counting sort.

**P1-C. Indirect dispatch (un-defers task 3.2).** `indirectArgs`: `numMaterials ×
(uint x,y,z)`. `build_args` sets `x = ceil(materialCount[mid] / GROUP)`, `y=z=1`.
Each `shade_<mid>` is `vkCmdDispatchIndirect(indirectArgs, offset = mid*12)`. Land
the conservative direct-dispatch fallback (3.3) first — dispatch worst-case
`ceil(streamSize/GROUP)` groups with an early-out on `local >= materialCount[mid]` —
then A/B it against indirect.

**P1-D. Stage kernel I/O contracts.**
- `generate(streamBase)` → for lane `i`: pixel = `pixelForLane(streamBase+i)`,
  seed state (camera ray via existing `cameras/{pinhole,thick_lens}`, β=1, L=0,
  `rngState = seed(pixel, frame)`, depth=0, alive); append `i` to `rayQueue`.
- `intersect` → traverse the shared BVH (reuse the `mesh_head` traversal include)
  per `rayQueue` lane → `hitBuffer[i]`. Miss: `radiance += β · env`, clear alive.
  Hit: bump `materialCount[hit.materialId]`.
- `shade_<mid>` (**per-material — this pins the deferred task 2.2 entry**): seed a
  `StdSurfaceParams` from the flat constants, overlay the graph via the 2.2
  emitter's `evalGraphSurface_<mid>(P,N,T,UV, params, inout sp)`, build the BSDF,
  then (a) NEE: sample a light (`lights/*`), eval BSDF, MIS-combine into `radiance`;
  (b) sample the BSDF for continuation → update `rayDir`, `throughput`, `bsdfPdf`,
  set specular flag; `depth++`.
- `logic` → Russian roulette on `throughput`; flush terminated lanes' `radiance`
  to the accumulation image at `pixelIndex`; compact survivors back into `rayQueue`
  for the next bounce.

**P1-E. Per-frame bounce loop (one sample/pixel, tiled).**
```
for streamBase in 0, streamSize, 2·streamSize, … < numPixels:   # P1 streaming
    generate(streamBase)
    for depth in 0 .. maxDepth-1:
        intersect                       # rayQueue → hits + material counts
        build_args                      # prefix-sum counts → offsets + indirect dims
        for mid in materials: shade_mid # indirect-dispatched, per-material
        logic                           # RR · flush terminated L · compact survivors
        if rayCount == 0: break
    # accumulation step folds this stream's per-pixel radiance into the running mean
```

**P1-F. Descriptor layout (un-defers task 3.1).** A new `vk_wavefront.py`
`WavefrontPasses` owns the stage pipelines + these buffers: `WavefrontPathState`,
`rayQueue`+counter, `hitBuffer`, `materialQueue`+`materialCount`+`materialOffset`,
`indirectArgs`; plus read access to the shared geometry/BVH/instances/lights/env/
uniform bindings (same as the megakernel) and write to the accumulation image. Each
stage pipeline binds the subset it needs — modelled on `SkinningPasses`. With these
buffers fixed, task 3's N-pipeline + indirect-dispatch infra is now concrete.

**P1-G. bdpt addendum (P3, out of P1 scope).** Adds a `lightPathState` vertex buffer
+ a connection+shadow stage; the connection endpoints reuse `shade_<mid>`'s BSDF
evaluation as the deferred `evalBSDF_<mid>` entry. Pinned here only to confirm the
P1 path-state and per-material-kernel shapes don't preclude it.
