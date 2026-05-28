## Context

`usd-animation-playback` established the playback clock, the load-time
`AnimationIndex`, the per-frame `_apply_animation_frame` re-eval hook, and the
accumulation-via-state-hash contract. It deliberately excluded skeletal
animation. UsdSkel assets animate by skinning: a `Skeleton` defines joints +
bind transforms, a `SkelAnimation` supplies time-sampled joint transforms, and
each skinned `Mesh` carries `UsdSkelBindingAPI` weights (`jointIndices`,
`jointWeights`, `geomBindTransform`). The renderer bakes meshes once to
object-space vertex/index/BVH buffers (32-byte vertex stride
`[pos|u|normal|v]`); per-instance world transforms live in the TLAS
`instance_buffer`. The BVH is a pure-Python median-split builder (leaf size 4,
depth-first array, node = `[min|leftOrCount|max|rightOrFirst]`) with **no refit**
path. The `vertex_buffer`/`bvh_buffer` are `StorageBuffer`s created
`STORAGE | TRANSFER_DST` (compute-writable) but currently bound read-only in
`main_pass`. The renderer already dispatches an auxiliary BXDF-grid compute pass,
so multiple `ComputePipeline`s are supported.

Target asset `SoC-ElephantWithMonochord.usdc`: 2 SkelRoots (27-joint Elefant /
1312 verts, 2-joint Monochord / 62 verts), 4 influences per vertex, **no blend
shapes**, 3023 joint samples over 0→3024 @ 60 fps.

## Goals / Non-Goals

**Goals:**
- Skinned USD meshes deform per frame and play through the existing transport.
- GPU linear-blend skinning into the BLAS vertex buffer; no per-frame CPU vertex
  work and no GPU→CPU readback.
- GPU BVH refit keeping the path tracer correct over deformed geometry.
- Correct, reusable joint-matrix assembly leveraging pxr's LBS math.

**Non-Goals:**
- Blend shapes (asset has none).
- GPU-side joint-matrix computation (CPU/pxr is cheap at these joint counts).
- Dual-quaternion skinning; motion blur.
- Atomic/parallel inner-node refit (serial reverse-order pass suffices now).
- Guaranteed Metal parity in this change (Vulkan-first; Metal is a follow-up if
  the backend's known compute issues obstruct).

## Decisions

### D1: Hybrid skinning — CPU joint matrices, GPU per-vertex blend
Each animated frame, the CPU calls pxr `SkeletonQuery.ComputeSkinningTransforms`
(or equivalent) to get per-joint skinning matrices, then folds in the up-axis
correction and the SkelRoot world transform, producing a small array of
skel→world matrices (≤ a few dozen). These upload to a joint-matrix SSBO. The
GPU skinning shader does the per-vertex blend. This reuses pxr's correct
influence/geomBindTransform handling without reimplementing LBS, while keeping
the O(vertices) work parallel.

- *Alternative (full GPU, joint matrices on GPU)*: would require porting pxr's
  joint-resolution + bind-inverse math to a shader; high risk, no benefit at
  these joint counts. Rejected.

### D2: Skinning writes deformed verts in the authored-points space; the loader's TLAS transform is kept
**Revised during implementation.** Empirically, pxr's `ComputeSkinnedPoints`
returns deformed points in the **same space as the mesh's authored rest points**
(here geomBindTransform is identity), and my `lbs_points` reproduces it exactly.
So the skinning pass writes deformed positions/normals in that authored-points
space into the BLAS, and the skinned instance **keeps the TLAS transform the
loader already assigned** — which carries the prim placement and the up-axis
correction. No identity-TLAS, no world/up-axis folding into the joint matrices.
The BVH is built/refit in that same space; the existing instance transform
places everything at trace time, exactly as it does for the bind-pose load.

- *Original plan (world-space verts + identity TLAS, world folded into joint
  matrices)*: rejected after verifying the simpler reuse is correct for this
  asset and far lower risk. (For assets where skeleton space ≠ authored-points
  space — geomBind ≠ identity with a distinct skel placement — a per-skeleton
  skel→world transform on the instance would be needed; noted as a future
  generalization.)

### D3: GPU BVH refit reusing bind-pose topology
The bind-pose BVH is built once (existing Python builder). Refit runs two GPU
phases per frame: (1) one thread per leaf recomputes its AABB from the deformed
triangles it references; (2) a serial pass over inner nodes in **reverse array
order** sets each to the union of its two children. Reverse order is valid
because the depth-first preorder build guarantees a parent's index precedes both
children, so children are refit before their parent. Correct and simple;
quality degrades only under extreme deformation, acceptable for skinning.

- *Alternative (atomic bottom-up, Karras)*: fully parallel inner propagation via
  per-node arrival flags; scales to huge rigs but is materially harder to get
  right. Deferred as a perf follow-up.
- *Alternative (CPU rebuild via readback)*: simple BVH code but per-frame
  GPU→CPU readback + Python rebuild; only viable for tiny meshes. Used only as a
  transient implementation checkpoint (see Migration), not the shipped path.

### D4: Separate compute pipelines with their own descriptor sets
`skin.slang` and `bvh_refit.slang` are new `ComputePipeline`s with dedicated
descriptor sets binding rest verts, influences, joint matrices, and RW
`vertex_buffer`/`bvh_buffer`. `main_pass`'s descriptor layout is untouched, so
the documented Metal binding-count pressure is isolated to the new, small sets.
Frame command recording inserts memory barriers: skin → refit → main_pass.

### D5: Reuse the playback clock + transport from usd-animation-playback
`AnimationIndex` gains `skinned_mesh_paths` (and the skeletal contribution to
`has_animation`). No new transport UI — the existing Animation section appears
once `has_animation` is true. `_apply_animation_frame` gains a skeletal branch
that uploads joint matrices and enqueues the skinning + refit dispatches when the
time code changed. `current_time_code` is already in `_current_state_hash`, so
accumulation resets correctly while skinning plays.

## Risks / Trade-offs

- **GPU BVH correctness over deformed geometry** → Refit keeps topology; verify
  via headless A/B that the deformed mesh both *changes* and *renders without
  holes/artifacts* at multiple time codes. The transient CPU-rebuild checkpoint
  (D3) isolates skinning bugs from refit bugs.
- **Refit quality under large deformation** → Acceptable for skinning; an atomic
  rebuild/refit is the escape hatch if artifacts appear.
- **Metal backend fragility** (documented compute/cursor hang issues) → Vulkan is
  authoritative; Metal parity attempted but flagged as follow-up rather than
  blocking.
- **Coordinate-space errors** (the highest-likelihood bug) → Validate the folded
  joint matrices by comparing the first-frame GPU-skinned result against pxr's
  `UsdSkelBakeSkinning`/`ComputeSkinnedPoints` reference on CPU in a unit test.
- **Descriptor binding pressure** → Isolated to new pipelines (D4); no main_pass
  impact.

## Migration Plan

1. Land load-time skel detection + rest/influence buffers + bind-pose bake +
   index/transport wiring (renders static bind pose; transport visible).
2. Land GPU skinning verified against a **temporary CPU BVH rebuild** (proves
   skinning/space correctness with a known-good BVH).
3. Replace the temporary CPU rebuild with the GPU refit pass.
4. Headless A/B + the CPU-reference unit test gate each step. Rollback = revert
   the change; `usd-animation-playback` is unaffected.

## Open Questions

- Should skinned normals be skinned in-shader (Σ wᵢ Mᵢ n, upper-3×3) or
  recomputed from deformed positions? (Lean: skin in-shader — cheaper, smoother,
  matches LBS.)
- Joint-matrix buffer as UBO vs SSBO? (Lean: SSBO — variable joint counts,
  matches existing storage-buffer plumbing.)
