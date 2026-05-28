## Why

The `usd-animation-playback` change added playback for cheap time-sampled prims
(xform/camera/lights) but explicitly excluded skeletal animation. Many authored
USD assets (e.g. `SoC-ElephantWithMonochord.usdc`) animate **only** through
UsdSkel skinning, so they currently load frozen in bind pose and the animation
transport stays hidden. This change deforms skinned meshes per frame so those
assets actually animate.

## What Changes

- At load, detect **UsdSkel** bindings (SkelRoot → Skeleton + SkelAnimation +
  skinned Meshes via `UsdSkelBindingAPI`). Bake each skinned mesh's bind-pose
  geometry as its BLAS and record it as a skinned instance.
- Build static GPU buffers per skinned mesh: rest positions+normals and
  per-vertex `jointIndices[4]` + `jointWeights[4]`.
- Per animated frame: compute per-joint skinning matrices on the **CPU via pxr**
  (`SkeletonQuery.ComputeSkinningTransforms`), folding the up-axis correction and
  SkelRoot world transform in, and upload them as a small joint-matrix buffer.
- Add a **GPU skinning compute pass** (`skin.slang`) that linear-blend-skins
  rest verts → **world-space** deformed positions+normals, writing into the
  existing `vertex_buffer` BLAS slot.
- Add a **GPU BVH refit compute pass** (`bvh_refit.slang`) that keeps the
  bind-pose tree topology and recomputes node AABBs from the deformed verts
  (parallel leaves + serial reverse-order inner propagation).
- Skinned instances use an **identity TLAS transform** (world-space geometry);
  the bind-pose load bake means an unplayed scene still renders correctly.
- Extend the `usd-animation-playback` animation index so a skinned stage reports
  `has_animation`, surfacing the existing transport + driving the existing clock
  and accumulation reset.

Out of scope: UsdSkel **blend shapes** (the target asset has none); GPU-side
joint-matrix computation; dual-quaternion skinning; an atomic/parallel
inner-node refit (the serial reverse-order pass is sufficient at current rig
sizes).

## Capabilities

### New Capabilities
- `usd-skeletal-anim`: UsdSkel detection + bind-pose bake, GPU linear-blend
  skinning into the BLAS vertex buffer, GPU BVH refit over deformed geometry,
  and surfacing skinned stages through the existing playback
  clock/transport/accumulation (the `has_animation` contract gains a skeletal
  category, specified here as added behavior rather than a delta since
  `usd-animation-playback` is not yet archived).

### Modified Capabilities
<!-- None: usd-animation-playback is not yet archived, so its index extension is
     specified as ADDED requirements in this change's spec instead of a delta. -->


## Impact

- **Code**: `src/skinny/usd_loader.py` (skel binding detection, rest-vertex +
  influence extraction, bind-pose bake, extend `AnimationIndex`);
  `src/skinny/renderer.py` (skinned-instance state, per-frame joint-matrix
  upload, skinning + refit dispatch with barriers, skeletal branch in
  `_apply_animation_frame`); `src/skinny/mesh.py` (expose bind-pose BVH
  topology / rest buffers as needed).
- **Shaders**: new `src/skinny/shaders/skin.slang` and `bvh_refit.slang`
  (compute); `main_pass.spv` unaffected, but both new shaders compile to SPIR-V
  via `slangc`. `vertex_buffer` + `bvh_buffer` gain RW bindings **in the new
  pipelines only** (own descriptor sets; main_pass bindings untouched).
- **Backends**: Vulkan first (verified headless). Metal parity is a risk given
  documented compute/descriptor hang issues; flagged as follow-up if it
  obstructs, not a blocker.
- **Dependencies**: none new — pxr UsdSkel is already available.
