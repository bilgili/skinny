## 1. UsdSkel detection + extraction (load time)

- [x] 1.1 In `usd_loader.py`, add UsdSkel binding discovery (UsdSkel.Cache → SkelRoot → skeleton, anim source, skinning targets); return per skinned mesh: rest points, rest normals, `jointIndices[4]`, `jointWeights[4]`, joint order, geomBindTransform
- [x] 1.2 Extend `AnimationIndex` with `skinned_mesh_paths` (+ skeletal contribution to `has_animation`)
- [x] 1.3 Unit test on the ElephantWithMonochord stage: 2 skinned meshes detected, 27/2 joints, 4 influences/vtx, `has_animation` true; a non-skel stage records none

## 2. Bind-pose bake + skinned-instance state

- [x] 2.1 Skinned meshes bake at authored bind pose via the normal loader path and keep the loader's TLAS transform (revised: deformed points share authored-points space → no identity-TLAS / world-fold; see design D2)
- [x] 2.2 Track skinned instances on the renderer: hold the `SkeletalScene` (keeps cache+stage alive); match instances to bindings by prim path (GPU rest/influence buffers deferred to Group 4)
- [x] 2.3 Confirm an unplayed skinned scene renders correct bind pose (loader bakes authored points; headless)

## 3. CPU joint matrices

- [x] 3.1 Per animated frame, compute per-joint skinning matrices via pxr at `current_time_code`; fold in up-axis correction + SkelRoot world transform → skel→world matrices
- [x] 3.2 Upload joint matrices to a per-mesh GPU SSBO each frame (_SkinnedMeshGPU.upload_joint_matrices)
- [x] 3.3 Unit test: applying the matrices to rest verts via LBS matches pxr's reference skinned points within tolerance (no GPU)

## 4. GPU skinning pass

- [x] 4.1 shaders/skin.slang compute written (LBS rest pos/normal + influences + joint matrices → deformed vertex; mul(M,v) matches the numpy upload convention); compiles to SPIR-V via slangc
- [x] 4.2 SkinningPasses: standalone skin ComputePipeline + own descriptor set (rest verts, jointIdx, jointWt, jointMats, RW vertex_buffer); slangc-compiled; main_pass untouched
- [x] 4.3 Interim CPU skinning: per frame, CPU LBS (validated lbs_points) → rebuild each skinned BLAS via bake_mesh → re-upload. Proves skinning/space against a known-good tree
- [x] 4.4 Headless A/B: ElephantWithMonochord deforms between t=1 and t=1500 (tests/test_headless_skel.py)

## 5. GPU BVH refit pass

- [x] 5.1 shaders/bvh_refit.slang written (single-thread reverse-order refit: leaf AABB from deformed tris, inner = union of children); compiles to SPIR-V
- [x] 5.2 Standalone refit ComputePipeline + own descriptor set (verts, indices, RW bvh_buffer)
- [x] 5.3 Isolated one-shot submit orders skin → refit with a buffer barrier, before the render reads the buffers (no edit to the shared render recording). CPU path retained only as the non-Vulkan fallback
- [x] 5.4 Verified by readback: GPU skinned positions match CPU lbs to 1.5e-8; refit root AABB exactly bounds the deformed verts (err 0.0)

## 6. Playback integration

- [x] 6.1 Skeletal branch added to _apply_animation_frame (_apply_skeletal_frame; CPU interim — GPU dispatch swaps in at Group 4/5)
- [x] 6.2 Existing transport drives skinned playback; current_time_code already resets accumulation (paused converges)

## 7. Verification

- [x] 7.1 ruff: new files clean, no new errors; pytest -m 'not gpu' green (173); headless skel green
- [x] 7.2 Headless A/B test committed (tests/test_headless_skel.py), green in the built 3.13 venv
- [ ] 7.3 Manual (Vulkan): load `SoC-ElephantWithMonochord.usdc`, play/scrub, confirm the elephant animates and paused frames converge
- [x] 7.4 Metal parity deferred by design: GPU passes are Vulkan-only (raw vk.*); Metal backend uses the wired CPU skinning fallback. A Metal (slang-rhi) compute port is a future change
