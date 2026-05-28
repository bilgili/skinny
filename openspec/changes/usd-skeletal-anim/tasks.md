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
- [ ] 3.2 Upload joint matrices to a GPU joint-matrix SSBO
- [x] 3.3 Unit test: applying the matrices to rest verts via LBS matches pxr's reference skinned points within tolerance (no GPU)

## 4. GPU skinning pass

- [ ] 4.1 Add `shaders/skin.slang` compute: rest pos/normal + influences + joint matrices → world-space deformed pos/normal into the BLAS vertex-buffer slot; skin normals in-shader (upper-3×3)
- [ ] 4.2 New `ComputePipeline` for skinning with its own descriptor set (rest verts, influences, joint matrices, RW vertex_buffer); compile to SPIR-V via slangc
- [x] 4.3 Interim CPU skinning: per frame, CPU LBS (validated lbs_points) → rebuild each skinned BLAS via bake_mesh → re-upload. Proves skinning/space against a known-good tree
- [x] 4.4 Headless A/B: ElephantWithMonochord deforms between t=1 and t=1500 (tests/test_headless_skel.py)

## 5. GPU BVH refit pass

- [ ] 5.1 Add `shaders/bvh_refit.slang`: parallel leaf-AABB recompute from deformed triangles + serial reverse-array-order inner-node union; bind `bvh_buffer` + `vertex_buffer` RW
- [ ] 5.2 New `ComputePipeline` for refit with its own descriptor set
- [ ] 5.3 Order the frame as skin → refit → main_pass with GPU memory barriers; remove the interim CPU rebuild
- [ ] 5.4 Verify refitted AABBs contain deformed geometry and the render shows no holes/stretch-through

## 6. Playback integration

- [x] 6.1 Skeletal branch added to _apply_animation_frame (_apply_skeletal_frame; CPU interim — GPU dispatch swaps in at Group 4/5)
- [x] 6.2 Existing transport drives skinned playback; current_time_code already resets accumulation (paused converges)

## 7. Verification

- [ ] 7.1 `ruff check src/` introduces no new errors; new files clean; `pytest -m "not gpu"` green
- [x] 7.2 Headless A/B test committed (tests/test_headless_skel.py), green in the built 3.13 venv
- [ ] 7.3 Manual (Vulkan): load `SoC-ElephantWithMonochord.usdc`, play/scrub, confirm the elephant animates and paused frames converge
- [ ] 7.4 Metal: attempt parity; if the backend's known compute/descriptor issues obstruct, document and defer (Vulkan authoritative)
