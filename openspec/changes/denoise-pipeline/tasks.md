## 1. Spike — record the MetalFX contract before any renderer edit

- [ ] 1.1 Install PyObjC into the repo-root Python 3.13 environment. Record which
      package supplies MetalFX: `pyobjc-framework-MetalFX`, or `pyobjc-core` plus
      `objc.loadBundle` on `MetalFX.framework`. Record the version.
- [ ] 1.2 Write `tools/spike_metalfx.py`. Build a headless `MetalContext`, read
      `device.native_handles`, and wrap the `MTLDevice` pointer with
      `objc.objc_object(c_void_p=…)`. Prove the wrapper answers a device query.
      Use the managed-scope teardown rule; never kill the process.
- [ ] 1.3 In the spike, create one `MTLFXTemporalDenoisedScalerDescriptor` and
      one scaler. Record the exact required input properties, the optional ones,
      the accepted texture formats per input, and the accepted input-to-output
      size ratios. Record whether an input extent equal to the output extent is
      accepted.
- [ ] 1.4 In the spike, allocate slang-rhi textures, read `Texture.native_handle`,
      and denoise one synthetic noisy frame with synthetic auxiliary images.
      Write the input and output images to PNG. Confirm the output differs from
      the input and holds no NaN.
- [ ] 1.5 Record the exposure contract: whether MetalFX expects a pre-exposed
      image or a linear image plus an exposure texture.
- [ ] 1.6 Check in `openspec/changes/denoise-pipeline/metalfx_contract.md` with
      every recorded fact from 1.1 to 1.5. Every later task reads this file.
- [ ] 1.7 Verify the GPU is healthy after the spike: build a fresh
      `MetalContext` and complete a trivial dispatch.

## 2. The seam — `denoise.py`, hostless

- [ ] 2.1 Create `src/skinny/denoise.py`. Declare the `Denoiser` protocol
      (`name`, `required_aovs`, `resize`, `reset`, `denoise`, `destroy`) and the
      per-frame record type. Import no GPU package at module scope.
- [ ] 2.2 Declare the auxiliary-image registry in `denoise.py`: name, pixel
      format, and contents for `diffuse_albedo`, `specular_albedo`,
      `normal_depth`, and `motion`. Match the formats to the record from 1.6.
- [ ] 2.3 Add `resolve_render_extent(output_w, output_h, scale)` — the one pure
      derivation of the render extent. Scale 1.0 SHALL return the output extent
      unchanged.
- [ ] 2.4 Add the name registry and `create_denoiser(name, ctx)`. Return `None`
      for `none`. Refuse a `required_aovs` entry that names no registered
      auxiliary image, naming the unknown image.
- [ ] 2.5 Add `NullDenoiser` — declares no auxiliary image and copies its input
      to its output. It proves the plumbing in groups 4 to 7 with no MetalFX and
      no new dependency.
- [ ] 2.6 Write `tests/test_denoise_seam.py`: registry contents, the extent
      resolver at scale 1.0 and at 0.5, unknown-auxiliary-image refusal, and a
      subprocess import with `vulkan` and `slangpy` blocked at the meta path.
- [ ] 2.7 Add the vendor-name grep gate: no `MetalFX`, `MTLFX`, `NRD`, `OptiX`,
      or `OIDN` identifier outside `denoise.py` and the implementation modules.

## 3. Envelope, flags, and bring-up — hostless

- [ ] 3.1 Add `denoiser` and `denoise_scale` to `EnvelopeQuery`. Add the codes
      `DENOISER_BACKEND_UNSUPPORTED`, `DENOISER_EXTRA_MISSING`, and
      `DENOISE_SCALE_WITHOUT_DENOISER` to `ALL_CODES` in canonical order, and
      write their rules in `evaluate`.
- [ ] 3.2 Assign each new code an owner: a CLI guard set in `CLI_GUARD_CODES`, or
      `CLI_UNOWNED_CODES` with a recorded reason. Keep the partition test green.
- [ ] 3.3 Add `--denoiser` and `--denoise-scale` to the shared flag definition in
      `cli_common.py`, with the `SKINNY_DENOISER` and `SKINNY_DENOISE_SCALE`
      environment fallbacks. Neither is persisted.
- [ ] 3.4 Add `reject_denoiser_flags(args)` — the flag-level guard. Refuse an
      unknown name, a scale outside 0.25 to 1.0, a scale other than 1.0 with no
      denoiser, and a missing optional dependency with the install command in the
      message. Print the fixed `skinny:` prefix.
- [ ] 3.5 Add `reject_denoiser_backend(denoiser, backend)` — the guard that needs
      the resolved backend.
- [ ] 3.6 Wire both guards into `plan_bringup`: the flag guard next to
      `reject_mcp_unsupported`, the backend guard immediately after
      `select_backend` and before the plan is returned. Add `denoiser` and
      `denoise_scale` to `BringupPlan`.
- [ ] 3.7 Extend `tests/test_bringup.py`: the canonical order still matches both
      transcribed legacy sequences, the refusal strings are pinned verbatim, and
      the backend guard refuses before any context is constructed.
- [ ] 3.8 Confirm `parity.combo_is_valid` sets no denoiser and that the matrix
      enumeration is byte-identical before and after group 3.

## 4. Resolution split and GPU resources

- [ ] 4.1 Add `render_width` and `render_height` to the `gpu_resources` sizes
      record. Derive them with `resolve_render_extent`.
- [ ] 4.2 Mark each existing `ResourceDecl` with the extent it sizes on. Render
      extent: accumulation image, light-splat buffer, wavefront record buffers,
      ReSTIR reservoirs. Output extent: display image, HUD overlay, swapchain.
- [ ] 4.3 Declare the denoised image (output extent, RGBA32F) and the four
      auxiliary images (render extent) as `ResourceDecl` entries, each allocated
      only when the active denoiser declares it. Assign Vulkan bindings 58 to 62
      and the matching Metal shader-global names.
- [ ] 4.4 Count the Metal compute-argument texture slots for the auxiliary pass
      with the 119-slot bindless pool bound. Confirm the total stays under 128
      before group 5 is written. If it does not, record the fold and adjust.
- [ ] 4.5 Re-capture `tests/fixtures/gpu_resource_inventory.json` from the live
      renderer on Vulkan RGB, Metal RGB, and Metal spectral. Capture reality;
      never hand-edit it to match the code.
- [ ] 4.6 Add `outputWidth`, `outputHeight`, `jitterOffset`, and `jitterMode` to
      the `FrameConstants` Slang declaration and to the `slang_layout` packer.
      Keep the build-gated tail fields at their recorded offsets. Update the
      pinned goldens.
- [ ] 4.7 Map an output pixel to a render pixel at the one tool-picking site. Add
      a test that pins the mapping at scale 1.0 and at 0.5.
- [ ] 4.8 Run the hostless suite: `.venv/bin/pytest -m "not gpu"`. Confirm the
      accumulation image is bit-identical with no denoiser.

## 5. Auxiliary-image pass

- [ ] 5.1 Write `src/skinny/shaders/gbuffer.slang` with entry `computeMain`. Trace
      one primary ray per pixel through the existing scene-trace module. Write
      only the auxiliary images the build enables.
- [ ] 5.2 Produce diffuse albedo, specular albedo, roughness, world normal, and
      linear view depth from the first-hit material. For a skin, graph, or
      Python material, write the base colour as an approximation and record the
      limit.
- [ ] 5.3 Produce the motion vector by reprojecting the first-hit world position
      with the previous frame's view-projection matrix. Store the previous
      matrix on the renderer and update it once per frame.
- [ ] 5.4 Add the `GBUFFER` family to `shader_variants.py`. Refuse the spectral,
      MLT, and neural axes at key construction. Update the permanent goldens in
      `tests/test_shader_variants.py`.
- [ ] 5.5 Build the pass on both backends through the existing `ComputePipeline`
      wrappers. Bind by name on Metal, by descriptor on Vulkan.
- [ ] 5.6 Dispatch the pass once per frame while a denoiser is active, before the
      denoiser call. Never dispatch it otherwise.
- [ ] 5.7 GPU check on Metal: render a known scene, read each auxiliary image,
      and confirm the normal is unit length, the depth is positive in front of
      the camera, the albedo is in the range 0 to 1, and the motion vector is
      zero for a static camera.
- [ ] 5.8 Confirm every pre-existing `.spv` file is byte-identical.

## 6. Display pass and output routing

- [ ] 6.1 Write `src/skinny/shaders/display_resolve.slang` with entry
      `computeMain`. Read the denoised image at output extent and call the
      existing `wfWriteDisplay` helper. Add no second tonemap definition.
- [ ] 6.2 Add the `DISPLAY_RESOLVE` family to `shader_variants.py` with the same
      axis refusals as 5.4. Update the goldens.
- [ ] 6.3 Dispatch the display pass after the denoiser, overwriting the display
      image. Leave the megakernel and wavefront display tails unedited.
- [ ] 6.4 Add `renderer.display_source()`. Return the denoised image while a
      denoiser runs, the accumulation image otherwise. Keep
      `read_accumulation()` returning the raw accumulation image always.
- [ ] 6.5 Route `render_headless()`, `save_screenshot()`, and the EXR and
      Radiance writers through `display_source()`.
- [ ] 6.6 Add the test that fails when an output path reads the accumulation
      image directly, naming the bypassing path.
- [ ] 6.7 Add the `denoise_enabled` and `denoise_strength` controls to `params.py`
      with `resets_accumulation=False`. Update the opt-out enumeration test to
      the four expected paths.
- [ ] 6.8 Call `denoiser.reset()` from the accumulation-reset site. Add a test
      that a camera move resets both and that toggling the denoiser resets
      neither.
- [ ] 6.9 GPU check on Metal with `NullDenoiser`: the display image with the null
      denoiser active is visually identical to the display image with no
      denoiser, at scale 1.0.

## 7. Reported jitter

- [ ] 7.1 Add the Halton (2,3) offset to the primary-ray generation in
      `cameras/pinhole.slang`, selected by `fc.jitterMode`. Keep the random
      offset as the other mode.
- [ ] 7.2 Compute the per-frame offset on the host, pack it into
      `FrameConstants`, and report the same value to the denoiser.
- [ ] 7.3 Confirm the accumulation image with no denoiser is bit-identical to the
      pre-change image.
- [ ] 7.4 Add a test that the reported offset for a frame equals the offset the
      packed uniform carried.

## 8. MetalFX implementation

- [ ] 8.1 Add the `[metalfx]` extra to `pyproject.toml`, using the package the
      spike recorded in 1.1.
- [ ] 8.2 Write `src/skinny/denoise_metalfx.py`. Import the Objective-C bridge
      lazily, so a host without the extra can still import `denoise.py`.
- [ ] 8.3 Build the scaler and one `MTLCommandQueue` from the shared `MTLDevice`
      handle. Declare `required_aovs` from the recorded contract in 1.6.
- [ ] 8.4 Implement `denoise()`: bind each texture through
      `Texture.native_handle`, set the jitter offset, the exposure, and the
      reset flag, encode, commit, and wait.
- [ ] 8.5 If the recorded contract excludes the accumulation image's format,
      declare an input image in an accepted format and add the one copy pass that
      populates it.
- [ ] 8.6 Implement `resize()` — rebuild the scaler on an extent change — and
      `reset()` — set the scaler's reset flag for the next frame.
- [ ] 8.7 Implement `destroy()`. Release the scaler and the queue. Make it
      idempotent. Register it in the renderer teardown list, before the device
      closes.
- [ ] 8.8 Refuse at construction a denoise scale outside the range the recorded
      contract permits, naming the permitted range.
- [ ] 8.9 Register `metalfx` in the name registry and mark it Metal-only.

## 9. Upscaling

- [ ] 9.1 Run `--denoiser metalfx --denoise-scale 0.5` on a Metal host. Confirm
      the accumulation image is at render extent and the display image is at
      output extent.
- [ ] 9.2 Confirm the HUD text, the gizmo, and tool picking are correct at output
      extent while the scale is 0.5.
- [ ] 9.3 Confirm `skinny-render --denoise-scale 0.5` writes a file at the
      requested output size.
- [ ] 9.4 Confirm the swapchain resize path rebuilds both extents and calls
      `resize()` on the denoiser.

## 10. Verification and gates

- [ ] 10.1 Run the hostless suite: `.venv/bin/pytest -m "not gpu"`. It must be
      green. A skip is not a pass — confirm the new tests ran.
- [ ] 10.2 Run `ruff check src/` with an explicit target, so the root `.gitignore`
      cannot make it a vacuous pass.
- [ ] 10.3 Run the Metal cleanup harness with a denoiser active:
      `PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest
      tests/test_metal_cleanup.py -m gpu -q`.
- [ ] 10.4 Run the GPU parity matrix gate. Every recorded baseline and every
      self-consistency tolerance must be unchanged, because no combination sets a
      denoiser.
- [ ] 10.5 Render the quality A/B: one scene at matched sample counts with no
      denoiser and with MetalFX, at scale 1.0 and at 0.5. Produce a labelled
      side-by-side at a shared exposure and tonemap. Report relMSE, PSNR, and
      FLIP against a converged reference from `metrics.compute_metrics`. Show the
      image.
- [ ] 10.6 Verify the GPU is healthy: a fresh `MetalContext` builds and a trivial
      dispatch completes.

## 11. Documentation

- [ ] 11.1 Write `docs/Denoising.md`: the seam, the auxiliary-image registry, the
      per-frame stage order, the resolution split, and the MetalFX
      implementation. Author the stage-order diagram as an SVG under
      `docs/diagrams/`.
- [ ] 11.2 Add the denoiser rows to the Compatibility matrix in `CLAUDE.md` and
      in `README.md` — backend, execution mode, and the refused combinations.
- [ ] 11.3 Add the new bindings 58 to 62 to the descriptor binding map in
      `docs/Architecture.md`, plus the two new shader families and the two new
      modules in the module map.
- [ ] 11.4 Document the `--denoiser` and `--denoise-scale` flags in `README.md`,
      and the `[metalfx]` extra in the install section.
- [ ] 11.5 Record the known limits: the focus-plane and furnace overlays are off
      while a denoiser runs, motion vectors are camera-only, and the auxiliary
      albedo is approximate for non-flat materials.
- [ ] 11.6 Add the `CHANGELOG.md` entry.

## 12. Review and merge

- [ ] 12.1 Run `openspec validate denoise-pipeline`.
- [ ] 12.2 Run the codex pre-merge review over the whole change. Fix or
      consciously dismiss every finding. If codex is unavailable, say so and run
      a review subagent instead.
- [ ] 12.3 Merge from the worktree and archive the change.
