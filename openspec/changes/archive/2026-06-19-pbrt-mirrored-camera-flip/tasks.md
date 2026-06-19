> Status (worktree `pbrt-mirrored-camera-flip`): **COMPLETE (21/21), UNCOMMITTED.**
> Finishes the last renderer-side follow-up from `pbrt-v4-scene-import`
> ("mirrored-camera flip support"). Importer detection was already in main
> (`pbrt/api.py:263`, commit `b40893b`). Renderer now honors the flag end-to-end:
> loader → `CameraOverride.mirrored` → `FrameConstants.cameraMirror` →
> `zoomedNDC` ndc.x negate (+ `sampleWi` for BDPT). GPU-verified mirror==fliplr on
> Metal **and** Vulkan; new `mirrored_arealight` pbrt parity scene GREEN at relMSE
> 0.009 / FLIP 0.021 (mirror-off fails at 0.70, proving the flip earns parity).
> Also fixed a latent Vulkan UBO silent-truncation bug (512 B pin → dropped any
> scalar-tail field past 512 B; now `_VK_UNIFORM_BUFFER_BYTES` 768 + import guard).
> ruff clean; touched-area host tests green; 10 pre-existing baseline failures
> (materialx API drift, torch drift, online-training trigger) confirmed identical
> on main. `openspec validate --strict` valid.

## 1. Host — thread the `mirrored` flag (failing tests first)

- [x] 1.1 Write a failing unit test (`tests/pbrt/test_camera.py` or
      `tests/test_usd_loader.py`): an imported USD camera carrying
      `customData["pbrt"]["mirrored"] = True` loads to a `CameraOverride` with
      `mirrored == True`; absent/false → `False`
- [x] 1.2 Add `mirrored: bool = False` to `CameraOverride` (`scene.py`); document it
      in the dataclass docstring (screen-space horizontal mirror of an improper
      pbrt camera; does not change position/forward/basis)
- [x] 1.3 In `usd_loader._extract_camera`, read `prim.GetCustomDataByKey("pbrt")`
      → `mirrored` (truthy) and pass it to the `CameraOverride(...)` ctor — 1.1 green

## 2. Host — carry into the renderer + UBO

- [x] 2.1 Write a failing test asserting `_pack_uniforms()` length is unchanged
      except for one new `uint`, and the `_FC_FIELDS` cumulative-size drift guard
      still equals `len(_pack_uniforms())` (catches byte desync before the field is
      added)
- [x] 2.2 Store the flag as renderer host state (e.g. `self._camera_mirror`), set
      from `CameraOverride.mirrored` in `_apply_camera_override` /
      `_override_to_orbit` consumers; default `False`
- [x] 2.3 Append `uint cameraMirror` to `FrameConstants` (`common.slang`) after
      `recordMode`; append it in `_pack_uniforms`; add `("cameraMirror", 4)` to
      `_FC_FIELDS` — 2.1 green (Vulkan + Metal MSL packers in lockstep)
- [x] 2.4 Add `cameraMirror` to `_current_state_hash()` so a mirror-state change
      resets accumulation

## 3. Shader — apply the flip

- [x] 3.1 In `zoomedNDC` (`common.slang`): `if (fc.cameraMirror != 0) ndc.x = -ndc.x;`
      (after the existing `ndc.y` flip) — covers pinhole + thick_lens primary rays,
      megakernel + wavefront
- [x] 3.2 In `PinholeCamera.sampleWi` and `ThickLensCamera.sampleWi`: negate
      `ndc.x` under the same flag before the `xs`/pixel mapping, so BDPT / light-
      tracing camera connections agree with the mirrored primary frame
- [x] 3.3 Recompile `main_pass.spv` (`slangc`) and bust the wavefront `.spv` cache;
      confirm both megakernel and wavefront NEE/path routes pick up the rebuilt
      shaders in the worktree (isolated from main's cache)

## 4. Verification — host A/B (no pbrt binary needed)

- [x] 4.1 Headless A/B: render the `mirrored_arealight` scene (or any scene with the
      flag toggled) with `cameraMirror` off vs on; assert `on == np.fliplr(off)`
      within noise — on native Metal (via `select_backend`)
- [x] 4.2 Repeat 4.1 forcing the Vulkan backend (MoltenVK) — proves the flip is
      backend-agnostic and that both `.spv` paths apply it

## 5. Verification — full pbrt parity gate

- [x] 5.1 Author `tests/pbrt/corpus/mirrored_arealight.pbrt` = the existing
      `diffuse_arealight` scene with `Scale -1 1 1` inserted before its `LookAt`
      (improper camera); everything else identical
- [x] 5.2 Render the pbrt-v4 reference EXR (`~/projects/pbrt-v4`, same spp/res as
      the sibling scenes); commit it with `git add -f` under `corpus/refs/` and add
      its entry (version, content hash, per-scene tolerance) to the corpus manifest
- [x] 5.3 Wire the per-scene gate into `test_parity.py`: exposure-aligned relMSE /
      FLIP within tolerance; sanity-check that the SAME scene rendered WITHOUT the
      flag fails the gate (i.e. the flip is what earns parity, not a no-op)
- [x] 5.4 Run the full GPU parity gate (Metal): all corpus scenes incl.
      `mirrored_arealight` green

## 6. Docs + close-out

- [x] 6.1 `docs/PbrtImport.md`: move the improper-camera row from "approx
      (mirrored)" to matched, with the measured per-scene error
- [x] 6.2 `docs/Architecture.md`: note `FrameConstants.cameraMirror` in the
      FrameConstants / camera section
- [x] 6.3 `CHANGELOG.md`: add the mirrored-camera flip entry
- [x] 6.4 Update `openspec/changes/pbrt-v4-scene-import/tasks.md` status line:
      strike "mirrored-camera flip support" from the remaining list
- [x] 6.5 `ruff check src/` clean; full `pytest` (unit + GPU parity) green;
      `openspec validate pbrt-mirrored-camera-flip --strict`; verify no doc drift
