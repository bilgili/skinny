## Why

An **improper** pbrt camera — a camera-to-world transform with negative
determinant, e.g. a `Scale -1 1 1` (or any odd number of reflections) before
`LookAt` — produces a horizontally **mirrored** image relative to a proper
right-handed camera. The pbrt importer already **detects** this
(`pbrt/api.py:_emit_camera`, `T.is_orientation_reversing`) and records
`customData["pbrt"]["mirrored"] = True` on the camera prim plus a report note,
rather than silently shipping a wrong render. But the renderer **drops** the
reflection: `usd_loader._extract_camera` reconstructs a right-handed
`CameraOverride` from position + forward, and `_override_to_orbit` rebuilds an
orthonormal right-handed basis (yaw/pitch). So an imported improper camera renders
**horizontally flipped vs pbrt** — the last open item from `pbrt-v4-scene-import`
("Remaining (renderer-side, out of importer scope): … mirrored-camera flip
support.").

## What Changes

- **Honor the `mirrored` flag in the renderer.** The loader reads
  `customData["pbrt"]["mirrored"]` into a new `CameraOverride.mirrored` field; the
  renderer carries it as host state and folds it into `FrameConstants` as a new
  scalar `uint cameraMirror`.
- **Flip primary rays at one chokepoint.** `zoomedNDC` (`common.slang`) negates
  `ndc.x` when `cameraMirror != 0`. Both the `pinhole` and `thick_lens` cameras,
  in **both** megakernel and wavefront execution modes, route their primary-ray
  NDC through `zoomedNDC`, so all four combinations inherit the flip from a single
  edit. The camera position / forward / basis stay right-handed, so normals, BVH
  traversal, winding, and world-space math are untouched — only screen-space x is
  mirrored, exactly reproducing a `Scale -1` reflection camera.
- **Keep BDPT / light tracing consistent.** The `sampleWi` (camera-connection)
  path in `pinhole` and `thick_lens` negates `ndc.x` before the pixel mapping
  under the same flag, so bidirectional camera connections land on the mirrored
  pixel and agree with the mirrored primary frame.
- **Accumulation correctness.** `cameraMirror` joins `_current_state_hash()` so a
  scene whose mirror state differs starts a clean accumulation.

No GUI or CLI knob: the flip is driven solely by the imported metadata (YAGNI).

## Capabilities

### New Capabilities
- `mirrored-camera-rendering`: The renderer SHALL render a camera flagged
  improper/mirrored (negative-determinant pbrt camera-to-world, carried as
  `customData["pbrt"]["mirrored"]`) as a horizontal screen-space mirror, matching
  the pbrt reference, across both execution modes, both camera models, and both
  backends — without altering world-space geometry, normals, or winding.

### Modified Capabilities
<!-- None. `pbrt-scene-import` → "Camera translation" already covers detecting and
     flagging the improper camera (importer side, unchanged). This change is the
     renderer-side rendering behavior, a new capability. -->

## Impact

- **Host:** `scene.py` (`CameraOverride.mirrored`), `usd_loader.py`
  (`_extract_camera` reads the flag), `renderer.py` (host state, `_pack_uniforms`
  tail field, `_FC_FIELDS` drift guard, `_current_state_hash`).
- **Shaders:** `common.slang` (`FrameConstants.cameraMirror`, `zoomedNDC`),
  `cameras/pinhole.slang` + `cameras/thick_lens.slang` (`sampleWi`). Recompile
  `main_pass.spv` + the wavefront `.spv` cache.
- **Tests:** new `mirrored_arealight` parity corpus scene + pbrt-v4 reference EXR
  (gated relMSE/FLIP in `test_parity.py`); host unit tests for the loader threading
  and the UBO byte-layout/drift guard; a headless A/B asserting mirrored output is
  the column-flip of the non-mirrored render on Metal and Vulkan.
- **Docs:** `docs/PbrtImport.md` (parity matrix: improper-camera row approx →
  matched), `docs/Architecture.md` (FrameConstants `cameraMirror` note),
  `CHANGELOG.md`.
- **Backwards compatible:** default `cameraMirror = 0` ⇒ a non-mirrored render is
  byte-identical (`ndc.x` untouched). Proper cameras are unaffected.
