# Design — pbrt mirrored-camera flip (renderer-side)

## Context

pbrt allows a camera-to-world transform with negative determinant (an odd number
of reflections, classically `Scale -1 1 1` before `LookAt`). pbrt renders such a
camera faithfully: the produced image is the horizontal mirror of the
proper-camera image. skinny reconstructs a **right-handed** camera from the
authored transform (`_extract_camera` → position + forward; `_override_to_orbit` →
yaw/pitch/distance), which discards the reflection, so an imported improper camera
renders mirrored vs the pbrt reference.

The importer half is already done (commit `b40893b`): `pbrt/api.py::_emit_camera`
calls `T.is_orientation_reversing(cam.camera_to_world)` and, when true, sets
`camera_md["mirrored"] = True` (persisted via `metadata.tag_prim` into
`customData["pbrt"]`) and appends a report note. This change finishes the
renderer half.

## Decisions

**D1 — Flip mechanism: negate `ndc.x` in `zoomedNDC`, gated by a flag.**
`zoomedNDC(fc, pixel, jitter)` (`common.slang`) is the single function every
primary-ray generator calls: `PinholeCamera.generateRay` and
`ThickLensCamera.generateRay`, in both the megakernel and wavefront pipelines.
Negating `ndc.x` there when `fc.cameraMirror != 0` mirrors the image in
screen space and nothing else.

Rejected alternatives:
- *Bake the mirror into the view matrices on the host* (negate the right axis in
  `view`/`viewInverse`): makes the camera basis left-handed, risking winding /
  normal / `sampleWi` projection assumptions elsewhere in the pipeline. Rejected.
- *Flip the final framebuffer horizontally* (post-process column reverse):
  cheapest, but breaks every pixel-space readback (BXDF scene-pick `pickPixel` /
  `toolBuffer`, lens-vignette debug, `sampleWi` light splats) and is a display
  hack rather than a camera model. Rejected.

**D2 — Flag transport: a new scalar `uint cameraMirror` at the tail of
`FrameConstants`.** Appended after `recordMode` (the current last field). There is
no spare bit in an existing bitfield with compatible semantics (`detailFlags` is
detail-map specific; overloading it couples unrelated state). A dedicated scalar
is the documented extension pattern — every "scalar tail" field in
`FrameConstants` was added the same way. Touch points kept in sync:
`common.slang` struct, `renderer._pack_uniforms` append, `renderer._FC_FIELDS`
(the Metal MSL relocation list + its cumulative-size drift guard).

**D3 — Drive from metadata only (no GUI/CLI knob).** The flip is a faithful
reproduction of the authored pbrt camera, not an interactive control. Adding a
toggle means front-end UI across every GUI, a CLI flag, settings persistence, and
docs — none of which serve the parity goal. YAGNI. (If an arbitrary-camera mirror
toggle is wanted later, it reuses the same `cameraMirror` field.)

**D4 — Flip `sampleWi` too (BDPT / light tracing).** `zoomedNDC` covers primary
rays only. `sampleWi` (camera connection, used by BDPT and any light-tracing
splat) computes the target pixel from a world point independently. Without a
matching flip it would land on the un-mirrored pixel, so BDPT renders of a
mirrored camera would be wrong even though the path-tracer parity gate (primary
rays only) would pass. Both `PinholeCamera.sampleWi` and
`ThickLensCamera.sampleWi` negate `ndc.x` before the `xs`/pixel mapping under the
same flag. Small, removes a latent bug.

**D5 — Accumulation reset.** `cameraMirror` is included in
`_current_state_hash()`. In practice the flag is set once at scene load and is
constant thereafter, but folding it into the hash keeps the invariant "any state
that changes the image resets accumulation" intact.

## Data flow

```
pbrt Scale -1 ... LookAt
   │  importer (done): T.is_orientation_reversing → customData["pbrt"]["mirrored"]=True
   ▼
USD camera prim customData
   │  usd_loader._extract_camera: read flag → CameraOverride.mirrored
   ▼
CameraOverride.mirrored (scene.py)
   │  renderer: store self._camera_mirror; _current_state_hash; _pack_uniforms
   ▼
FrameConstants.cameraMirror (uint)
   │  zoomedNDC: ndc.x = -ndc.x      (pinhole + thick_lens, mega + wavefront)
   │  sampleWi:  ndc.x = -ndc.x      (pinhole + thick_lens)
   ▼
horizontally-mirrored frame == pbrt reference
```

## Geometry note

The change of basis B (left-handed pbrt → right-handed skinny, `transform.py`) is
already orientation-reversing for *geometry* and is handled by the existing winding
flip during baking. The camera improper-ness is a **separate** reflection living
only in the camera transform; B does not absorb it (B is applied to every CTM
uniformly). So the camera mirror must be applied at ray-gen, independent of the
geometry winding path — these are orthogonal and must not be conflated.

## Verification

- **Parity gate (full):** add `mirrored_arealight` to the corpus = the existing
  `diffuse_arealight` scene with `Scale -1 1 1` inserted before `LookAt`. Render a
  pbrt-v4 reference EXR (`~/projects/pbrt-v4`, version pinned + hashed in the
  manifest, same as the other refs). Gate exposure-aligned relMSE / FLIP at the
  corpus tolerance in `test_parity.py`. Skips cleanly with no GPU/refs.
- **A/B sanity (host, no pbrt):** headless-render the scene with and without the
  flag and assert `mirrored == np.fliplr(non_mirrored)` within noise, on native
  Metal and Vulkan, proving the flip is exactly a column reversal and backend-
  agnostic.
- **Unit:** loader threads `customData["pbrt"]["mirrored"]` → `CameraOverride`;
  `_pack_uniforms` length + `_FC_FIELDS` cumulative-size drift guard stay
  consistent after the new field.

## Risks / mitigations

- **UBO byte drift between Vulkan packer and Metal MSL relocation.** Mitigated by
  the existing `_FC_FIELDS` drift-guard assertion (`Σ sizes == len(_pack_uniforms())`);
  add `("cameraMirror", 4)` in both places in lockstep.
- **Stale `.spv`.** A `common.slang` edit requires recompiling `main_pass.spv` and
  busting the wavefront `.spv` cache; the worktree's `.spv` files are isolated from
  main's so parity A/B reflects the new shader (see worktree dev gotchas).
