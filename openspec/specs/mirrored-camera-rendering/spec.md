# mirrored-camera-rendering Specification

## Purpose
TBD - created by archiving change pbrt-mirrored-camera-flip. Update Purpose after archive.
## Requirements
### Requirement: Improper (mirrored) camera renders as a horizontal mirror

The renderer SHALL render a camera flagged improper/mirrored — a pbrt
camera-to-world transform with negative determinant, carried through import as
`customData["pbrt"]["mirrored"] = True` on the camera prim — as a horizontal
screen-space mirror, so the converged image matches the pbrt reference for that
camera. The loader SHALL surface the flag as `CameraOverride.mirrored` and the
renderer SHALL fold it into `FrameConstants` as a flag read by primary-ray
generation.

#### Scenario: Imported improper camera matches pbrt
- **WHEN** a pbrt scene whose camera has an orientation-reversing transform (e.g. `Scale -1 1 1` before `LookAt`) is imported and rendered to convergence
- **THEN** the image is the horizontal mirror of the proper-camera render, and its exposure-aligned relMSE / FLIP versus the pbrt reference are within the corpus parity tolerance

#### Scenario: The flip is exactly a column reversal
- **WHEN** the same scene is rendered with the mirror flag off and on
- **THEN** the flag-on image equals the left-right column reversal (`fliplr`) of the flag-off image within noise

#### Scenario: Proper camera is unaffected
- **WHEN** a scene with a proper (non-mirrored) camera is rendered
- **THEN** the render is byte-identical to the pre-change behavior (the mirror flag defaults off and `ndc.x` is untouched)

### Requirement: Mirror applies across both execution modes, camera models, and backends

The mirror SHALL be applied at the shared primary-ray NDC chokepoint
(`zoomedNDC`), so it takes effect for both the pinhole and thick-lens camera
models, in both megakernel and wavefront execution modes, on both the Vulkan and
native Metal backends, without altering world-space geometry, surface normals, or
triangle winding (only screen-space x is mirrored).

#### Scenario: Wavefront and megakernel agree
- **WHEN** an improper-camera scene is rendered in megakernel mode and in wavefront mode
- **THEN** both produce the mirrored image (consistent with each other and with pbrt)

#### Scenario: Realistic (thick-lens) camera also mirrors
- **WHEN** the improper-camera scene uses a `realistic` lens (thick-lens path) rather than a pinhole
- **THEN** the rendered image is mirrored the same way (the thick-lens primary-ray path honors the flag)

#### Scenario: Geometry is not reflected
- **WHEN** an improper camera is rendered
- **THEN** scene geometry, normals, and winding are unchanged versus the proper-camera render — the only difference is the screen-space horizontal mirror

### Requirement: Bidirectional camera connections honor the mirror

The camera-connection (`sampleWi`) path used by BDPT and light tracing SHALL,
when the mirror flag is set, map a world point to the mirrored pixel consistent
with the mirrored primary-ray frame, so bidirectional estimators remain correct
under an improper camera.

#### Scenario: BDPT camera connection lands on the mirrored pixel
- **WHEN** BDPT (or a light-tracing splat) connects a world point to the camera under an improper-camera scene
- **THEN** the contribution is splatted to the same pixel the mirrored primary ray would originate from, so the BDPT image agrees with the path-traced mirrored image

### Requirement: Mirror state participates in accumulation reset

A change in the camera mirror state SHALL invalidate progressive accumulation, so
toggling or loading a differing mirror state starts a clean accumulation rather
than blending mirrored and non-mirrored samples.

#### Scenario: Changing mirror state resets accumulation
- **WHEN** the effective camera mirror state changes between frames
- **THEN** `_current_state_hash()` changes and the accumulation buffer resets to frame 0

