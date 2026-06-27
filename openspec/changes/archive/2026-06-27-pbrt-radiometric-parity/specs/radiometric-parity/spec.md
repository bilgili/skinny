## ADDED Requirements

### Requirement: pbrt film params are live-editable on the scenegraph camera

The importer SHALL author the pbrt film exposure controls — ISO and exposure time
(shutter) — as attributes on the `UsdGeom.Camera` prim (`skinny:film:iso`,
`skinny:film:exposureTime`, plus the standard `exposure` attr), NOT bake them into
emitter/environment radiance. The renderer SHALL read these attributes, derive the
imaging ratio `exposureTime · ISO / 100`, and apply it as a live linear output
scale that is re-read each frame, so ISO and exposure can be retuned on the fly
without re-importing the scene. Because scaling every emitter by a scalar is
algebraically identical to scaling the output for a linear path tracer, a scene
imported and rendered with default film params SHALL be radiometrically unchanged
from the previous (baked) behaviour.

#### Scenario: ISO/exposure round-trip through the scene graph

- **WHEN** a pbrt scene with a non-default `Film "iso"` (and/or a shutter interval)
  is imported
- **THEN** the `UsdGeom.Camera` carries `skinny:film:iso` and
  `skinny:film:exposureTime`, and no emitter or environment radiance has the
  imaging ratio baked into it

#### Scenario: retuning ISO live re-exposes without re-import

- **WHEN** the user changes the ISO or exposure-time control while a scene is loaded
- **THEN** the renderer re-reads the camera film attributes, re-applies the imaging
  ratio as the output scale, and resets progressive accumulation — with no scene
  re-import and no change to geometry, materials, or lights

### Requirement: Imported scenes match pbrt's absolute radiance

The headless linear-HDR render of an imported pbrt scene SHALL reproduce pbrt's
absolute output radiance — not only its exposure-matched structure — within a
recorded mean-ratio tolerance. The film imaging ratio SHALL reach every light
(including an infinite light that references a `.hdr` map directly, whose pbrt
`scale` SHALL NOT be dropped), and the emissive-light radiance SHALL be normalized
so that a diffuse-white patch under a known light matches pbrt's patch radiance
independent of light type. An absolute (un-exposure-aligned) gate SHALL guard this
alongside the existing exposure-blind gate.

#### Scenario: the imaging ratio is not dropped on a direct-.hdr env

- **WHEN** a pbrt `infinite` light references a `.hdr` map with a non-default `scale`
- **THEN** the emitted DomeLight carries that `scale` (the imaging ratio applies
  live), so its IBL contribution matches pbrt within tolerance — it is not rendered
  at unit scale

#### Scenario: emissive radiance is light-type-independent

- **WHEN** a diffuse-white patch is rendered under an area light, and separately
  under an infinite light, with neutral film params
- **THEN** both patch radiances match pbrt within the same tolerance (no ~1.6×
  area-light offset), confirming the emissive / NEE radiance normalization

#### Scenario: an absolute-radiance gate guards against regression

- **WHEN** a corpus scene is rendered for the parity matrix
- **THEN** an absolute (un-exposure-aligned) mean-ratio + relMSE gate is evaluated
  alongside the existing exposure-blind gate, and an absolute-brightness regression
  fails the build (the exposure-blind gate is retained, not replaced)
