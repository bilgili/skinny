# pbrt-camera-up-axis Specification

## Purpose
TBD - created by archiving change pbrt-camera-up-axis. Update Purpose after archive.
## Requirements
### Requirement: Authored camera up/roll is honored

The renderer SHALL reconstruct an imported camera's view basis from the authored
**up** vector, not a hardcoded world-up of `(0, 1, 0)`. The loader SHALL extract
the camera's world-space up (the camera's local +Y transformed by its world
matrix, the analogue of the existing local −Z → forward extraction) and surface
it as `CameraOverride.up`; the renderer SHALL build the camera basis from
`(position, forward, up)` so the authored roll about the forward axis is
preserved. This makes a pbrt scene whose camera up is not ≈ +Y (e.g. the Z-up
convention) render at the correct orientation instead of rolled.

The up SHALL default to `(0, 1, 0)` wherever a `CameraOverride` is constructed
without one and for every interactive/default camera, so the change is inert
unless the authored up differs from +Y.

#### Scenario: Z-up authored camera renders upright (no roll)

- **WHEN** a pbrt scene whose camera up vector is essentially +Z (the pbrt Z-up
  convention, e.g. `sssdragon` with up `(-0.317, 0.312, 0.895)`) is imported and
  rendered
- **THEN** the reconstructed camera basis reproduces the pbrt camera basis
  (forward, up, right, mapped through the pbrt→skinny change-of-basis) to
  floating-point tolerance, and the converged frame matches the pbrt v4 reference
  orientation (not rolled ~90°)

#### Scenario: Y-up cameras are unchanged (back-compat)

- **WHEN** a camera whose authored up is ≈ +Y is imported (the entire pbrt parity
  corpus, and every interactive/default camera)
- **THEN** the reconstructed basis and the rendered image are byte-identical to
  the pre-change behavior, and the pbrt parity corpus stays within its existing
  relMSE / FLIP tolerances

#### Scenario: Degenerate up parallel to forward does not break the basis

- **WHEN** the authored up is (near-)parallel to the forward direction (a camera
  looking along the world-up axis)
- **THEN** `_look_at` falls back to a secondary reference axis and returns a
  finite, orthonormal basis (no NaN, no zero `right` row)

#### Scenario: Up composes with the mirror flag

- **WHEN** a camera is both improper/mirrored (`Scale -1`) and non-Y-up (e.g.
  `sssdragon`)
- **THEN** the authored up orients the basis and the existing `mirrored` ndc.x
  horizontal flip still applies, independently — both corrections compose

