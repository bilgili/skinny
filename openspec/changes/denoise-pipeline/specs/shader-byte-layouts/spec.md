## ADDED Requirements

### Requirement: FrameConstants carries the output extent and the jitter state

`FrameConstants` SHALL carry the output extent and the denoiser frame state in
addition to the render extent it already carries: the output width, the output
height, the reported sub-pixel jitter offset, and the jitter-mode selector.

The fields SHALL be derived by the layout authority from the authoritative
`.slang` declaration, like every other registered field, and SHALL NOT be
hand-listed at a packer. Adding them SHALL update the packer body and the pinned
goldens together.

The append position SHALL respect the existing blob rule that keeps the
build-gated tail fields at their recorded offsets, so no gated offset moves.

#### Scenario: Offsets are derived, not typed

- **WHEN** the layout for `FrameConstants` is produced
- **THEN** each new field's offset comes from the parsed `.slang` declaration

#### Scenario: Gated tail offsets do not move

- **WHEN** the layout is produced for the base variant and for the MLT variant
- **THEN** every pre-existing field keeps the offset it had before this change,
  including the build-gated tail fields

#### Scenario: Goldens fail on drift

- **WHEN** a new field is added to the shader struct without updating the packer
- **THEN** the pinned golden test fails
