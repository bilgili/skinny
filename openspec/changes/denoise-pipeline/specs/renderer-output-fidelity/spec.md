## ADDED Requirements

### Requirement: One accessor decides the source image for every output path

The renderer SHALL expose one accessor that returns the image every file-output
path reads — the denoised image while a denoiser runs, the accumulation image
otherwise. `render_headless()`, `save_screenshot()`, the EXR writer, and the
Radiance writer SHALL all read it, so no output path can drift into reading a
different image.

A separate accessor SHALL return the raw accumulation image and SHALL never
return the denoised image. The parity harness SHALL read that one.

The denoised image SHALL be linear high-dynamic-range at output extent, so the
high-dynamic-range writers need no tonemapping and no rescaling.

#### Scenario: Every output path reads the same source

- **WHEN** the output paths are compared with a denoiser active
- **THEN** each reads the shared accessor, and none reads the accumulation image
  directly

#### Scenario: A new output path cannot bypass the accessor

- **WHEN** an output path is added that reads the accumulation image directly
- **THEN** a test fails, naming the bypassing path

#### Scenario: High-dynamic-range output is linear and full size

- **WHEN** an EXR is written with a denoiser upscaling from a smaller render
  extent
- **THEN** the file holds linear values at the output extent
