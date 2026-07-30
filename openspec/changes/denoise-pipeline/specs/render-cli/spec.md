## ADDED Requirements

### Requirement: Denoiser selection flags across every front-end

The denoiser SHALL be selected by `--denoiser` and `--denoise-scale`, defined in
the **same shared source** as the other render-selection flags, so all four
front-ends expose them from one definition and cannot drift.

`--denoiser` SHALL accept the registered denoiser names plus `none`, SHALL
default to `none`, and SHALL read `SKINNY_DENOISER` as an environment fallback,
with precedence explicit flag > environment variable > default.

`--denoise-scale` SHALL accept a float in the range 0.25 to 1.0, SHALL default to
1.0, and SHALL read `SKINNY_DENOISE_SCALE` as an environment fallback, with the
same precedence.

Neither flag SHALL be persisted, because both select GPU resource extents that
are fixed for the session.

#### Scenario: Default is no denoiser

- **WHEN** any front-end is launched with no `--denoiser` flag and no
  `SKINNY_DENOISER` set
- **THEN** no denoiser is active, no auxiliary image is allocated, and the
  rendered image is unchanged from before this change

#### Scenario: The same flags on every front-end

- **WHEN** `skinny`, `skinny-gui`, `skinny-render`, or `skinny-web` is run with
  `--help`
- **THEN** `--denoiser` and `--denoise-scale` are present with identical choices
  and identical defaults, from the shared definition

#### Scenario: An environment default is validated

- **WHEN** a front-end is launched with `SKINNY_DENOISE_SCALE=4.0` set and no
  flag passed
- **THEN** the launch is refused at startup with a message naming the flag and
  the permitted range, before any GPU device is constructed

### Requirement: Reject impossible denoiser combinations at startup

A denoiser combination the host cannot run SHALL be refused at startup with a
message that names the offending flag and the reason. No such combination SHALL
be silently downgraded to no denoiser.

The refused combinations SHALL be:

- a denoiser name the resolved GPU backend cannot run;
- a denoiser whose optional dependency is not installed, with the message naming
  the install command;
- `--denoise-scale` set to a value other than 1.0 while `--denoiser` is `none`;
- a denoise scale outside the range the selected denoiser permits.

Each refusal SHALL read the render-envelope predicate for its rule and SHALL own
only the code-to-message mapping.

#### Scenario: A scale without a denoiser is refused

- **WHEN** any front-end is launched with `--denoise-scale 0.5` and no denoiser
- **THEN** the launch is refused with a message naming both flags

#### Scenario: A backend mismatch is refused before the device is built

- **WHEN** a denoiser is requested that the resolved backend cannot run
- **THEN** the launch is refused, and the renderer's GPU context is never
  constructed

#### Scenario: No silent downgrade

- **WHEN** any refused denoiser combination is launched
- **THEN** the process exits non-zero, and no frame is rendered without the
  requested denoiser

## MODIFIED Requirements

### Requirement: Render-area resolution flags

The render-area pixel size SHALL be controlled by `--width` and `--height`
flags defined in the **same shared source** as the other render-selection
flags, so the interactive front-ends expose them from one definition and cannot
drift. Both flags:

- SHALL accept positive integers;
- SHALL default to **640** (`--width`) and **480** (`--height`) when neither the
  flag nor its environment fallback is set;
- SHALL read `SKINNY_WIDTH` / `SKINNY_HEIGHT` as environment fallbacks, with
  precedence explicit flag > environment variable > default.

`--width` and `--height` SHALL set the **output extent** — the size of the
displayed image, the written file, the HUD overlay, and the swapchain. When a
denoiser upscales, the **render extent** — the size the path tracer and the
accumulation image use — SHALL be derived from the output extent and
`--denoise-scale`. With no denoiser the two extents SHALL be equal, which is the
behaviour before this change.

When the shared flags are exposed, the windowed app (`skinny`) SHALL size both
its window and its output extent to the requested width/height, and the Qt
GUI (`skinny-gui`) SHALL size its offscreen output extent — the pixels the user
sees — to the requested width/height, without resizing the surrounding Qt window
or dock layout.

The headless renderer (`skinny-render`), which already defines its own
`--width` / `--height` for offline output size, SHALL opt out of the shared
definition so that no argparse flag conflict arises and its existing default
(1024×1024) is unchanged.

#### Scenario: Default render area is 640×480

- **WHEN** `skinny` or `skinny-gui` is launched with no `--width`/`--height`
  flag and no `SKINNY_WIDTH`/`SKINNY_HEIGHT` environment variable set
- **THEN** the render area is 640×480

#### Scenario: Flags size the skinny window and render target

- **WHEN** `skinny` is launched with `--width 800 --height 600` and no denoiser
- **THEN** the GLFW window, the output extent, and the render extent are all
  800×600

#### Scenario: A denoise scale separates the two extents

- **WHEN** `skinny` is launched with `--width 800 --height 600
  --denoiser metalfx --denoise-scale 0.5`
- **THEN** the GLFW window and the output extent are 800×600 and the render
  extent is 400×300

#### Scenario: Flags size the skinny-gui offscreen render area

- **WHEN** `skinny-gui` is launched with `--width 800 --height 600`
- **THEN** the offscreen output extent is 800×600, and the surrounding Qt window
  and dock layout keep their own size

#### Scenario: Environment fallback supplies the size

- **WHEN** `skinny` is launched with `SKINNY_WIDTH=1024` / `SKINNY_HEIGHT=768`
  set and no `--width`/`--height` flag passed
- **THEN** the render area is 1024×768, and an explicit `--width`/`--height` flag
  would override the environment value

#### Scenario: Same flags on the interactive front-ends

- **WHEN** `skinny` or `skinny-gui` is run with `--help`
- **THEN** `--width` and `--height` are present with identical defaults (640 and
  480) from the shared definition

#### Scenario: Headless keeps its own resolution flags

- **WHEN** `skinny-render` is run with `--help`
- **THEN** `--width` and `--height` are present with the headless default of
  1024×1024, and launching `skinny-render` raises no flag-conflict error

#### Scenario: Non-positive size is rejected at startup

- **WHEN** `skinny` or `skinny-gui` is launched with `--width 0` (or a negative
  width/height, or such a value from the environment fallback)
- **THEN** it prints a clear usage error naming the offending flag and exits
  without initializing the GPU
