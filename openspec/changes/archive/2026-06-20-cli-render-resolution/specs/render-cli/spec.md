## ADDED Requirements

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

When the shared flags are exposed, the windowed app (`skinny`) SHALL size both
its window and its GPU render target to the requested width/height, and the Qt
GUI (`skinny-gui`) SHALL size its offscreen render area — the pixels the
renderer computes — to the requested width/height, without resizing the
surrounding Qt window or dock layout.

The headless renderer (`skinny-render`), which already defines its own
`--width` / `--height` for offline output size, SHALL opt out of the shared
definition so that no argparse flag conflict arises and its existing default
(1024×1024) is unchanged.

#### Scenario: Default render area is 640×480

- **WHEN** `skinny` or `skinny-gui` is launched with no `--width`/`--height`
  flag and no `SKINNY_WIDTH`/`SKINNY_HEIGHT` environment variable set
- **THEN** the render area is 640×480

#### Scenario: Flags size the skinny window and render target

- **WHEN** `skinny` is launched with `--width 800 --height 600`
- **THEN** the GLFW window and the GPU render target are both 800×600

#### Scenario: Flags size the skinny-gui offscreen render area

- **WHEN** `skinny-gui` is launched with `--width 800 --height 600`
- **THEN** the offscreen render target the renderer computes is 800×600, and the
  surrounding Qt window and dock layout keep their own size

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
