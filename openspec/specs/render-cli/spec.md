# render-cli Specification

## Purpose

Provide a single, shared command-line surface for the three render-selection
axes — integrator (`path` / `bdpt`), execution mode (`megakernel` /
`wavefront`), and the wavefront-bdpt subpath-build walk (`fused` / `eye` /
`eye_light`) — so every front-end (windowed `skinny`, Qt `skinny-gui`, web
`skinny-web`, headless `skinny-render`) exposes the same flags from one
definition and cannot drift apart.
## Requirements
### Requirement: Unified render-selection flags across every front-end

The render-selection flags SHALL be exposed identically by every front-end — the
windowed app (`skinny`), the Qt GUI (`skinny-gui`), the web server
(`skinny-web`), and the headless renderer (`skinny-render`) — defined from a
single shared source so they cannot drift. The flags are:

- `--backend {auto,metal,vulkan}` — default `auto`, with a `SKINNY_BACKEND`
  environment fallback; selects the GPU backend, fixed for the session.
- `--integrator {path,bdpt,sppm}` — default `path`. `sppm` selects the
  Stochastic Progressive Photon Mapping integrator and requires
  `--execution-mode wavefront`.
- `--execution-mode {megakernel,wavefront}` — default `megakernel`, with a
  `SKINNY_EXECUTION_MODE` environment fallback.
- `--bdpt-walk {fused,eye,eye_light}` — default `fused`, with a
  `SKINNY_BDPT_WALK` environment fallback; affects only `wavefront` + `bdpt`.

On the interactive front-ends (`skinny`, `skinny-gui`, `skinny-web`)
`--integrator` SHALL set the **initial** integrator and the integrator SHALL
remain runtime-cycleable; the backend, execution mode, and walk SHALL remain
fixed for the session.

#### Scenario: Same flags on every front-end

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-web`, `skinny-render` is run
  with `--help`
- **THEN** all of `--backend`, `--integrator`, `--execution-mode`, `--bdpt-walk`
  are present with identical choices and defaults, and `--integrator` lists
  `path`, `bdpt`, and `sppm`

#### Scenario: Interactive front-end launches into the chosen integrator

- **WHEN** an interactive front-end is launched with `--integrator bdpt`
- **THEN** the renderer starts on the `bdpt` integrator, and the integrator can
  still be cycled at runtime

#### Scenario: Headless renders in the chosen execution mode

- **WHEN** `skinny-render` is run with `--execution-mode wavefront` on a Vulkan
  backend
- **THEN** the headless render uses the wavefront backend for the run

#### Scenario: Backend is selected on the command line

- **WHEN** any front-end is launched with `--backend vulkan` (or
  `SKINNY_BACKEND=vulkan`)
- **THEN** the renderer runs on the Vulkan backend for the whole session, and the
  backend is resolved through the single shared resolver used by every front-end

### Requirement: Reject impossible render-flag combinations at startup

The CLI front-ends SHALL validate the parsed render flags at startup and SHALL
exit with a clear error when a combination cannot work, rather than crashing
later or silently doing nothing. Specifically:

- When the integrator is explicitly `bdpt` AND either `--online-training` is set
  or the directional-proposal mixture includes the neural proposal, the
  front-ends SHALL raise a usage error naming the incompatible option and the fix
  (`--integrator path`), because BDPT does not consume the neural directional
  proposal on any backend or execution mode.
- When the integrator is `sppm` AND `--execution-mode` is `megakernel`, the
  front-ends SHALL raise a usage error stating that SPPM requires
  `--execution-mode wavefront` and naming that fix, because SPPM has no
  megakernel path.

The validation SHALL be shared across all front-ends (`skinny`, `skinny-gui`,
`skinny-web`, `skinny-render`) and SHALL NOT trip when the integrator is the
default/persisted path (not explicitly `bdpt` or `sppm`).

#### Scenario: bdpt + online-training exits with an error

- **WHEN** `skinny` (or any front-end) is launched with `--integrator bdpt
  --online-training`
- **THEN** it prints a clear error stating the combination is incompatible and
  to use `--integrator path`, and exits without initializing the GPU

#### Scenario: bdpt + neural proposal exits with an error

- **WHEN** a front-end that exposes `--proposals` is launched with `--integrator
  bdpt --proposals bsdf,neural`
- **THEN** it exits with the same clear error

#### Scenario: sppm + megakernel exits with an error

- **WHEN** a front-end is launched with `--integrator sppm --execution-mode megakernel`
- **THEN** it prints a clear error stating that SPPM requires
  `--execution-mode wavefront`, and exits without initializing the GPU

#### Scenario: compatible combinations are unaffected

- **WHEN** a front-end is launched with `--integrator path` (with or without a
  neural proposal / `--online-training`), or with `--integrator bdpt` and no
  neural proposal and no `--online-training`, or with `--integrator sppm
  --execution-mode wavefront`, or with no explicit `--integrator`
- **THEN** startup proceeds normally with no error

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

