## ADDED Requirements

### Requirement: Unified render-selection flags across every front-end

The render-selection flags SHALL be exposed identically by every front-end — the
windowed app (`skinny`), the Qt GUI (`skinny-gui`), the web server
(`skinny-web`), and the headless renderer (`skinny-render`) — defined from a
single shared source so they cannot drift. The three flags are:

- `--integrator {path,bdpt}` — default `path`.
- `--execution-mode {megakernel,wavefront}` — default `megakernel`, with a
  `SKINNY_EXECUTION_MODE` environment fallback.
- `--bdpt-walk {fused,eye,eye_light}` — default `fused`, with a
  `SKINNY_BDPT_WALK` environment fallback; affects only `wavefront` + `bdpt`.

On the interactive front-ends (`skinny`, `skinny-gui`, `skinny-web`)
`--integrator` SHALL set the **initial** integrator and the integrator SHALL
remain runtime-cycleable; the execution mode and walk SHALL remain fixed for the
session.

#### Scenario: Same flags on every front-end

- **WHEN** any of `skinny`, `skinny-gui`, `skinny-web`, `skinny-render` is run
  with `--help`
- **THEN** all three flags `--integrator`, `--execution-mode`, `--bdpt-walk`
  are present with identical choices and defaults

#### Scenario: Interactive front-end launches into the chosen integrator

- **WHEN** an interactive front-end is launched with `--integrator bdpt`
- **THEN** the renderer starts on the `bdpt` integrator, and the integrator can
  still be cycled at runtime

#### Scenario: Headless renders in the chosen execution mode

- **WHEN** `skinny-render` is run with `--execution-mode wavefront` on a Vulkan
  backend
- **THEN** the headless render uses the wavefront backend for the run
