## MODIFIED Requirements

### Requirement: Unified render-selection flags across every front-end

The render-selection flags SHALL be exposed identically by every front-end — the
windowed app (`skinny`), the Qt GUI (`skinny-gui`), the web server
(`skinny-web`), and the headless renderer (`skinny-render`) — defined from a
single shared source so they cannot drift. The flags are:

- `--backend {auto,metal,vulkan}` — default `auto`, with a `SKINNY_BACKEND`
  environment fallback; selects the GPU backend, fixed for the session.
- `--integrator {path,bdpt}` — default `path`.
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
  are present with identical choices and defaults

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
