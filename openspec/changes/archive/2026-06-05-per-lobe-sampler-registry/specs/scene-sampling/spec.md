## ADDED Requirements

### Requirement: Per-lobe sampler selection is host-registered and transported without new bindings

The renderer SHALL expose the per-lobe sampler selection through a host registry
of strategies (mirroring the directional-proposal registry): each strategy SHALL
declare its name, the lobes it is valid for, its shader dispatch id, and a CLI
token. The active selection SHALL be transported to the shader through a single
`FrameConstants` field (`flatLobeSamplers`, packed per lobe) and SHALL add **no**
new descriptor bindings, GPU buffers, or compute passes — the same "analytic
selection contributes no GPU state" rule the proposal seam follows. The selection
SHALL apply identically in the megakernel and wavefront execution modes.

#### Scenario: selection adds no GPU state

- **WHEN** any per-lobe sampler strategy is selected
- **THEN** the renderer transports the choice in the `flatLobeSamplers`
  `FrameConstants` field only, allocating no extra buffer, binding, or pass, in
  both megakernel and wavefront modes

#### Scenario: invalid strategy/lobe pairings are not offered

- **WHEN** the host builds the selectable strategies for a lobe
- **THEN** only strategies whose declared valid-lobe set includes that lobe are
  offered for it (e.g. spherical-cap VNDF is offered for coat/spec, not diffuse)

### Requirement: Per-lobe sampler selection on the command line, GUI, and persisted state

The active per-lobe sampler selection SHALL be selectable on the command line
(`--lobe-samplers`, with an environment-variable fallback), mirroring the
existing render-selection flags, and SHALL be surfaced as per-lobe selectors in
the interactive UI and persisted in the user settings snapshot. Changing any
per-lobe sampler selection SHALL reset progressive accumulation.

#### Scenario: CLI selection mirrors the other render-selection flags

- **WHEN** the application is launched with `--lobe-samplers
  coat=sphcap,spec=sphcap,diff=uniform` (or the environment-variable fallback)
- **THEN** that per-lobe selection is active for the session, consistent with how
  `--proposals` and `--integrator` behave

#### Scenario: changing a per-lobe sampler resets accumulation

- **WHEN** any of the coat, spec, or diffuse sampler selections changes
- **THEN** progressive accumulation resets so the new configuration accumulates
  cleanly
