# neural-directional-proposal Specification

## Purpose
TBD - created by archiving change neural-directional-proposal. Update Purpose after archive.
## Requirements
### Requirement: Learned directional proposal with exact solid-angle pdf
The neural directional proposal SHALL draw a bounce direction from a learned distribution
conditioned on the local shading state and SHALL report its exact probability density in
solid-angle measure (sr⁻¹), supporting both drawing a sample (forward) and evaluating the
density of an externally supplied direction (inverse) so it can participate in the mixture
pdf.

#### Scenario: Draw produces a valid sample and density
- **WHEN** the neural proposal is selected and a bounce direction is drawn at a vertex
- **THEN** it returns a unit direction in the upper hemisphere and a finite positive
  solid-angle pdf

#### Scenario: Density of a direction it did not draw
- **WHEN** the mixture needs the neural density for a light-sampled or other-proposal
  direction `wi`
- **THEN** the proposal returns the exact solid-angle pdf it would assign to `wi` via
  inverse evaluation

#### Scenario: Density integrates to one
- **WHEN** the learned density is integrated over the hemisphere for a fixed condition
- **THEN** the integral is approximately 1

### Requirement: Frozen offline-trained weights as loadable GPU state
The neural proposal SHALL load frozen, offline-trained network weights from a file and
upload them to GPU buffers it owns, allocating those buffers and their descriptor bindings
only while the proposal is active and releasing them when it is deselected.

#### Scenario: Activation allocates and loads
- **WHEN** the neural proposal is selected for a render
- **THEN** its weight buffers are allocated, bound, and populated from the weights file

#### Scenario: Deselection releases
- **WHEN** the neural proposal is deselected
- **THEN** its buffers and bindings are released and no neural GPU state remains

### Requirement: Wavefront-only neural inference pass
The renderer SHALL run neural proposal inference as a wavefront compute pass that consumes
per-lane hit state and produces a per-lane direction and pdf for the bounce stage, and the
neural proposal SHALL NOT be available in the megakernel backend.

#### Scenario: Wavefront produces per-lane proposal
- **WHEN** a render uses the wavefront backend with the neural proposal active
- **THEN** a neural pass writes a per-lane `(wi, pdf)` that the bounce MIS mixture consumes

#### Scenario: Megakernel rejects neural
- **WHEN** the neural proposal is requested with the megakernel backend
- **THEN** the renderer reports it as unsupported rather than silently ignoring the request

### Requirement: Unbiased composition and default parity
With the default proposal set the renderer SHALL produce output identical to before this
change, and enabling the neural proposal SHALL remain unbiased so that the converged image
with `{bsdf, neural}` matches the reference image.

#### Scenario: Default selection is byte-identical
- **WHEN** the renderer runs with the default proposal set `{bsdf}` on a fixed scene,
  seed, and frame count
- **THEN** the output image is pixel-identical to the pre-change renderer

#### Scenario: Neural proposal is unbiased
- **WHEN** `{bsdf, neural}` renders a scene to a high sample count
- **THEN** the converged image matches the BDPT / bsdf-only reference within noise

### Requirement: Slang inference matches the reference implementation
The Slang neural inference SHALL reproduce the reference (PyTorch) flow's sampling and pdf
for identical inputs within a numerical tolerance.

#### Scenario: pdf parity on fixed inputs
- **WHEN** the same condition and base sample are fed to the Slang and PyTorch flows
- **THEN** the produced direction and its solid-angle pdf agree within tolerance

### Requirement: Offline training-record dump
The renderer SHALL be able to emit per-vertex path records of position, outgoing-sampled
direction, and contribution to a file so a per-scene network can be trained offline.

#### Scenario: Dump captures records
- **WHEN** record-dump mode is enabled during a render
- **THEN** per-vertex `(position, wi, contribution)` records are written for offline training

### Requirement: Per-sample network version
Each neural proposal sample SHALL carry the network version that produced it, and any later
density evaluation for that sample SHALL use that version's pdf, with baseline version 0 for
the frozen network.

#### Scenario: Sample stamps its version
- **WHEN** the neural proposal draws a sample
- **THEN** the sample records the active network version and its pdf is evaluated against
  that version

### Requirement: Command-line, GUI, and persisted selection
The neural proposal SHALL be selectable via the proposal-set command-line flag, surfaced in
the interactive UI selector, and persisted in user settings, consistent with the other
proposals, and changing the selection SHALL reset progressive accumulation.

#### Scenario: Selection flag activates the proposal
- **WHEN** the application is launched with `--proposals bsdf,neural`
- **THEN** the neural proposal is active for the session and the selection persists

#### Scenario: Changing selection resets accumulation
- **WHEN** the neural proposal is enabled or disabled
- **THEN** progressive accumulation resets

### Requirement: Equations are backed by embedded shader code and a symbol map

`docs/NeuralGuiding.md` SHALL present, beneath each governing equation, a
generated Slang excerpt of the function that implements it (its signature plus
the line(s) performing the equation's arithmetic, embedded via the
`docs-equation-code-embedding` generator) and a per-equation table mapping each
math symbol to the Slang identifier that carries it. This is in addition to the
existing equation → implementation mapping. The embedded excerpts SHALL be generated from the shipped shaders
(`sampling/neural_flow.slang`, `sampling/neural_proposal.slang`,
`sampling/proposal.slang`, `integrators/path_record.slang`), not hand-transcribed.

#### Scenario: Each equation shows the code that computes it

- **WHEN** a reader views an equation in `docs/NeuralGuiding.md` (e.g. the
  rational-quadratic spline `y = y_k + h_k(sθ²+…)/(…)` or the mixture pdf `p_mix`)
- **THEN** a generated fenced `slang` excerpt of the implementing function
  (`nf_rqs_fwd`, `mixtureProposalPdf`, …) appears beneath it, tagged with its
  source file

#### Scenario: Math symbols are mapped to Slang identifiers

- **WHEN** a reader needs the variable correspondence for an equation whose Slang
  names diverge from the math symbols
- **THEN** a per-equation table maps each symbol to its Slang name
  (e.g. `θ → theta`, `s → delta`, `h_k → hgt`, `w_k → w`, `d_k → d0`,
  `d_{k+1} → d1`, `y_k → y0`)

#### Scenario: Embedded code cannot silently drift

- **WHEN** a documented neural-flow shader function is renamed or moved
- **THEN** regenerating the doc (or running the embed check) surfaces the change
  rather than leaving a stale snippet, because the excerpts are generated from
  marked source regions

### Requirement: Live-updated weights with per-sample version
The neural directional proposal SHALL accept a weight swap between frames (rather
than only frozen offline weights), SHALL stamp each emitted sample with the network
version that produced it, and SHALL evaluate the density of a sample against its
stamped version so that an asynchronous swap leaves the estimator unbiased.

#### Scenario: Weights swap between frames
- **WHEN** new weights are published while the proposal is active
- **THEN** the proposal uses them from the next frame boundary without re-allocating its
  buffers or interrupting rendering

#### Scenario: Sample carries and uses its version
- **WHEN** a sample is drawn and later has its density evaluated in the mixture after a swap
- **THEN** the density is computed from the network version stamped on the sample, not the
  current render-side version

### Requirement: Frozen-weight behavior preserved when training is off
With online training disabled the neural proposal SHALL behave exactly as the frozen
Stage 1 proposal, loading baked weights and never swapping.

#### Scenario: Online disabled is frozen
- **WHEN** the neural proposal runs without online training enabled
- **THEN** weights are loaded once and never swapped, and behavior matches the frozen
  proposal

