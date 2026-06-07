# scene-sampling Specification

## Purpose
TBD - created by archiving change pluggable-scene-sampling. Update Purpose after archive.
## Requirements
### Requirement: Two orthogonal, composable sampling seams

The renderer SHALL expose two pluggable sampling seams that attach at distinct
points in the integrator and compose independently: a **directional-proposal**
seam at the BSDF bounce (which direction the path takes), and a
**reuse/resampling** seam around direct (NEE) and indirect lighting. Each seam
SHALL be backed by a host `SamplingPlugin` (`ProposalPlugin` or `ReusePlugin`)
that owns its lifecycle and any optional GPU passes, buffers, descriptor
bindings, and `FrameConstants` uniform fields. A plugin that needs no GPU state
SHALL contribute none. The two seams SHALL be selectable independently, so any
proposal set composes with any reuse mode.

#### Scenario: Proposal and reuse selected independently

- **WHEN** a proposal set and a reuse mode are chosen for a render
- **THEN** the integrator applies the chosen proposal mixture at the bounce and
  the chosen reuse mode around lighting, with neither selection constraining the
  other

#### Scenario: Analytic plugin contributes no GPU state

- **WHEN** a proposal that is a pure analytic callable (e.g. the BSDF proposal)
  is active
- **THEN** it adds no extra compute pass, buffer, or descriptor binding — only
  its uniform mask bit and selection weight

### Requirement: Directional-proposal mixture with one-sample MIS

The bounce SHALL draw its next direction from a **mixture** of active proposals
selected at runtime by `fc.proposalMask` and weighted by `fc.proposalAlpha`
(weights normalized to sum to 1). The mixture SHALL use one-sample MIS: select
one proposal with probability proportional to its weight, draw `wi`, and form
the throughput as `f(wo,wi)·cos / mixPdf` where `mixPdf = Σ_k α_k · pdf_k(wi)`
over the active proposals. Every proposal SHALL return its exact probability
density in **solid-angle measure (sr⁻¹)**, and SHALL return 0 for directions it
cannot produce. The density used downstream (sphere-light and environment-miss
MIS) SHALL be `mixPdf`, the actual density that produced the sample.

#### Scenario: Mixture estimator divides by the full mixture pdf

- **WHEN** more than one proposal is active and a non-delta direction is drawn
- **THEN** the path throughput divides by `Σ_k α_k · pdf_k(wi)` summed over all
  active proposals, not by the drawing proposal's pdf alone

#### Scenario: Delta lobes pass through unmixed

- **WHEN** the selected proposal returns a singular (delta) sample, such as a
  perfect mirror or smooth-dielectric lobe (pdf == 0)
- **THEN** the mixture is skipped and the material's own exact delta weight is
  used, and no continuous proposal contributes to that vertex

### Requirement: NEE multiple-importance-sampling is coupled to the proposal mixture

Next-event estimation SHALL compute its MIS companion weight against the **same**
proposal-mixture density used at the bounce. For a light-sampled direction
`wiToLight`, the BSDF-technique pdf in the NEE power-heuristic weight SHALL be
`mixturePdf(wiToLight)` over the active proposals, not the bare BSDF pdf. When
the only active proposal is the BSDF proposal, this density SHALL equal the BSDF
pdf, leaving the NEE estimate unchanged.

#### Scenario: Enabling a non-BSDF proposal keeps NEE unbiased

- **WHEN** the environment-importance proposal is enabled alongside the BSDF
  proposal
- **THEN** the NEE MIS companion weight uses the combined mixture pdf, and the
  converged image matches the bsdf-only reference (no bias introduced on the
  direct-lighting term)

### Requirement: Baseline preserves current output exactly

The renderer SHALL, with the default selection (proposal set `{bsdf}` and reuse
mode `none`/identity), produce output pixel-identical to the renderer before this
change, in both megakernel and wavefront execution modes. The identity reuse
mode SHALL forward direct lighting verbatim to the existing `allLightsNEE` and
spawn the indirect ray exactly as today.

#### Scenario: Default selection is byte-identical to today

- **WHEN** the renderer runs with the default proposal set `{bsdf}` and
  `reuse=none` on a fixed scene, seed, and frame count
- **THEN** the output image is pixel-identical to the pre-change renderer, for
  both the megakernel and wavefront backends

### Requirement: Environment-importance proposal

The change SHALL ship a second directional proposal that importance-samples the
environment lighting distribution, reusing the existing environment CDF
descriptor bindings without adding new GPU buffers. It SHALL convert between
world and tangent space internally so the bounce remains tangent-space, and it
SHALL report its exact solid-angle pdf for mixture evaluation. Enabling it
alongside the BSDF proposal SHALL remain unbiased and SHALL reduce variance on
image-based-lit surfaces at low sample counts.

#### Scenario: Environment proposal is unbiased and reduces variance

- **WHEN** the proposal set `{bsdf, env}` renders an IBL-lit scene
- **THEN** the high-sample-count image converges to the same reference as the
  bsdf-only render, and the low-sample-count variance on a glossy or diffuse
  surface is lower than bsdf-only

### Requirement: Reuse seam interface with identity baseline

The change SHALL define the reuse-seam interface and ship the identity
implementation only; reservoir-based resampling (ReSTIR) is out of scope. The
`ReusePlugin` host socket SHALL permit a later plugin to inject compute passes,
per-pixel buffers, and descriptor bindings around the shade stage. Selecting a
reuse mode SHALL be a pass-structural decision applied when the renderer's
passes are built (analogous to the execution-mode selection), not a per-bounce
runtime branch.

#### Scenario: Identity reuse forwards to stock lighting

- **WHEN** the reuse mode is `none`
- **THEN** direct lighting is the stock `allLightsNEE` result and the indirect
  ray is spawned exactly as before, with no extra passes or buffers allocated

### Requirement: Wavefront-primary with megakernel proposal subset

The seam SHALL target the wavefront execution backend as primary: reuse plugins
and any proposal pre-passes SHALL be expressible as wavefront compute passes
with persistent buffers. The inline directional-proposal mixture SHALL also work
in the megakernel backend (the subset that fits a single dispatch). A reuse mode
that requires multiple passes SHALL be available only where the backend supports
it.

#### Scenario: Inline proposal mixture runs in both backends

- **WHEN** an analytic proposal mixture (e.g. `{bsdf, env}`) is selected
- **THEN** it is applied at the bounce in both megakernel and wavefront modes

### Requirement: Command-line, GUI, and persisted selection across all front-ends

The active proposal set and reuse mode SHALL be selectable on the command line
(`--proposals`, `--reuse`, with environment-variable fallbacks), mirroring the
existing render-selection flags, and wired identically into the `skinny`,
`skinny-gui`, and `skinny-web` entry points. The selection SHALL be surfaced in
the interactive UI (including the debug viewport) and persisted in the user
settings snapshot. Changing the proposal set, the proposal weights, or the reuse
mode SHALL reset progressive accumulation.

#### Scenario: Selection flags mirror the other render-selection flags

- **WHEN** the application is launched with `--proposals bsdf,env` (or the
  environment-variable fallback)
- **THEN** that proposal set is active for the session across whichever
  front-end was launched, consistent with how `--integrator` and
  `--execution-mode` behave

#### Scenario: Changing the sampling selection resets accumulation

- **WHEN** the proposal set, a proposal weight, or the reuse mode changes
- **THEN** progressive accumulation resets so the new configuration accumulates
  cleanly

### Requirement: Unbiasedness and PDF contract

All proposal densities SHALL be expressed in solid-angle measure, and the
mixture density SHALL be their alpha-weighted sum. The `ProposalSample` type
SHALL carry a network-version field (baseline value 0) and the contract SHALL
require that a sample's pdf come from the version that produced it, so a future
online-trained proposal can be added without an interface change. A debug build
SHALL be able to assert that non-delta proposal densities are finite and
positive and that each proposal's density integrates to approximately 1 over the
hemisphere.

#### Scenario: Debug build validates proposal densities

- **WHEN** the renderer is built with the sampling-debug option enabled
- **THEN** non-delta proposal pdfs are asserted finite and positive, and a test
  confirms each proposal's pdf integrates to ≈ 1 over the hemisphere

### Requirement: BSDF proposal density is self-consistent across draw and weight

The BSDF directional proposal SHALL draw directions and report densities from a
**single** BSDF model, so that the density used to **draw** a bounce (the
material's `sample()`) equals the density used to **weight** it in the mixture
pdf and in NEE's companion pdf (the material's `evaluate()`), for **all**
materials including layered / coated ones. The proposal mixture's unbiasedness
depends on this equality; a material whose `sample()` and `evaluate()` disagree
SHALL be treated as a defect, not an accepted approximation.

#### Scenario: layered material stays unbiased under the mixture

- **WHEN** the `{bsdf, env}` proposal renders a layered coat+metal material (e.g.
  brass) IBL-only
- **THEN** the converged image matches the bsdf-only and BDPT references with no
  bias, because the drawing density and the weighting density come from the same
  model

#### Scenario: draw-time and weight-time pdf agree

- **WHEN** a non-delta bounce direction `wi` is produced for any flat /
  `std_surface` material
- **THEN** the BSDF proposal's draw-time pdf (`sample().pdf`) and weight-time pdf
  (`evaluate().pdf`) for `(wo, wi)` are equal

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

### Requirement: Proposal with GPU state and a wavefront pre-pass
The scene-sampling seam SHALL support a directional proposal that owns GPU buffers and
descriptor bindings and is produced by a wavefront compute pre-pass writing a per-lane
direction and pdf consumed at the bounce, in addition to purely analytic proposals that
own no GPU state. Selecting such a proposal SHALL allocate its GPU state and deselecting it
SHALL release it, without affecting analytic-only selections, and the bounce SHALL consume
the pre-pass's per-lane `(wi, pdf)` through the same mixture-MIS contract as an inline
proposal.

#### Scenario: Stateful proposal allocates a pass and buffers
- **WHEN** a proposal that owns GPU state is selected on the wavefront backend
- **THEN** the renderer builds its pre-pass and buffers and the bounce consumes the
  precomputed per-lane `(wi, pdf)` via the mixture pdf

#### Scenario: Analytic-only selection allocates nothing
- **WHEN** only analytic proposals such as `{bsdf, env}` are selected
- **THEN** no proposal pre-pass, buffer, or extra descriptor binding is allocated

### Requirement: Mixture unbiasedness under a time-varying proposal density
The scene-sampling mixture-MIS contract SHALL hold when a proposal's density changes
between frames, by keying each sample's density to the network version stamped on it
rather than the proposal's current state, so that asynchronous weight updates change
variance but not the expected value of the estimator.

#### Scenario: Density keyed to the sample's version
- **WHEN** the mixture computes the combined `Σ α_k·pdf_k` for a sample whose neural
  proposal has since been updated
- **THEN** the neural term uses the density of the version that drew the sample, keeping the
  one-sample-MIS weighting consistent

#### Scenario: NEE coupling stays consistent across a swap
- **WHEN** a light-sampled direction is evaluated against the neural proposal density after
  a weight swap
- **THEN** the neural density used for the NEE/MIS coupling is that of the active render-side
  version for that frame, preserving the unbiased coupling

