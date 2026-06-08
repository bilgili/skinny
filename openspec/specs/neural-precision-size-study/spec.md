# neural-precision-size-study Specification

## Purpose
TBD - created by archiving change neural-precision-size-study. Update Purpose after archive.
## Requirements
### Requirement: Build-time-configurable network size
The network size SHALL be selectable at build time without editing shader source — its
coupling layers, spline bins, and MLP hidden width — and every module that runs the
flow (the wavefront pre-pass, the inline inverse, the record entry) SHALL compile
against the selected dimensions consistently.

#### Scenario: A non-default size builds and runs
- **WHEN** a network size other than the default is selected and the neural proposal is
  active
- **THEN** the flow modules compile at those dimensions and the proposal draws samples
  and evaluates pdfs without a size mismatch

#### Scenario: Weights must match the built size
- **WHEN** a baked weights file's architecture differs from the built dimensions
- **THEN** loading reports the mismatch rather than indexing the buffers incorrectly

#### Scenario: Default size is unchanged
- **WHEN** no size override is given
- **THEN** the network uses the shipped default dimensions and behaves identically to
  the pre-change proposal

### Requirement: Selectable inference precision with a fixed-precision spline core
The neural inference SHALL support selecting the floating-point precision of the MLP —
full precision, reduced-precision weight storage, reduced-precision MLP compute, and
an 8-bit weight-storage mode (e4m3) decoded to float in the scalar GEMM — while the
rational-quadratic spline evaluation and the reported solid-angle pdf SHALL remain at
full precision in every mode. The 8-bit storage mode SHALL decode in-shader with no
device feature requirement, so it is portable across the Vulkan, Metal, and MoltenVK
backends.

#### Scenario: Reduced-precision storage halves the weight footprint
- **WHEN** the reduced-precision storage mode is selected
- **THEN** the network weights occupy half the GPU bytes of the full-precision mode and
  inference still produces a valid sample and finite positive pdf

#### Scenario: Reduced-precision compute is exercised
- **WHEN** the reduced-precision compute mode is selected on a device that supports it
- **THEN** the MLP linear layers evaluate in reduced precision while the spline core and
  pdf stay full precision

#### Scenario: 8-bit storage quarters the weight footprint portably
- **WHEN** the fp8 (e4m3) weight-storage mode is selected
- **THEN** the network weights occupy a quarter of the full-precision GPU bytes, the
  shader decodes them to float in the scalar GEMM without requiring any device
  capability, and inference still produces a valid sample and finite positive pdf

#### Scenario: Default precision is unchanged
- **WHEN** no precision override is given
- **THEN** inference runs at full precision, identical to the pre-change proposal

### Requirement: Graceful fallback when reduced precision is unsupported
The renderer SHALL detect whether the device supports reduced-precision shader compute
and storage, and SHALL fall back to full precision (reporting the fallback) rather than
failing when a reduced-precision mode is requested on an unsupporting device.

#### Scenario: Unsupported device falls back
- **WHEN** a reduced-precision mode is requested but the device lacks the capability
- **THEN** the renderer runs the proposal at full precision and reports the fallback

### Requirement: Unbiased composition across precision modes
Enabling any precision mode SHALL preserve the mixture's unbiasedness, so the converged
image with the neural proposal matches the full-precision-reference converged image
within noise regardless of the selected precision.

#### Scenario: Reduced precision stays unbiased
- **WHEN** the neural proposal renders a scene to a high sample count in a
  reduced-precision mode
- **THEN** the converged image matches the full-precision / BSDF-only reference within
  noise

### Requirement: Size×precision quality-vs-cost study
The project SHALL provide a study that, across a bounded grid of network sizes and
precision modes, measures quality (pdf-parity drift against the full-precision
reference for the precision axis, held-out training likelihood for the size axis, plus
an in-renderer unbiased-and-firefly check) against cost (inference time per frame on the
Metal/MoltenVK backend and weight-buffer bytes), and SHALL report which grid cells were
run.

#### Scenario: The study emits a quality-vs-cost table
- **WHEN** the study harness is run over the configured grid
- **THEN** it produces, for each evaluated cell, the quality and cost measurements and a
  record of which cells were covered

#### Scenario: Reduced-precision drift is reported, not hidden
- **WHEN** a precision mode diverges from the full-precision reference beyond the
  full-precision tolerance
- **THEN** the measured drift is reported as a study result rather than failing silently

