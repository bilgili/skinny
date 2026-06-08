## MODIFIED Requirements

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
