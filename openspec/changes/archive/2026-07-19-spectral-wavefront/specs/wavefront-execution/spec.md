# Wavefront execution — spectral carrier delta

## ADDED Requirements

### Requirement: Wavefront path-state carries a spectrum under the spectral define

The wavefront's GPU-resident path-state and subpath carriers SHALL, under the
`SKINNY_SPECTRAL` compile-time variant, transport radiance/throughput as a hero-wavelength
spectrum and carry the path's sampled wavelengths, so that transport across the staged
dispatches is per-wavelength. The color-carrying fields SHALL use the shared `Spectrum` typealias (which is
`float3` in the RGB build and `float4` in the spectral build) so that the non-spectral stream
record layout, buffer sizing, and compiled SPIR-V are unchanged; the added per-lane sampled
wavelengths SHALL appear only under the spectral define. The Python stream-layout mirrors and
both the scalar and Metal (MSL) stride checks SHALL match the shader structs in both variants.

#### Scenario: RGB stream layout unchanged

- **WHEN** the wavefront kernels and stream buffers are built without the spectral define after
  the change lands
- **THEN** the path-state record stride, queue buffer sizes, and each non-spectral wavefront
  `.spv` are identical to their pre-change form

#### Scenario: spectral stream layout matches host and both backends

- **WHEN** the wavefront kernels are built with the spectral define
- **THEN** the widened path-state stride (Spectrum carriers plus sampled wavelengths) matches
  the `wavefront_layout` mirrors and both the scalar and MSL stride asserts, on Vulkan and Metal

#### Scenario: film resolves per wavelength at the resolve stage

- **WHEN** a spectral wavefront path folds a completed lane into the accumulation image
- **THEN** the lane's spectral radiance is resolved to linear sRGB via the shared CIE film
  resolve before the accumulation write, using the wavelengths drawn at that path's start
