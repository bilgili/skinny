## ADDED Requirements

### Requirement: Environment maps are imported and sampled as float, never requantized to 8-bit

The importer SHALL NOT requantize a referenced `.exr`/`.pfm` infinite-light map to a
Radiance RGBE (8-bit mantissa) intermediate. The dome loader SHALL read `.exr`/`.pfm`
maps directly as float32 (alongside `.hdr`, which is decoded from RGBE to float32 at
load), so the reprojected float radiance reaches the GPU `RGBA32F` environment texture
with no 8-bit hop. A constant infinite light (no map file) MAY keep its RGBE/intensity
path. The result SHALL match pbrt, which keeps `.exr` as half/float and samples all
environment maps as floating-point radiance.

#### Scenario: a .exr infinite-light map loads without an RGBE round-trip

- **WHEN** a pbrt `infinite` light references a `.exr` (or `.pfm`) environment map
- **THEN** the imported scene's dome loads that map as float32 (reprojected
  equal-area → equirect as needed) and uploads it to the `RGBA32F` env texture
  without writing or reading an intermediate 8-bit RGBE `.hdr`

#### Scenario: high-dynamic-range pixels are preserved to float precision

- **WHEN** an env map contains pixels far brighter than 1.0 (e.g. a small bright sun)
- **THEN** the sampled GPU radiance reproduces those values at float precision (no
  ≥1% mantissa quantization vs the source), matching pbrt's float sampling

### Requirement: Environment maps are resampled with float bilinear filtering at adequate resolution

The importer / loader SHALL resample an environment map with bilinear (area-weighted)
filtering rather than nearest-neighbour, and SHALL NOT cap a high-resolution map at a
fixed low internal resolution that discards energy. A high-frequency env map SHALL
reproduce pbrt's float bilinear lookup within the parity tolerance.

#### Scenario: a high-frequency env map keeps its energy

- **WHEN** a high-resolution env map with a small bright feature is imported and an
  env-lit scene is rendered
- **THEN** the absolute and exposure-blind error against the pbrt v4 reference is
  within the env-fidelity parity tolerance, with no nearest-neighbour aliasing of the
  bright feature
