## MODIFIED Requirements

### Requirement: Volumetric subsurface transport is unbiased and bounded

The estimator SHALL transport light through the interior as a random walk: refract
at the dielectric boundary, delta-track (Woodcock) the medium with per-channel
`σ_t` and HG scattering, and Fresnel-split internal-reflection vs refraction at the
entry boundary and at every interior face the walk reaches (transmit with
probability `Ft` carrying the environment out, else internally reflect back into
the medium and continue). It SHALL be a single-pdf, bounded-throughput (no clamp),
Russian-roulette-terminated estimator, so the path / BDPT / ReSTIR invariants hold.

On the **wavefront** path the walk SHALL find each segment's boundary by tracing
the actual scene geometry (`traceScene`) from the scatter vertex, so the path
length follows the real mesh rather than a flat-slab approximation. On the
**megakernel** path — where the whole walk runs in one dispatch and a per-segment
trace would risk the GPU watchdog — the walk MAY use the 1D-slab approximation
(perpendicular depth bounded by `hit.backT`). Both modes SHALL conserve energy in
the furnace test and SHALL run on both the Vulkan and native Metal backends.

The interior bounce cap SHALL be large enough to conserve energy across the
optical-depth range typical subsurface scenes hit, subject to the platform
GPU-watchdog limit on the single-dispatch path.

#### Scenario: wavefront walk follows the real geometry

- **WHEN** a curved/complex subsurface object (the sssdragon, a sphere) is rendered
  in wavefront mode
- **THEN** each scattered segment is traced to the actual mesh boundary, and the
  converged radiance is closer to a brute-force / pbrt reference than the flat-slab
  estimator (which over-estimates path length and renders too dark/red)

#### Scenario: energy conservation (furnace) across optical depth

- **WHEN** a homogeneous subsurface sphere with `σ_a → 0` is rendered in a constant
  environment, at thin and thick optical depth (`τ ≈ 1`, `20`, `200`) and across η
- **THEN** the converged result is ~unity at every depth and is η-independent

#### Scenario: distant-lit subsurface uses NEE

- **WHEN** a subsurface object is lit by a distant light (not only the environment)
- **THEN** the walk adds a per-scatter next-event-estimation term — shadow-traced to
  the boundary, attenuated by the in-medium transmittance and the exit Fresnel — so
  the directly-lit response is present and unbiased

#### Scenario: megakernel stays watchdog-safe

- **WHEN** a subsurface scene is rendered in megakernel mode
- **THEN** the walk uses the slab approximation (no per-segment trace) and does not
  trip the macOS GPU watchdog

#### Scenario: non-subsurface scenes are unchanged

- **WHEN** a scene with no subsurface material (the pbrt parity corpus, or a true
  `dielectric` glass) is rendered
- **THEN** the result is byte/behaviour-identical to the pre-change renderer
