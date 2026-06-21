## MODIFIED Requirements

### Requirement: Volumetric subsurface transport is unbiased and bounded

The estimator SHALL transport light through the interior as a random walk: refract
at the dielectric boundary, delta-track (Woodcock) the medium with per-channel
`σ_t` and HG scattering, and Fresnel-split internal-reflection vs refraction on
boundary hits — **at both the entry boundary and at every interior face the walk
reaches**: on reaching a face the walk SHALL transmit with probability `Ft` (carry
the environment out) and otherwise internally reflect back into the medium and
continue, rather than discarding the reflected fraction. It SHALL be a single-pdf,
bounded-throughput (no clamp), Russian-roulette-terminated estimator, so the path
/ BDPT / ReSTIR invariants hold. It SHALL run in both megakernel and wavefront
execution modes and on both the Vulkan and native Metal backends.

The interior bounce cap SHALL be large enough to conserve energy across the
optical-depth range typical subsurface scenes hit (a high-albedo walk needs on the
order of `τ²` scatter events to escape a medium of optical depth `τ`), subject to
the platform GPU-watchdog limit on the Metal single-dispatch path.

#### Scenario: energy conservation (furnace) across optical depth

- **WHEN** a homogeneous subsurface sphere with `σ_a → 0` is rendered in a constant
  environment, at thin **and** thick optical depth (e.g. `τ ≈ 1`, `20`, `200`)
- **THEN** the converged result is ~unity at every depth (energy is neither created
  nor lost) — it does NOT collapse at high `τ` — and the albedo inversion
  round-trips within tolerance

#### Scenario: boundary energy is conserved, not discarded

- **WHEN** a subsurface object with a dielectric boundary (`η > 1`) is rendered
- **THEN** the internally-reflected fraction at each boundary escape is reflected
  back into the medium (not dropped), so the converged radiance is independent of
  whether the path happens to reach a face at a high-Fresnel / total-internal-
  reflection angle

#### Scenario: path tracer and BDPT agree

- **WHEN** a subsurface sphere is rendered under the path tracer and BDPT, in both
  execution modes
- **THEN** both converge to the same radiance per pixel within noise, on both
  backends

#### Scenario: non-subsurface scenes are unchanged

- **WHEN** a scene with no subsurface material (the pbrt parity corpus, or a true
  `dielectric` glass) is rendered
- **THEN** the result is byte/behaviour-identical to the pre-change renderer — the
  flat opacity/refraction path is untouched and the corpus parity is unchanged
