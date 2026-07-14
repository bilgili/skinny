# photon-mapping — SPPM env-direct completeness delta

## ADDED Requirements

### Requirement: SPPM terminal visible points capture complete env direct lighting

The wavefront SPPM eye stage SHALL add the BSDF-sampled env-miss MIS companion at
each terminal diffuse visible point, so that env direct there equals NEE +
BSDF-miss, matching the path tracer. Because the eye terminates at the first
diffuse visible point (photons carry the indirect), the companion of that
vertex's MIS-weighted environment NEE is otherwise never traced by a subsequent
bounce, leaving the env NEE down-weighted with no counterpart and env direct
systematically under-counted under a broad environment. The companion SHALL be
one BSDF sample whose escaped environment radiance is MIS-weighted against the env
NEE. Small analytic lights SHALL be unaffected (their BSDF-sampling pdf is ~0, so
the NEE is already effectively full weight).

#### Scenario: env-lit diffuse surface matches the path anchor

- **WHEN** an env-only scene (a diffuse plane under the dome) renders under
  `--integrator sppm` at a converged, firefly-robust sample count
- **THEN** the flat-plane radiance matches the `(path, wavefront)` reference
  within tolerance (not ~0.75× of it)

#### Scenario: small analytic lights unchanged

- **WHEN** a scene lit only by a sphere/emissive/distant light (no environment)
  renders under `--integrator sppm`
- **THEN** the result is unchanged — the added env-miss contributes nothing
  (the escaped ray carries zero env radiance), so sphere-lit parity is preserved
