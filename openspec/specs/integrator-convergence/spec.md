# integrator-convergence Specification

## Purpose
TBD - created by archiving change path-bdpt-convergence. Update Purpose after archive.
## Requirements
### Requirement: Path tracer captures specular paths to area lights

The path tracer SHALL contribute, at full weight, the emission of an emissive
triangle (area light) hit by a BSDF-sampled ray after a delta / perfectly-specular
bounce. The delta condition is the BSDF sample pdf being non-positive (`pdf <= 0`),
covering both mirror reflection and refraction off a smooth dielectric. This makes
the unidirectional path tracer unbiased for specular-to-emitter transport (the
reflection of an area light in a glass surface, and the specular leg of a caustic).

#### Scenario: Area-light reflection in a smooth dielectric is captured
- **WHEN** a camera ray specularly reflects off a smooth dielectric and the reflected ray hits an area light
- **THEN** the path estimate includes that area light's emission (the reflection is visible), rather than dropping it

#### Scenario: Specular caustic leg is captured
- **WHEN** a path reaches an area light through a delta refraction (e.g. floor → BSDF ray → glass refract → light)
- **THEN** the emission is added at full weight, so the caustic accumulates instead of being zero

### Requirement: No double-counting of area-light emission

Area-light emission SHALL be counted exactly once. A non-delta (`pdf > 0`) bounce
continues to receive the area light only through next-event estimation; its
BSDF-hit emission is NOT additionally added. A delta (`pdf <= 0`) bounce — which
has no NEE partner — receives it only through the BSDF-hit emission. The two cases
are mutually exclusive.

#### Scenario: Diffuse bounce is not double-counted
- **WHEN** a diffuse (non-delta) surface both runs NEE toward an area light and a BSDF ray happens to hit that light
- **THEN** the area light contributes once (via NEE), not twice

#### Scenario: Energy is conserved on a closed white scene
- **WHEN** an energy-conservation (furnace/white-room) scene is rendered after the fix
- **THEN** total measured energy matches the pre-fix value within noise (no spurious gain)

### Requirement: Path tracer converges to the pbrt reference

For a given scene at convergence, the unidirectional path tracer SHALL produce
the same expected image as the pbrt v4 reference, differing only in noise. The
reference is the checked-in pbrt corpus EXR (`tests/pbrt/corpus/refs/`), not
BDPT: skinny's BDPT is measured ~1.7× too bright versus pbrt even on a purely
diffuse scene (a separate normalization bug, see the design note), so it cannot
anchor the convergence gate. Specifically on `glass_arealight` the path tracer's
exposure-aligned relMSE / FLIP versus the pbrt reference SHALL be within the
corpus manifest tolerance, and the caustic and area-light reflection SHALL be
present.

#### Scenario: glass_arealight path matches the pbrt reference
- **WHEN** `glass_arealight` is rendered headless with the path tracer at a fixed high spp
- **THEN** exposure-aligned relMSE / FLIP versus the pbrt reference EXR are within the corpus tolerance (FLIP ≈ 0.025 post-fix, vs ≈ 0.058 pre-fix)

#### Scenario: Divergence regression fails the gate
- **WHEN** a future change reintroduces a specular-emitter bias in the path tracer
- **THEN** the path-vs-reference convergence test fails (FLIP/relMSE rise past tolerance), naming the mismatch

### Requirement: Non-specular transport is unchanged

The fix SHALL only affect paths whose spawning bounce was delta. A scene with no
delta bounces SHALL render identically (within Monte-Carlo noise) before and after
the change.

#### Scenario: Diffuse-only scene unchanged
- **WHEN** a purely diffuse scene is rendered before and after the fix at the same seed/spp
- **THEN** the two path-traced images match within noise

### Requirement: Fix applies to all path-integrator surfaces

The specular-emitter emission rule SHALL be applied consistently across every
path-integrator surface: the megakernel path (`path.slang`), the record-dump
variant (`path_record.slang`), and the wavefront path. All SHALL converge to the
pbrt reference.

#### Scenario: Wavefront path also converges
- **WHEN** `glass_arealight` is rendered with `--execution-mode wavefront --integrator path`
- **THEN** it converges to the pbrt reference within tolerance (the wavefront path is not left biased)

