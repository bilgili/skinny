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

### Requirement: BDPT counts area-light emission exactly once

The bidirectional path tracer SHALL count an area light's emission exactly once
per path. The `t = 0` strategy — the camera/BSDF eye subpath landing on an
emissive triangle — SHALL add that emission at full weight **only** when no
next-event-estimation (NEE) partner exists for it: the primary/first eye hit
(`s == 2`, where the preceding vertex is the delta camera), a delta /
perfectly-specular bounce into the light (`eye[s - 2].isDelta`), or a scene with
no emissive-triangle NEE (`numEmissiveTriangles == 0`). Otherwise the `t = 1`
strategy (`connectT1`, BDPT's power-heuristic-weighted NEE) SHALL own that area
light, exactly as the unidirectional path tracer gates its BSDF-sampled emissive
emission. The rule SHALL apply on every BDPT surface — the megakernel
(`bdpt.slang`) and the wavefront shade stage (`wavefront_bdpt.slang`).

#### Scenario: Diffuse area-light direct lighting is not double-counted
- **WHEN** a diffuse (non-delta) eye vertex both connects to an area light via `connectT1` NEE and the eye subpath later lands on that same area light
- **THEN** the area light contributes once (via NEE), not once via NEE plus once at full weight via the emissive eye hit

#### Scenario: Directly visible and specular-reached lights are still captured
- **WHEN** the area light is the first eye hit, or is reached by a delta/specular bounce (no NEE partner)
- **THEN** the emissive eye hit is added at full weight, so directly visible lights and specular reflections/caustics of the light are not dropped

### Requirement: BDPT converges to the pbrt reference in absolute energy

For a given scene at convergence, BDPT SHALL produce the same expected absolute
(pre-exposure-alignment) energy as the pbrt v4 reference, differing only in noise,
and SHALL track skinny's own unidirectional path tracer (which already matches the
reference). Because the corpus parity and path-convergence gates exposure-align
before comparing — dividing out a uniform brightness error — a dedicated gate SHALL
compare **un-aligned** mean energy on the area-light corpus scenes, for both the
megakernel and wavefront execution modes.

#### Scenario: diffuse_arealight BDPT energy matches pbrt
- **WHEN** `diffuse_arealight` (purely diffuse, no delta bounces) is rendered headless with BDPT
- **THEN** `mean(bdpt)/mean(ref)` is within the gate tolerance of 1 (post-fix ≈ 0.88; the pre-fix ≈ 1.76 fails), and `mean(bdpt)/mean(path)` ≈ 1

#### Scenario: glass_arealight BDPT energy matches pbrt
- **WHEN** `glass_arealight` (a smooth dielectric under an area light) is rendered headless with BDPT
- **THEN** `mean(bdpt)/mean(ref)` is within tolerance (post-fix ≈ 0.87; the pre-fix ≈ 1.49 fails), with specular transport intact

#### Scenario: Wavefront BDPT energy matches the megakernel
- **WHEN** `diffuse_arealight` is rendered with `--execution-mode wavefront --integrator bdpt`
- **THEN** its absolute energy matches the megakernel result and the pbrt reference within the gate tolerance (the wavefront shade stage is not left double-counting)

#### Scenario: A BDPT over-brightness regression fails the gate
- **WHEN** a future change reintroduces an unweighted area-light strategy in BDPT (e.g. the emissive eye hit at full weight against NEE)
- **THEN** the BDPT absolute-energy gate fails, naming the `mean(bdpt)/mean(ref)` (and `mean(bdpt)/mean(path)`) mismatch

