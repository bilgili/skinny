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

### Requirement: BDPT strategies form a single MIS partition

BDPT SHALL weight every contributing strategy for a given path — the t=0 emissive
eye hit, the t=1 next-event connection, the t≥2 generic connections, and the s=1
light-tracer splat — within one multiple-importance-sampling partition whose
weights sum to 1. No strategy SHALL be added at full weight when another strategy
can also generate the same path. In particular the s=1 splat SHALL be MIS-weighted
(power heuristic) against the eye-side strategies, and the t=1 next-event
connection SHALL use the same `misWeight` partition as the t≥2 connections (not a
standalone 2-strategy heuristic that ignores the t≥2 alternatives).

#### Scenario: s=1 splat is not double-counted on diffuse
- **WHEN** a directly-visible diffuse surface is lit by an area light and the light subpath also splats onto that surface via the s=1 strategy
- **THEN** the splat is weighted ≈0 there (the eye side owns the path), so it is not added on top of the eye-side estimate

#### Scenario: t=1 and t≥2 partition correctly on indirect transport
- **WHEN** an indirect path (e.g. camera → diffuse → diffuse → area light) is reachable by both the t=1 next-event connection and a t≥2 generic connection
- **THEN** their MIS weights sum to 1 for that path, so the indirect contribution is counted once, not over-weighted

### Requirement: BDPT display converges to the path tracer

BDPT's display output SHALL match the unidirectional path tracer in absolute
energy at convergence, differing only by Monte-Carlo noise and by genuine
light-transport features the path tracer is biased against (specular caustics,
which BDPT renders and the path tracer misses). The display output is the
tonemapped image the application shows, including the s=1 splat composite. A
dedicated gate SHALL compare the BDPT display to the path tracer, because the
accumulation-based gates exclude the splat and cannot observe this difference.

#### Scenario: diffuse corpus BDPT display matches the path tracer
- **WHEN** the pure-diffuse area-light corpus scene (no caustics) is rendered to display with BDPT and with the path tracer
- **THEN** their mean display energies match within the gate tolerance (the pre-fix BDPT display was ~1.12× the path tracer)

### Requirement: BDPT walks light subpaths from distant lights

The BDPT light-subpath sampler SHALL emit real light rays from distant
(directional) lights — origin sampled on a disk covering the scene bounds,
direction along the light, with the disk-area factor in the subpath throughput
and the matching area-measure origin pdf — and SHALL launch the light-subpath
random walk from that ray, so distant-light radiance participates in the s ≥ 2
connection strategies and the s = 1 camera splat. Distant-light **specular
caustics** (delta light through a specular chain onto a diffuse receiver), which
unidirectional path tracing cannot sample, SHALL be rendered by BDPT and SHALL
agree with the SPPM photon estimate of the same transport at an equal-time or
explicitly recorded budget using firefly-robust same-budget region statistics.
The extended strategy set SHALL remain a single MIS partition: the eye-side
distant NEE (`connectT1`'s distant branch, today an unconditional full-weight
term) SHALL join the `misWeight` partition against the walked strategies and the
camera splat, and its weight SHALL degrade to exactly full weight when no walked
partner exists, so distant-light direct illumination is neither double-counted
nor dimmed on scenes without specular transport. The first walked vertex's
forward pdf SHALL use the parallel-projection area density of the emission disk
(no distance-squared falloff — the disk placement distance is arbitrary). The
behavior SHALL hold in both the RGB and spectral builds (the spectral build
recoloring the distant emission per-λ via the authored SPD, matching the SPPM
photon emitter), on both backends, and in both execution modes (the megakernel
and wavefront BDPT variants SHALL enable the walk consistently).

#### Scenario: Distant-light glass caustic agrees between BDPT and SPPM

- **WHEN** a scene authoring a DistantLight above a non-dispersive
  delta-transmissive glass object over a diffuse ground is rendered with `bdpt`
  and `sppm` at equal time (or an explicitly recorded high budget), compared via
  firefly-robust same-budget region medians
- **THEN** the caustic region's energy agrees between the two integrators within
  the recorded tolerance, while `path` is recorded as the excluded integrator
  for that component (delta-delta SDS is unsampleable by unidirectional path
  tracing)

#### Scenario: Distant-light direct lighting is not double-counted

- **WHEN** any scene lit by distant lights (with or without specular geometry)
  is rendered with `bdpt` before and after the distant-walk strategy is enabled
- **THEN** directly-lit (non-caustic) regions keep their energy within noise —
  the walked strategies add only transport that eye-side NEE could not sample,
  and full-image energy stays MIS-consistent with the `path` reference wherever
  path can sample the transport

#### Scenario: Spectral BDPT carries the distant walk

- **WHEN** the same distant-caustic scene is rendered with `--spectral bdpt`
- **THEN** the distant-light subpath emission is recolored per hero wavelength
  (authored SPD when present, upsampled RGB otherwise) and the RGB build's
  compiled kernels remain byte-identical under the spectral split guard

