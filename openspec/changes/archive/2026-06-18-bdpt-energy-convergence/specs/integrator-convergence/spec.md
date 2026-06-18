## ADDED Requirements

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
