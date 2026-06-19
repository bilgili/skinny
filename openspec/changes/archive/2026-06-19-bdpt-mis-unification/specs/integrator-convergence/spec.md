## ADDED Requirements

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
