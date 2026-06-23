## ADDED Requirements

### Requirement: Per-sample radiance clamp from the pbrt film `maxcomponentvalue`

The renderer SHALL support a per-sample radiance clamp equivalent to pbrt's
`Film "rgb" "float maxcomponentvalue"`. When a clamp threshold `C` is active
(`C > 0`), every Monte-Carlo sample's radiance that is accumulated into the film
SHALL be scaled, before accumulation, so that its largest RGB component does not
exceed `C`: a sample with `max(r,g,b) = m > C` is multiplied by `C / m`, which
preserves hue. A sample with `m ≤ C` is unchanged. When no threshold is active
(`C = 0`, the default), accumulation SHALL be byte-identical to the pre-existing
renderer — the clamp adds no cost and changes no pixel.

The threshold SHALL be sourced from the imported pbrt film (`maxcomponentvalue`),
carried through the USD scene metadata to the renderer, and supplied to the GPU as
a single `FrameConstants` scalar. The clamp SHALL apply uniformly across the
integrators that feed the film — the path tracer, BDPT (its camera contribution
and its light-path splat), and SPPM — so that the firefly-suppression matches
pbrt's `RGBFilm::AddSample` regardless of integrator. Changing the threshold SHALL
reset progressive accumulation (it is part of the accumulation state hash).

The clamp SHALL be applied in the same radiance domain as the existing film
exposure: skinny bakes the pbrt `iso`/`imagingRatio` exposure into emitters, so the
clamp threshold compares against the already-exposure-scaled sample radiance, the
same domain pbrt clamps in.

#### Scenario: a scene with `maxcomponentvalue` suppresses fireflies to match pbrt

- **WHEN** a pbrt scene whose film sets `maxcomponentvalue` (e.g. the
  contemporary-bathroom corpus scene, `50`) is imported and rendered
- **THEN** each accumulated sample's max RGB component is at most the threshold, the
  firefly pixels that otherwise dominate the error collapse, and the render matches
  the pbrt reference EXR within the scene's strict parity tolerance

#### Scenario: a scene without `maxcomponentvalue` is unchanged

- **WHEN** a scene whose film does not set `maxcomponentvalue` is rendered
- **THEN** `filmMaxComponent` is `0`, the clamp is a no-op, and the rendered image
  is identical to the renderer without this feature

#### Scenario: the clamp preserves hue

- **WHEN** a sample's radiance exceeds the threshold in one or more components
- **THEN** the whole RGB triple is scaled by a single factor `C / max(r,g,b)` so the
  chromaticity is unchanged and only the magnitude is reduced
