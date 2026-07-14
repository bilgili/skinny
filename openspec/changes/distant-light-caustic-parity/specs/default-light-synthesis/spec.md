## ADDED Requirements

### Requirement: Default light is injected only into light-less scenes

The renderer SHALL inject its built-in (synthesized) default DistantLight into a
loaded USD scene only when the scene authors **no powered light at all** — no
DistantLight or SphereLight with non-zero power, no emissive-material mesh, and
no authored DomeLight (detected as the loader's `scene.environment`; the
renderer's built-in HDRI backdrop SHALL NOT count as an authored light). A scene
that authors any powered light SHALL render with exactly its authored lights on
every front-end (interactive, headless CLI, web, Qt): the per-frame
distant-light mirror SHALL upload zero synthesized records for such a scene.
Truly unlit scenes — including scenes authoring only zero-power lights — and the
default no-USD session SHALL keep the slider-driven default light unchanged. The
policy SHALL be derived from the load-time scene; live scene-graph edits need
not re-derive it (documented limitation). The `direct_light_index` global off
switch SHALL retain its existing semantics on top of this policy.

#### Scenario: Authored SphereLight suppresses the phantom sun

- **WHEN** a USD scene authoring only a SphereLight (e.g. the glass-caustics
  scene) is rendered on the headless CLI with default options
- **THEN** `FrameConstants.numDistantLights` is 0 (no synthesized DistantLight is
  uploaded), and the rendered image contains no lighting contribution from the
  built-in default light

#### Scenario: Light-less scene keeps the default light

- **WHEN** a USD scene authoring geometry but no light of any kind is loaded
- **THEN** the synthesized default DistantLight is uploaded exactly as before,
  and the light sliders continue to drive it

#### Scenario: All integrators agree once the phantom sun is gone

- **WHEN** the glass-caustics scene (authored SphereLight only) is rendered with
  `path`, `bdpt`, and `sppm` at matched sample counts
- **THEN** `bdpt` matches `path` in full-image energy within tolerance, and the
  SPPM render contains no bright-speckle local fireflies in the shadow/penumbra
  ground at the default SPPM initial radius (the previous speckle was entirely
  the phantom light's caustic)
