## MODIFIED Requirements

### Requirement: Default light is injected only into light-less scenes

The renderer SHALL choose exactly one lighting authority for each scene. When a
loaded USD scene contains any supported authored lighting source, the renderer
SHALL use only lighting from that scene and SHALL synthesize neither its
built-in DistantLight nor its built-in image-based light. When the scene
contains no authored lighting source, the renderer SHALL activate both
fallbacks together: one built-in DistantLight and one built-in IBL.

Authored-light presence SHALL be determined independently of current power or
enabled state. A supported authored DistantLight, SphereLight, DomeLight,
RectLight, or DiskLight SHALL suppress the fallback pair even when disabled or
zero intensity. RectLight and DiskLight SHALL count before their conversion to
emissive geometry, and an explicitly emissive material instance SHALL also
count as authored lighting. The default no-USD session SHALL count as
light-less.

In authored-light mode, missing light types SHALL remain missing: a SphereLight-
only scene SHALL receive no environment fill, and a DomeLight-only scene SHALL
receive no default DistantLight. Hidden or persisted fallback settings,
including environment selection/intensity and the default direct-light toggle,
SHALL NOT alter authored USD lighting. USD-authored light properties and
enabled state remain authoritative. Furnace mode remains an explicit diagnostic
override outside this policy.

The authority decision SHALL apply identically to all front-ends, the headless
API, Vulkan, and Metal, and SHALL be re-evaluated after a stage load or resync
so adding the first authored light removes both fallbacks and removing the last
authored light restores both.

#### Scenario: Any authored USD light suppresses both fallbacks

- **WHEN** a USD scene contains any supported authored light, including a
  SphereLight-only, DomeLight-only, or zero-intensity-light scene
- **THEN** no built-in DistantLight and no built-in IBL contribution is present,
  and the render uses only the lighting authored in the USD

#### Scenario: Light-less scene receives the complete fallback pair

- **WHEN** a USD scene contains geometry but no authored lighting source
- **THEN** exactly one built-in DistantLight and the built-in IBL are active
  together

#### Scenario: Missing authored light types are not filled

- **WHEN** a scene authors a SphereLight but no DomeLight or DistantLight
- **THEN** the SphereLight contributes, the environment is black, and no
  synthesized DistantLight is uploaded

#### Scenario: Fallback state cannot override authored lighting

- **WHEN** a scene with authored DistantLight and DomeLight values is loaded
  while persisted fallback intensity, environment, or direct-light-toggle
  values differ
- **THEN** the rendered lights use the authored USD values and ignore all
  fallback-only values

#### Scenario: Runtime authority transition is atomic

- **WHEN** a stage resync adds its first authored light or removes its last one
- **THEN** the renderer switches atomically between no fallback lights and the
  complete fallback pair, never a mixed one-light fallback state

## ADDED Requirements

### Requirement: Default-light controls follow fallback authority

The renderer-owned `IBL` and `Direct Light` control groups SHALL be visible only
while the built-in fallback pair is active. When USD-authored lighting is
authoritative, both groups SHALL be absent from every shared control UI,
including their section headings, and neither `/Skinny/DefaultLight` nor
`/Skinny/DefaultDome` SHALL be injected into the scene graph. USD-authored light
nodes and their editable properties SHALL remain available in the scene graph.

Control visibility SHALL use the same authority decision as rendering and SHALL
update after asynchronous scene load and runtime stage resync. The headless
`--env-intensity` and `--no-direct` options and their API equivalents SHALL
remain valid fallback controls but SHALL have no effect while authored lighting
is active.

#### Scenario: Authored-light scene hides renderer-owned controls

- **WHEN** a USD scene containing any authored lighting source finishes loading
- **THEN** the IBL and Direct Light fallback sections, widgets, and synthesized
  scene-graph nodes are absent, while authored light nodes remain editable

#### Scenario: Light-less scene shows both fallback control groups

- **WHEN** a scene with no authored lighting source is active
- **THEN** both fallback light groups and both synthesized scene-graph nodes are
  present and control the built-in DistantLight and IBL

#### Scenario: Controls follow a runtime authority transition

- **WHEN** stage resync changes the scene between authored-light and light-less
  states
- **THEN** both fallback control groups and both synthesized nodes disappear or
  reappear together without restarting the front-end
