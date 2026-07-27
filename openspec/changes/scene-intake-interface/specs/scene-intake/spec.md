# scene-intake (delta)

## ADDED Requirements

### Requirement: Scene intake returns a value and holds no renderer reference

Scene intake SHALL expose one interface that reads a USD stage — in full, as a
streamed batch, or at a given time code — and returns a `SceneUpdate` value
describing instances, materials, lights, camera, volume, controls, skeletal
bindings and film clamp. Intake MUST NOT hold, mutate, or import from the
renderer: the current back-reference in `resolve_control_binding` (reading
`renderer._usd_scene`, calling `renderer.apply_material_override`, setting
`renderer._usd_live_dirty`, and importing `skinny.params` to string-path into
renderer attributes) SHALL be inverted so intake returns a binding description
that the renderer applies. Consumers SHALL NOT reach intake internals: the 15
function-local imports of 9 private loader symbols in `renderer.py` are
removed, with each symbol either promoted into the interface or folded into
the update.

#### Scenario: Intake is importable and assertable without a renderer

- **WHEN** the intake interface is used to read a synthetic stage in a test
  process with no `Renderer` instance and no GPU device
- **THEN** it returns a `SceneUpdate` whose contents can be asserted directly

#### Scenario: No lazy reach into intake internals

- **WHEN** the source tree is searched for function-local imports from the
  loader module
- **THEN** none remain, and the module-level import graph shows the real
  dependency

#### Scenario: Control bindings resolve without touching the renderer

- **WHEN** a USD-authored control binding is resolved
- **THEN** intake returns a description of the target and value coercion, the
  renderer performs the override, and the user-visible control behaviour is
  unchanged

### Requirement: One application path adopts every scene update

The renderer SHALL apply scene updates through one path, replacing the three
adoption paths that exist today (`set_usd_scene`, the streaming poll, and the
post-edit geometry resync), each of which currently performs a different
subset of adoption work in a different order. The single path SHALL state the
adoption order once — film clamp, volume grid sync, scene material generation,
camera framing, control defaults, default-light injection, camera-node refresh
— and SHALL preserve renderer-side runtime state across a geometry
replacement: instance-enabled flags, light-enabled flags, and live material
overrides keyed by source prim path with a fallback to name.

#### Scenario: Runtime state survives a stage re-read

- **WHEN** geometry is replaced by a post-edit re-read while instances are
  disabled, lights are disabled, and material overrides are live
- **THEN** all three are still in effect afterwards, asserted directly rather
  than as a side effect of one specific adoption path

#### Scenario: Adoption order is stated once

- **WHEN** a scene arrives by initial load, by streamed batch, or by post-edit
  re-read
- **THEN** the same application path runs, and any step that is deliberately
  specific to one trigger is expressed as a field of the update rather than as
  a different code path

#### Scenario: Scene-change detection is explicit

- **WHEN** the UI needs to detect that the scene was replaced
- **THEN** it reads an explicit version counter, not the object identity of
  the renderer's scene attribute

### Requirement: Time-indexed re-read is a call on the interface

Per-frame re-extraction of animated lights, camera and skeletal state SHALL be
a call on the intake interface at a time code, not a re-derivation performed
by the renderer from intake privates. The values produced MUST be identical to
those the current per-frame extraction produces for the same time code.

#### Scenario: Animated re-read matches the current extraction

- **WHEN** lights and camera are re-read at a sequence of time codes through
  the interface and compared against the values the pre-change per-frame
  extraction produced for the same codes
- **THEN** they are identical
