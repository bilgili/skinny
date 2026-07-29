# scene-intake Specification

## Purpose

Scene intake is the boundary between a USD stage and the renderer. It reads a
stage and returns a value; the renderer applies that value through one path.
Keeping the direction one-way is the whole point: intake describes what a stage
contains, and the renderer decides what to do about it. A reference in the
other direction makes the two circular at runtime and unassertable without a
GPU.

## Requirements
### Requirement: Scene intake returns a value and holds no renderer reference

Scene intake SHALL expose one interface that reads a USD stage — in full, as a
streamed batch, or at a given time code — and returns a value. A whole-stage
read SHALL return a `SceneUpdate` describing instances, materials, lights,
camera, volume, controls, skeletal bindings and film clamp; a time-code read
SHALL return the animated subset (instance transforms, lights, camera) as its
own value, because a per-frame delta describes no material, volume or control
change and a type that claimed otherwise would be mostly empty by
construction. Intake MUST NOT hold, mutate, or import from the renderer. In
particular, resolving a control binding SHALL return a description of the
target that the renderer applies — intake MUST NOT read the renderer's scene,
call its override methods, set its dirty flags, or import `skinny.params` to
string-path into its attributes. Consumers SHALL NOT reach intake internals:
no function-local import of a loader symbol may appear outside the loader and
intake modules, so the module-level import graph shows the real dependency.

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

The renderer SHALL apply every scene update through one path, whatever
triggered it — initial load, streamed batch, or post-edit resync. That path
SHALL state the adoption order once: scale and film clamp, volume grid sync,
scene material generation, camera framing, control defaults, upload,
default-light injection, camera-node refresh. It SHALL preserve renderer-side
runtime state across a geometry replacement — instance-enabled flags,
light-enabled flags, and live material overrides keyed by source prim path with
a fallback to name — and SHALL NOT carry that state onto a scene that replaces
rather than re-reads the loaded one, since authored values and live edits share
one channel and the outgoing authored value would beat the incoming one.

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

#### Scenario: Losing the scene is a scene change

- **WHEN** the loaded model state is cleared, so the scene and stage are
  dropped without a replacement arriving
- **THEN** the same version counter advances, and the state derived from the
  released stage — controls, animation index, clock, up-axis rotation,
  skeletal handle — is cleared with it

### Requirement: Time-indexed re-read is a call on the interface

Per-frame re-extraction of animated lights, camera and skeletal state SHALL be
a call on the intake interface at a time code, not a re-derivation performed
by the renderer from intake privates. The returned sample SHALL distinguish a
part of the stage that was not read from one that was read and found empty, so
a transform-only re-read cannot clear the scene's lights.

#### Scenario: Animated re-read matches the recorded extraction

- **WHEN** lights and camera are re-read at a sequence of time codes through
  the interface and compared against a recorded capture of what the per-frame
  extraction produced for the same codes
- **THEN** they are identical

#### Scenario: An unread part of the stage is not an empty one

- **WHEN** a caller re-reads transforms at a time code without asking for
  lights
- **THEN** the sample reports that lights were not read, and applying it leaves
  the scene's lights untouched

