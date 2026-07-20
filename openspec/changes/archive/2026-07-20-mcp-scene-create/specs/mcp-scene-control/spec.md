## ADDED Requirements

### Requirement: Editable stage creation without a pre-loaded file

The server SHALL expose a `scene_create` tool that gives the renderer a fresh,
editable in-memory USD stage so a client can begin structural editing
(`scene_add_*`, `scene_save`) without a scene having been loaded from disk first.

The synthesized stage SHALL contain a single `/World` Xform declared as the stage
default prim, with `upAxis = Y` and `metersPerUnit = 1`, and SHALL carry a
non-destructive edit target so subsequent structural edits author into an edit
layer rather than any file. It SHALL NOT author geometry, lights, or a camera:
the renderer's synthetic default light, dome, and camera nodes SHALL appear in
the resulting scene graph exactly as they do for a loaded scene, so the created
scene is lit and enumerable without further calls.

`scene_create` SHALL run through the same structural execution path as the other
mutation tools (returning a result when it settles within the grace period, else
a pollable job id), and SHALL return the scene version counters. It SHALL NOT
return a path, because the synthesized stage is anonymous until `scene_save`
writes it to an allowed root.

#### Scenario: Create then add on a renderer with no loaded scene

- **WHEN** a client calls `scene_create` against a renderer that has no editable
  stage, then calls `scene_add_primitive`
- **THEN** `scene_create` succeeds and the subsequent add authors under `/World`
  rather than failing with "no editable USD stage is loaded"

#### Scenario: Created scene is enumerable and lit

- **WHEN** a client calls `scene_create` and then `scene_list`
- **THEN** the enumeration resolves and includes `/World` and the synthetic
  default light, dome, and camera nodes

#### Scenario: Result carries versions, not a path

- **WHEN** `scene_create` completes
- **THEN** its result reports the scene version counters and SHALL NOT include a
  `path`, since nothing has been written to disk

### Requirement: scene_create refuses to discard an existing editable stage

`scene_create` SHALL refuse when an editable stage is already present, returning
an error rather than silently replacing the stage and discarding unsaved
structural edits. The refusal SHALL be overridable by a `force` argument
(default false); with `force = true` the tool SHALL replace the current stage
with a fresh empty one.

#### Scenario: Refused when a scene is already loaded

- **WHEN** a client calls `scene_create` (without `force`) while an editable
  stage is loaded
- **THEN** the tool returns an error stating a scene is already loaded, and the
  current stage and its unsaved edits are left intact

#### Scenario: Forced replacement

- **WHEN** a client calls `scene_create` with `force = true` while an editable
  stage is loaded
- **THEN** the current stage is replaced by a fresh empty `/World` stage
