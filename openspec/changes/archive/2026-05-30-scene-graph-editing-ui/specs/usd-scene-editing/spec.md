## MODIFIED Requirements

### Requirement: Remove a node non-destructively

The renderer SHALL provide `remove_node(prim_path)` that deactivates the prim by
authoring `active = false` in the edit layer and re-reads the stage so the node
stops contributing to the render. The re-read SHALL cover meshes (instances),
distant/sphere lights, and the camera, so removing any of these prims drops it
from the scene. The original file SHALL remain unchanged.

#### Scenario: Removed mesh disappears

- **WHEN** `remove_node` is called on a present mesh prim path
- **THEN** the prim is inactive on the stage and its instances are absent from the scene

#### Scenario: Removed light disappears

- **WHEN** `remove_node` is called on a present light prim path
- **THEN** the prim is inactive and the light no longer contributes to the render

#### Scenario: Remove rebuilds surviving geometry from cache

- **WHEN** a node is removed
- **THEN** the surviving instances are re-uploaded without re-baking their meshes

## ADDED Requirements

### Requirement: Geometry edits refresh the derived scene graph

After `add_model` and `remove_node`, the renderer SHALL rebuild the derived scene
graph (`_scene_graph`, including the synthesized default-light injection) and bump
its version counter, so observers that poll the scene graph (the UI panels)
repaint to reflect the change.

#### Scenario: Scene graph reflects an add

- **WHEN** `add_model` completes
- **THEN** the rebuilt scene graph contains the new prim and the version counter has increased

#### Scenario: Scene graph reflects a remove

- **WHEN** `remove_node` completes
- **THEN** the rebuilt scene graph omits the removed prim and the version counter has increased

### Requirement: Lights carry prim-path identity preserved across resync

`LightDir` and `LightSphere` SHALL carry the originating `prim_path`. When a
geometry edit re-reads lights from the stage, the renderer SHALL preserve each
light's runtime `enabled` flag by matching prim path, so a user toggle is not
lost by an unrelated add/remove. Synthesized `/Skinny/*` default lights SHALL be
re-injected rather than dropped.

#### Scenario: Light enable survives an unrelated edit

- **WHEN** a light is toggled off and then an unrelated model is added
- **THEN** the light remains off after the resync

#### Scenario: Default lights survive a resync

- **WHEN** a scene relying on synthesized default lights undergoes a geometry edit
- **THEN** the default lights are still present in the rebuilt scene graph
