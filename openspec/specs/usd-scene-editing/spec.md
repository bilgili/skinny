# usd-scene-editing Specification

## Purpose
TBD - created by archiving change runtime-scene-graph-editing. Update Purpose after archive.
## Requirements
### Requirement: Stage is the authoritative model with a non-destructive edit layer

On loading a USD scene the renderer SHALL retain the live `Usd.Stage` as the
authoritative scene model and attach a dedicated edit sublayer
(`<scene>.edits.usda`) as the strongest sublayer of the root layer in memory,
setting it as the stage edit target. The original root layer SHALL NOT be written
to disk by any editing operation. The flat scene and GPU buffers SHALL be a
derived cache that the renderer refreshes from the stage on each edit.

#### Scenario: Edit layer attached at load

- **WHEN** a USD scene is loaded for editing
- **THEN** an edit sublayer is present as the strongest sublayer and is the stage edit target

#### Scenario: Original file untouched by edits

- **WHEN** any editing operation authors to the stage
- **THEN** the opinion is written to the edit layer and the original file on disk is unchanged

### Requirement: Add a model by USD reference

The renderer SHALL provide `add_model(usd_path, parent_prim_path="/World", name=None, transform=None)`
that defines an `Xform` prim under the parent in the edit layer, adds a reference
to `usd_path`, authors the given transform (identity when omitted), bakes only
the newly referenced subtree, appends the resulting instances to the scene,
rebuilds the GPU geometry buffers, and returns the new prim's path. When the
parent prim does not exist it SHALL be created as an `Xform`.

#### Scenario: Added model appears in the scene

- **WHEN** `add_model` is called with a valid USD file path
- **THEN** the returned prim exists on the stage and the scene's instance count increases by the referenced model's instances

#### Scenario: Added model is visible

- **WHEN** a model is added and a frame is rendered
- **THEN** the rendered pixels change relative to before the add

#### Scenario: Added model honors its transform

- **WHEN** `add_model` is given a non-identity transform
- **THEN** the new instances are placed using that transform

### Requirement: Remove a node non-destructively

The renderer SHALL provide `remove_node(prim_path)` that deactivates the prim by
authoring `active = false` in the edit layer, drops every instance under that
prim path from the scene, and rebuilds the GPU geometry buffers. The original
file SHALL remain unchanged.

#### Scenario: Removed node disappears

- **WHEN** `remove_node` is called on a present prim path
- **THEN** the prim is inactive on the stage and its instances are absent from the scene

#### Scenario: Remove rebuilds surviving geometry from cache

- **WHEN** a node is removed
- **THEN** the surviving instances are re-uploaded without re-baking their meshes

### Requirement: Set a node transform with a fast resync

The renderer SHALL provide `set_transform(prim_path, matrix)` that authors
`xformOp:transform` on the prim in the edit layer and refreshes only the affected
instance transforms via the transform-only upload path, without re-baking or
re-concatenating geometry.

#### Scenario: Transform updates instance placement

- **WHEN** `set_transform` is called on a present prim path
- **THEN** the affected instances move to the new transform and no geometry re-bake occurs

### Requirement: Edits target prims by USD prim path

`MeshInstance` SHALL carry the originating `prim_path`, and the renderer SHALL
maintain an index from prim path to its instances, rebuilt whenever the instance
list is rebuilt. The existing `apply_instance_transform` and `apply_node_enabled`
operations SHALL identify their target by prim path rather than by integer
instance index, and all in-repo callers (the rotate gizmo, the ImGui scene-graph
panel, and the Qt scene-graph panel) SHALL pass the prim path. Resolution SHALL
succeed for mesh instances without a built scene graph (via the prim-path index)
and SHALL fall back to the scene-graph node for lights and cameras.

#### Scenario: Instance edits resolve by prim path

- **WHEN** `apply_instance_transform` or `apply_node_enabled` is called with a prim path
- **THEN** the matching instances are updated

#### Scenario: Index stays consistent after geometry edits

- **WHEN** a model is added and then a node removed
- **THEN** the prim-path-to-instance index matches the live instance list

### Requirement: Persist edits on request

The renderer SHALL provide `save_edits(path=None)` that writes the edit layer to
disk (default sibling `<scene>.edits.usda`). Saving SHALL produce a file that can
be reopened as a valid USD layer. No editing operation other than `save_edits`
SHALL write to disk.

#### Scenario: Saved edit layer is reopenable

- **WHEN** edits are made and `save_edits` is called
- **THEN** a USD layer file is written that reopens and contains the authored edits

### Requirement: Enumerate editable nodes

The renderer SHALL provide `list_nodes()` returning the editable prims of the
stage (at minimum each prim path and whether it is active), suitable for
scripting and as the data source for a future scene-graph UI.

#### Scenario: Listing reflects current edits

- **WHEN** a model is added and `list_nodes` is called
- **THEN** the new prim path is present in the result

### Requirement: Edits are atomic and reset accumulation

Editing operations SHALL validate their inputs before mutating the stage and
leave the scene and stage unchanged on failure. `add_model` SHALL raise
`ValueError` when `usd_path` is missing or unreadable, mutating nothing; if
baking the added subtree fails, the authored prim SHALL be removed.
`remove_node` and `set_transform` SHALL raise `ValueError` for an unknown prim
path. Every successful edit SHALL reset progressive accumulation.

#### Scenario: Invalid add leaves scene untouched

- **WHEN** `add_model` is called with a path that does not exist or is not valid USD
- **THEN** a `ValueError` is raised and the stage and scene are unchanged

#### Scenario: Unknown prim path is rejected

- **WHEN** `remove_node` or `set_transform` is called with a prim path that is not on the stage
- **THEN** a `ValueError` is raised and nothing is mutated

#### Scenario: Successful edit resets accumulation

- **WHEN** any edit completes successfully
- **THEN** the progressive accumulation frame counter resets to zero

