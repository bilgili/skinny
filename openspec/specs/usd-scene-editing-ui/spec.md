# usd-scene-editing-ui Specification

## Purpose
TBD - created by archiving change scene-graph-editing-ui. Update Purpose after archive.
## Requirements
### Requirement: Add a model from the scene-graph panel

Both the Qt and Panel scene-graph panels SHALL provide an "Add model" control
that opens a USD file picker (reusing the shared `model`-category picker and its
last-directory memory) and calls `add_model`. The parent prim SHALL be the
selected node's path when that node is a group-like prim (e.g. `Xform`/`Scope`),
otherwise `/World`. A non-USD selection or an `add_model` failure SHALL surface a
non-fatal message and leave the scene unchanged.

#### Scenario: Add under a selected group

- **WHEN** a group node is selected and the user picks a USD file via "Add model"
- **THEN** the model is added under the selected node's prim path and appears in the tree

#### Scenario: Add with no group selected

- **WHEN** no node (or a non-group node) is selected and the user adds a model
- **THEN** the model is added under `/World`

#### Scenario: Invalid pick is reported

- **WHEN** the user picks a non-USD file or `add_model` raises
- **THEN** a non-fatal message is shown and the scene is unchanged

### Requirement: Delete a node from the scene-graph panel

Both panels SHALL provide a delete affordance (Qt: tree context-menu item and the
`Delete` key; Panel: a delete button) that calls `remove_node` on the selected
prim. The affordance SHALL be disabled for the pseudo-root and for synthesized
`/Skinny/*` renderer-owned nodes. After deletion the panel SHALL refresh and the
node SHALL no longer render.

#### Scenario: Delete a deletable node

- **WHEN** the user deletes a mesh, light, or camera node
- **THEN** `remove_node` is called, the node stops rendering, and the tree refreshes

#### Scenario: Synthesized nodes are protected

- **WHEN** a synthesized `/Skinny/*` node or the root is selected
- **THEN** the delete affordance is disabled

### Requirement: Save edits from the scene-graph panel

Both panels SHALL provide a "Save edits" control that calls `save_edits` and
reports success or failure. The control SHALL be disabled when no edit layer
exists (no USD scene loaded).

#### Scenario: Save writes the edit layer

- **WHEN** edits have been made and the user invokes "Save edits"
- **THEN** the edit layer is written and a success message is shown

#### Scenario: Save disabled without a scene

- **WHEN** no USD scene is loaded
- **THEN** the "Save edits" control is disabled

### Requirement: Transform edits author to the stage

Per-node transform editing in both panels SHALL author the change to the stage
via `set_transform` (so it is captured in the edit layer and persisted by
"Save edits"), rather than the runtime-only `apply_instance_transform`. The
panel's TRS values SHALL be composed into the matrix `set_transform` expects.

#### Scenario: Transform edit is persisted

- **WHEN** the user edits a node's translate/rotate/scale in the panel and then saves edits
- **THEN** the authored transform is present in the saved edit layer

### Requirement: Editing controls behave consistently across front-ends

The add, delete, save, and transform controls SHALL be available with equivalent
behavior in both the Qt and Panel scene-graph panels, sharing the parent
resolution, deletability, and TRS-to-matrix logic.

Consistency SHALL be structural rather than maintained by hand: the mapping
from a scene property (or material-graph input) to the control it needs SHALL
be declared once in the toolkit-free node spec, and each front-end SHALL
contribute only widget construction for a node type. The two independently
written property-to-widget mappings — Qt's `_build_property_widget` with its
eight helpers and Panel's `_build_scene_prop_widget` — MUST NOT both restate
the prop-type switch, and the shared edit semantics MUST NOT be re-inlined in a
front-end. Both front-ends already route edits through the shared
`apply_scene_property` dispatcher; the seam MUST keep that so, and MUST NOT let
one front-end omit a prop type the other supports.

#### Scenario: Same logic in both front-ends

- **WHEN** the same node is acted on in either front-end
- **THEN** the add-parent resolution, delete-enablement, and transform authoring are identical

#### Scenario: A property type cannot render in only one front-end

- **WHEN** the node spec declares a scene-property or graph-input type
- **THEN** a test asserts it is bound exactly once per front-end, and a type
  handled in one front-end but not the other fails the build

#### Scenario: Shared edit semantics are not re-inlined

- **WHEN** the front-end window modules are searched for locally re-implemented
  copies of the shared edit semantics — notably the fan-out-first guard
- **THEN** none remain, and each front-end calls the shared implementation

#### Scenario: Interaction bindings are reconciled or recorded

- **WHEN** the key and mouse bindings of a control surface present in more than
  one front-end are compared — including the Camera Debug dock, where the Qt
  dock has no Escape binding and the web dock has neither keyboard nor mouse
- **THEN** each divergence is either removed or recorded with its reason, and
  the recorded set is asserted by test

