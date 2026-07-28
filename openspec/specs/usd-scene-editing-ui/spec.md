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

Property edits SHALL be routed by the **shared** scene-property dispatcher. A
front-end MUST NOT reimplement that dispatch locally: a copy drifts from the
original silently, and a front-end that returns nothing where the shared
implementation returns a reason string cannot report the failure either. Every
renderer-reference kind the shared dispatcher routes — including every light
kind it maps — SHALL be reachable from every front-end that presents those
properties.

Interaction bindings for a control surface present in more than one front-end
SHALL be reconciled, or each divergence recorded with its reason and asserted by
test. A binding omitted because its key is also used for continuous movement is
not automatically a divergence: where one front-end already serves both uses
from separate channels, the other SHALL do the same rather than drop the
binding.

#### Scenario: Same logic in both front-ends

- **WHEN** the same node is acted on in either front-end
- **THEN** the add-parent resolution, delete-enablement, and transform authoring are identical

#### Scenario: Every routed reference kind is reachable

- **WHEN** a property is edited on a node of any kind the shared dispatcher
  routes — including an environment/dome light
- **THEN** the edit takes effect in every front-end that offers the property,
  and no kind falls through every branch to a silent no-op

#### Scenario: Dispatch is not reimplemented per front-end

- **WHEN** the front-end window modules are searched for local reimplementations
  of the shared property dispatch, its vector-edit helper, or its material
  ancestor lookup
- **THEN** none remain, and each front-end calls the shared implementation and
  surfaces the reason it returns on failure

#### Scenario: Interaction bindings are reconciled or recorded

- **WHEN** the key bindings of a control surface present in more than one
  front-end are compared — including the Camera Debug dock's depth-of-field
  plane toggle, whose key is also a movement key in both front-ends
- **THEN** each divergence is either removed or recorded with its reason, and the
  recorded set is asserted by test

