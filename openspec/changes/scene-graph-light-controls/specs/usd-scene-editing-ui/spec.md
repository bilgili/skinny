## ADDED Requirements

### Requirement: Add lights from the scene-graph panel

Both the Qt and Panel scene-graph panels SHALL provide an "Add light" control
with actions for DistantLight, SphereLight, DomeLight, RectLight, and DiskLight.
Choosing an action SHALL author a prim of that USD schema into the active edit
layer and SHALL immediately resync the rendered scene and scene-graph tree.

The parent prim SHALL be the selected node's path when that node is group-like
(for example Xform or Scope), otherwise `/World`. The new prim path SHALL be a
valid unique path, and the light SHALL have explicit white color, intensity,
exposure, and type-specific size/angle defaults suitable for subsequent
property editing. The operation SHALL remain non-destructive until "Save edits"
persists the edit layer.

The control SHALL be disabled when no editable USD stage is loaded. An invalid
type or creation/resync failure SHALL surface a non-fatal message, roll back any
partially created prim, and leave the scene unchanged. Qt SHALL execute creation
on the render worker; Panel SHALL execute it under the session render lock.

#### Scenario: Add a light under the selected group

- **WHEN** a group node is selected and the user chooses "Add SphereLight"
- **THEN** a uniquely named SphereLight prim is authored below that group,
  appears in the refreshed tree, and contributes to the rendered scene

#### Scenario: Add a light with no group selected

- **WHEN** no node or a non-group node is selected and the user adds a light
- **THEN** the new light prim is authored below `/World`

#### Scenario: Every supported light type is available

- **WHEN** the user opens the "Add light" control
- **THEN** actions for DistantLight, SphereLight, DomeLight, RectLight, and
  DiskLight are available and route to the matching USD schema

#### Scenario: First authored light takes authority

- **WHEN** a light is added to a previously light-less scene
- **THEN** the new authored light becomes authoritative and both renderer-owned
  fallback-light nodes and contributions disappear together

#### Scenario: Light creation requires an editable scene

- **WHEN** no USD edit layer is active
- **THEN** the "Add light" control is disabled and no mutation is attempted

#### Scenario: Failed light creation is rolled back

- **WHEN** defining or resyncing a new light fails
- **THEN** any partially authored prim is removed, a non-fatal error is shown,
  and the previously active scene remains intact
