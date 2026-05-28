# usd-driven-control-ui Specification

## Purpose
TBD - created by archiving change usd-driven-control-ui. Update Purpose after archive.
## Requirements
### Requirement: USD-declared controls are discovered at load

The loader SHALL discover control declarations on the stage by finding every
prim with an authored `skinny:ui:type` attribute, parsing each into a control
spec carrying its type, target, label (defaulting to the prim name), slider
range (`min`/`max`/`step`), `choices`, `default`, and `order`. Controls SHALL be
ordered by an authored `order` then by prim path. Malformed declarations
(unknown type, missing required fields) SHALL be skipped with a warning rather
than failing the load.

#### Scenario: Control prims parsed into specs

- **WHEN** a stage with prims carrying `skinny:ui:type` is loaded
- **THEN** one control spec per valid prim is produced, ordered by `order` then prim path

#### Scenario: Label falls back to prim name

- **WHEN** a control prim has no `skinny:ui:label`
- **THEN** its label is the prim's name

#### Scenario: Malformed control is skipped

- **WHEN** a control prim has an unknown `skinny:ui:type` or is missing a required field
- **THEN** it is skipped with a logged warning and the rest of the controls still load

#### Scenario: Stage with no declarations yields none

- **WHEN** a stage authors no `skinny:ui:type` attributes
- **THEN** no control specs are produced

### Requirement: Controls bind via a prefix-typed target

Each control's `target` SHALL select its binding kind by prefix and resolve to
live get/set behavior: `renderer:<path>` and `mtlx:<field>` SHALL route through
the renderer parameter accessors; `material:<name>:<input>` SHALL resolve the
named material and route through the material-override API; `usd:<primPath>.<attr>`
SHALL read and write the named USD attribute. An unresolvable target (unknown
prefix, missing material, missing attribute) SHALL produce an inert control with
a warning rather than an error.

#### Scenario: Renderer-parameter control drives the renderer

- **WHEN** a control with `target = "renderer:env_intensity"` is edited
- **THEN** the renderer's `env_intensity` updates and the change takes effect

#### Scenario: Material-input control drives a material override

- **WHEN** a control with `target = "material:Skin:roughness"` is edited
- **THEN** the named material's `roughness` override is applied

#### Scenario: USD-attribute control writes the stage

- **WHEN** a control with `target = "usd:/Light.inputs:intensity"` is edited
- **THEN** that USD attribute is set on the stage

#### Scenario: Unresolvable target is inert

- **WHEN** a control targets a material or attribute that does not exist
- **THEN** the control is present but inert and a warning is logged, with no crash

### Requirement: Editing a USD-attribute control refreshes live state

When a `usd:` target attribute is edited, the renderer SHALL write the attribute
to the stage and refresh the live-applicable scene state read from the stage —
lights, instance transforms, and the camera — so attributes in that set update in
place. Attributes outside that set SHALL be written but MAY require a reload to
take visible effect.

#### Scenario: Light attribute updates in place

- **WHEN** a `usd:` control edits an authored light's intensity
- **THEN** the rendered illumination reflects the new value without reloading the scene

#### Scenario: Non-live attribute is written but deferred

- **WHEN** a `usd:` control edits an attribute outside the lights/transforms/camera set
- **THEN** the attribute is written to the stage and applies on the next reload

### Requirement: Declared controls appear in the shared UI

The renderer SHALL surface the declared controls as a "Scene Controls" section in
the shared parameter/spec tree so they appear in the Qt panel, the web panel, and
the debug viewport, building a slider, toggle, combo, or color widget per control
according to its type. The section SHALL be shown only when the loaded stage
declares at least one control.

#### Scenario: Controls render across front-ends

- **WHEN** a stage declaring controls is loaded
- **THEN** a "Scene Controls" section with the per-control widgets is present in the Qt panel, web panel, and debug viewport

#### Scenario: No section without declarations

- **WHEN** a stage declares no controls
- **THEN** no "Scene Controls" section is shown

#### Scenario: Widget type matches the declaration

- **WHEN** a control declares `type = "combo"` with `choices`
- **THEN** a combo widget listing those choices is built for it

### Requirement: Authored defaults are applied at load

When a control authors a `skinny:ui:default`, the renderer SHALL apply that value
to the control's target at load so the scene opens in the authored state;
otherwise the widget SHALL reflect the target's current value.

#### Scenario: Default applied on load

- **WHEN** a control authors `skinny:ui:default`
- **THEN** its target is set to that value when the scene loads

#### Scenario: No default reflects current value

- **WHEN** a control authors no default
- **THEN** the widget initializes from the target's current value

