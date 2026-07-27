# usd-scene-editing-ui (delta)

## MODIFIED Requirements

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
front-end, as the fan-out-first guard currently is at two Panel sites.

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
  one front-end are compared — including the Camera Debug dock, where Qt is
  currently missing the depth-of-field-plane binding and Panel has neither
  keyboard nor mouse
- **THEN** each divergence is either removed or recorded with its reason, and
  the recorded set is asserted by test
