# usd-scene-editing-ui (delta)

## MODIFIED Requirements

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
