## ADDED Requirements

### Requirement: Grab and drag the rotate gizmo in the Qt viewport

The Qt render viewport (`RenderViewport`) SHALL let the user grab a rotate-gizmo
ring with the left mouse button and rotate the targeted instance by dragging,
matching the existing GLFW behavior. On left press, after the modal viewport modes
(see precedence requirement), the viewport SHALL hit-test the gizmo in render-pixel
coordinates; a ring hit SHALL begin a gizmo drag and a miss SHALL fall through to
camera control unchanged. Subsequent moves during a drag SHALL update the
instance's rotation and SHALL NOT move the camera; release SHALL end the drag. All
renderer gizmo calls SHALL be made while holding the viewport render lock.

#### Scenario: Grab a ring and rotate

- **WHEN** an instance is targeted and the user left-presses on a gizmo ring
- **THEN** a gizmo drag begins and dragging rotates the instance about that ring's axis

#### Scenario: Miss falls through to the camera

- **WHEN** the user left-presses away from every gizmo ring
- **THEN** no gizmo drag begins and camera orbit/pan behaves as before

#### Scenario: Release ends the drag

- **WHEN** the user releases the left button during a gizmo drag
- **THEN** the drag ends and later moves orbit the camera again

### Requirement: Gizmo interaction precedence

On left press the Qt viewport SHALL evaluate the gizmo hit-test after
shift-autofocus, scene-pick arming, and zoom-rect arming, and before camera
control, so those modal interactions are not shadowed by the gizmo and the gizmo
is not shadowed by the camera.

#### Scenario: Armed modal mode wins over the gizmo

- **WHEN** zoom-rect (or scene-pick, or shift-autofocus) is armed and the user presses on a ring
- **THEN** the armed mode runs and no gizmo drag begins

#### Scenario: Gizmo wins over the camera

- **WHEN** no modal mode is armed and the press lands on a ring
- **THEN** a gizmo drag begins instead of a camera orbit

### Requirement: Gizmo ring hover feedback

The Qt viewport SHALL highlight the gizmo ring under the cursor when no mouse
button is held, and SHALL clear the highlight when the cursor moves off all
rings. Hover hit-testing SHALL use render-pixel coordinates and SHALL be
suppressed while a button is held or a gizmo drag is in progress.

#### Scenario: Hover highlights the ring under the cursor

- **WHEN** an instance is targeted and the cursor moves over a ring with no button down
- **THEN** that ring is highlighted

#### Scenario: Leaving the rings clears the highlight

- **WHEN** the cursor moves off all rings with no button down
- **THEN** no ring is highlighted

### Requirement: Gizmo interaction parity across render-surface front-ends

Every render-surface front-end that can target the rotate gizmo SHALL provide
gizmo grab, drag, and hover, so the manipulator is never display-only. A test
SHALL assert this wiring for each such front-end (currently the Qt viewport).

#### Scenario: The Qt viewport is wired

- **WHEN** the Qt viewport hosts the render surface and an instance is targeted
- **THEN** its press/move/release handlers reach the gizmo hit-test, drag, and hover API

#### Scenario: A display-only gizmo is a regression

- **WHEN** a render-surface front-end can target the gizmo but its mouse handlers never call the gizmo interaction API
- **THEN** the parity test fails
