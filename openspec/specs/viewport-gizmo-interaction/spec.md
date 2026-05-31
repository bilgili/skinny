# viewport-gizmo-interaction Specification

## Purpose

Direct mouse manipulation of the rotate gizmo in a render-surface viewport: grab
a ring and drag to rotate the targeted instance, with hover highlight and a fixed
precedence relative to the modal viewport modes and the camera. Specifies parity
so every render-surface front-end provides it.
## Requirements
### Requirement: Gizmo interaction precedence

On left press the Qt viewport SHALL evaluate the gizmo hit-test after
shift-autofocus, scene-pick arming, and zoom-rect arming, and before camera
control, so those modal interactions are not shadowed by the gizmo and the gizmo
is not shadowed by the camera. This precedence SHALL be identical for every gizmo
mode (rotate or translate, world or local).

#### Scenario: Armed modal mode wins over the gizmo

- **WHEN** zoom-rect (or scene-pick, or shift-autofocus) is armed and the user presses on a gizmo handle
- **THEN** the armed mode runs and no gizmo drag begins

#### Scenario: Gizmo wins over the camera

- **WHEN** no modal mode is armed and the press lands on a gizmo handle
- **THEN** a gizmo drag begins instead of a camera orbit

### Requirement: Gizmo interaction parity across render-surface front-ends

Every render-surface front-end that can target the transform gizmo SHALL provide
gizmo grab, drag, and hover, AND SHALL wire the space key to cycle the gizmo mode,
so neither the manipulator nor the mode switch is display-only. A test SHALL
assert this wiring for each such front-end (currently the Qt viewport and the
GLFW app).

#### Scenario: A front-end is fully wired

- **WHEN** a render-surface front-end hosts the render surface and an instance is targeted
- **THEN** its press/move/release handlers reach the gizmo hit-test, drag, and hover API, and its space-key handler reaches the gizmo mode-cycle

#### Scenario: A display-only gizmo or unbound mode-cycle is a regression

- **WHEN** a render-surface front-end can target the gizmo but its mouse handlers never call the gizmo interaction API, or its space key does not cycle the gizmo mode
- **THEN** the parity test fails

### Requirement: Grab and drag the transform gizmo in the Qt viewport

The Qt render viewport (`RenderViewport`) SHALL let the user grab a transform-gizmo
handle (a rotate ring or a translate axis) with the left mouse button and
manipulate the targeted instance by dragging, matching the GLFW behavior. On left
press, after the modal viewport modes (see precedence requirement), the viewport
SHALL hit-test the active gizmo in render-pixel coordinates; a handle hit SHALL
begin a gizmo drag and a miss SHALL fall through to camera control unchanged.
Subsequent moves during a drag SHALL update the instance's transform (rotation in
rotate modes, position in translate modes) and SHALL NOT move the camera; release
SHALL end the drag. All renderer gizmo calls SHALL be made while holding the
viewport render lock.

#### Scenario: Grab a handle and manipulate

- **WHEN** an instance is targeted and the user left-presses on a gizmo handle
- **THEN** a gizmo drag begins and dragging rotates (rotate modes) or translates (translate modes) the instance along that handle's axis

#### Scenario: Miss falls through to the camera

- **WHEN** the user left-presses away from every gizmo handle
- **THEN** no gizmo drag begins and camera orbit/pan behaves as before

#### Scenario: Release ends the drag

- **WHEN** the user releases the left button during a gizmo drag
- **THEN** the drag ends and later moves orbit the camera again

### Requirement: Gizmo handle hover feedback

The Qt viewport SHALL highlight the active gizmo's handle under the cursor when no
mouse button is held, and SHALL clear the highlight when the cursor moves off all
handles. Hover hit-testing SHALL use render-pixel coordinates and SHALL be
suppressed while a button is held or a gizmo drag is in progress.

#### Scenario: Hover highlights the handle under the cursor

- **WHEN** an instance is targeted and the cursor moves over a handle (ring or axis) with no button down
- **THEN** that handle is highlighted

#### Scenario: Leaving the handles clears the highlight

- **WHEN** the cursor moves off all handles with no button down
- **THEN** no handle is highlighted

### Requirement: Four transform-gizmo modes cycled by the space key

The viewport gizmo SHALL operate in one of four mutually exclusive modes —
rotate-world, rotate-local, translate-world, translate-local — and SHALL expose
the active mode as a single ordered enum. Pressing the space key SHALL advance
the mode to the next in `(index + 1) mod 4`, ordered grouped by type so the two
rotate modes are adjacent and the two translate modes are adjacent
(rotate-world → rotate-local → translate-world → translate-local → rotate-world).
The space key SHALL be ignored while a gizmo drag is in progress. Only the active
mode's gizmo SHALL be drawn and hit-tested at any time.

#### Scenario: Space advances through the four modes

- **WHEN** the active mode is rotate-world and the user presses space (no drag in progress)
- **THEN** the mode becomes rotate-local, and successive presses yield translate-world, then translate-local, then rotate-world again

#### Scenario: Space is ignored mid-drag

- **WHEN** a gizmo drag is in progress and the user presses space
- **THEN** the mode does not change and the drag continues

#### Scenario: Only the active gizmo is shown

- **WHEN** the active mode is a translate mode
- **THEN** the translation gizmo is drawn and hit-tested and the rotation rings are not, and vice versa for a rotate mode

### Requirement: Axis translation gizmo

In a translate mode the gizmo SHALL present three axis handles (X, Y, Z) drawn as
screen-space line segments (arrows) around the targeted instance's pivot. The
viewport SHALL hit-test these axis handles in render-pixel coordinates; grabbing a
handle SHALL begin a translate drag constrained to that single axis, and dragging
SHALL move the instance along that axis only, leaving its rotation and scale
unchanged. A miss SHALL fall through to camera control unchanged.

#### Scenario: Grab an axis and translate

- **WHEN** an instance is targeted in a translate mode and the user left-presses on an axis handle and drags
- **THEN** the instance moves along that one axis and its rotation and scale are unchanged

#### Scenario: Translate miss falls through to the camera

- **WHEN** the user left-presses away from every axis handle
- **THEN** no gizmo drag begins and camera orbit/pan behaves as before

### Requirement: World and local coordinate space

Each gizmo type SHALL provide a world-space variant whose handles align to the
canonical world X/Y/Z axes and a local-space variant whose handles align to the
targeted instance's current orientation (the rotation part of its transform). In
a local mode, rotation SHALL be performed about the instance's local axis and
translation SHALL be performed along the instance's local axis; in a world mode,
about/along the corresponding world axis.

#### Scenario: Local handles follow instance orientation

- **WHEN** the targeted instance is rotated and the active mode is a local mode
- **THEN** the gizmo handles are oriented by the instance's rotation rather than the world axes

#### Scenario: World handles ignore instance orientation

- **WHEN** the targeted instance is rotated and the active mode is a world mode
- **THEN** the gizmo handles remain aligned to the canonical world axes

#### Scenario: Local-axis manipulation

- **WHEN** the user drags a handle in a local mode
- **THEN** the instance rotates about (rotate modes) or translates along (translate modes) its own local axis

### Requirement: World/local glyph hint

The gizmo SHALL render a single-letter hint above the targeted instance's pivot —
`W` in world modes and `L` in local modes — as line segments in the existing
gizmo segment buffer, positioned at a fixed pixel offset above the projected
pivot in screen space and drawn on top of the scene. No additional HUD text line
SHALL be required to communicate the coordinate space.

#### Scenario: Glyph reflects the coordinate space

- **WHEN** the active mode is a world mode
- **THEN** a `W` glyph is drawn above the pivot, and switching to a local mode draws an `L` instead

#### Scenario: Glyph tracks the pivot in screen space

- **WHEN** the camera or instance moves so the pivot reprojects to a new screen position
- **THEN** the glyph stays at the fixed pixel offset above the new projected pivot position

### Requirement: Persisted gizmo mode

The active gizmo mode SHALL be persisted to `~/.skinny/settings.json` and
restored at startup, so the gizmo mode chosen in one session is the initial mode
in the next.

#### Scenario: Mode survives a restart

- **WHEN** the user sets the gizmo to translate-local and exits the application
- **THEN** on the next launch the gizmo's initial mode is translate-local

