## Why

In `skinny-gui` (the Qt front-end) the rotate gizmo is display-only. Selecting a
mesh instance in the `SceneGraphDock` sets the gizmo target and draws its three
rings, but the rings cannot be grabbed: `ui/qt/viewport.py`'s `RenderViewport`
mouse handlers cover shift-autofocus, scene-pick, zoom-rect, and camera drag, and
never call `gizmo_hit_test` / `gizmo_begin_drag` / `gizmo_update_drag` /
`gizmo_end_drag` / `gizmo.set_hover`. A left-click on a ring falls straight
through to `self._left = True` and orbits the camera, so the gizmo is decorative.

The drag logic already exists and works in the GLFW front-end
(`app.py:175-188`, `_on_mouse_move:210-214`); it was simply never ported to the
Qt viewport. This change ports it, with hover feedback, and adds a parity guard
so a future render-surface front-end cannot silently ship a display-only gizmo
again.

## What Changes

- **Qt gizmo grab/drag**: `RenderViewport.mousePressEvent` hit-tests the gizmo
  (in render-pixel space, after the modal modes, before camera) and begins a drag
  on a ring hit; `mouseMoveEvent` updates the drag; `mouseReleaseEvent` ends it.
  All renderer gizmo calls hold `self._render_lock`, like the other Qt mouse
  paths.
- **Qt gizmo hover**: `mouseMoveEvent` with no button down sets the hovered ring
  via `gizmo.set_hover(gizmo_hit_test(...))`, so the ring under the cursor
  highlights before grab (`setMouseTracking(True)` is already enabled).
- **Testable dispatch seam**: the press-time decision (which modal mode wins,
  gizmo vs. camera) is factored into a pure helper so it can be unit-tested
  without a `QApplication`.
- **Parity guard**: a test asserts the Qt viewport wires gizmo interaction
  (grab + hover), so the manipulator is never display-only on a render-surface
  front-end.

## Capabilities

### New Capabilities
- `viewport-gizmo-interaction`: direct mouse manipulation of the rotate gizmo in a
  render-surface viewport — grab a ring and drag to rotate the targeted instance,
  with hover highlight and a fixed precedence relative to the modal viewport modes
  and the camera. Specifies parity so every render-surface front-end provides it.

## Impact

- **Code**: `src/skinny/ui/qt/viewport.py` (gizmo hit-test/drag/hover in the three
  mouse handlers + render-pixel mapping + render-lock); a small pure dispatch
  helper (new function, e.g. in `ui/qt/viewport.py` or a sibling module) for the
  press-time precedence decision.
- **Dependencies**: none new. Reuses the existing `RotateGizmo`, the
  `Renderer.gizmo_*` API, and `_widget_to_render_pixel`.
- **Tests**: unit test for the pure dispatch helper (modal-mode precedence; ring
  hit vs. miss → gizmo-drag vs. camera); a parity test asserting the Qt viewport
  calls the gizmo API on press/move/release. GUI itself smoke-tested manually in
  `skinny-gui`.
- **Out of scope** (not filed): GLFW viewport instance picking. GLFW has no
  instance-selection UI, so its gizmo never targets anything; wiring it needs the
  GPU pick buffer to carry an instance id (`HitInfo`/`main_pass.slang` change +
  `main_pass.spv` recompile). Translate/scale gizmos, multi-select, and undo/redo
  are also out of scope.
