## Context

The rotate gizmo (`gizmo.py` `RotateGizmo`) does CPU-side hit-testing and drag
math; the renderer exposes `set_gizmo_target`, `gizmo_hit_test`,
`gizmo_begin_drag`, `gizmo_update_drag`, `gizmo_end_drag`, and `gizmo.set_hover`.
Both the *draw* path (`_refresh_gizmo_segments`) and the renderer-side hit-test
use `self.camera`, so screen geometry is consistent between what is drawn and
what is grabbed — no view/proj mismatch to resolve here.

Two front-ends host a live render surface:
- **GLFW** (`app.py`): the window *is* the render surface, so `InputHandler`
  passes raw cursor pixels straight to the gizmo. It wires grab/drag/hover and
  release (`_on_mouse_button`, `_on_mouse_move`). It has **no** instance-selection
  UI, so nothing calls `set_gizmo_target` and the gizmo never appears.
- **Qt** (`ui/qt/viewport.py` `RenderViewport`): the render image is blitted into
  a `QWidget` (`window=None`; no GLFW window). `SceneGraphDock` selection calls
  `set_gizmo_target`, so the gizmo *draws*, but the viewport's mouse handlers
  never call the gizmo API — the reported bug.

This change ports the GLFW interaction to the Qt viewport. It cannot copy the
GLFW code verbatim for two reasons: Qt runs the renderer on a worker thread
behind `self._render_lock`, and Qt widget coordinates differ from render pixels.

## Goals / Non-Goals

**Goals:**
- In `skinny-gui`, grab a gizmo ring and drag to rotate the targeted instance.
- Hover highlight on the ring under the cursor (no button down).
- Gizmo precedence: after shift-autofocus / scene-pick / zoom-rect arming, before
  camera control — matching GLFW.
- A parity guard so a render-surface front-end cannot ship a display-only gizmo.

**Non-Goals:**
- GLFW viewport instance picking (needs a pick-buffer instance id + shader
  recompile). Not filed.
- Translate / scale gizmos, multi-select, undo/redo.
- Any change to gizmo math, the renderer gizmo API, or the draw path.

## Decisions

### D1: Mirror GLFW press precedence, gizmo before camera
On left press the Qt viewport evaluates, in order: shift-autofocus → scene-pick
(`_pick_armed`) → zoom-rect arming (`_zoom_arming`) → **gizmo hit-test** → camera.
A ring hit calls `gizmo_begin_drag` and marks an in-progress gizmo drag; a miss
falls through to the existing `self._left = True` camera path. This matches
`app.py`, where the gizmo sits after zoom-arm and before camera, so existing
camera and zoom behavior is unchanged.

### D2: Map to render pixels first
`gizmo_hit_test` / `gizmo_begin_drag` expect render-pixel coordinates (top-left
origin, matching `RWTexture2D` indexing). Qt handlers already map widget → render
pixels with `_widget_to_render_pixel(x, y)`; the gizmo calls reuse it. A `None`
mapping (cursor outside the rendered image) is treated as a miss.

### D3: Hold the render lock around gizmo calls
`gizmo_update_drag` mutates instance transforms via `apply_instance_transform`,
which the render worker reads concurrently. Every gizmo renderer call is wrapped
in `with self._render_lock:` — consistent with the autofocus/pick/zoom paths in
the same handlers. (`gizmo_hit_test` and `set_hover` are reads but are wrapped
too, for uniformity and because hit-test reads camera/instance state.)

### D4: Hover on tracked moves
`setMouseTracking(True)` is already enabled, so `mouseMoveEvent` fires with no
button down. When no button is held and no zoom drag is active, the handler sets
`gizmo.set_hover(gizmo_hit_test(mapped))`. During a gizmo drag, moves route to
`gizmo_update_drag` and skip both hover and camera. Hover is suppressed while any
button is down (camera drag in progress).

### D5: Factor the press decision into a pure seam
The press-time precedence is a pure function `gizmo_press_action` — input: the
armed-mode flags + whether the gizmo reports a ring hit; output: an action tag
(`autofocus` / `pick` / `zoom` / `gizmo` / `camera`). To make press/move/release
testable as a unit (not just the precedence), it is wrapped by a thin
GUI-agnostic `GizmoMouseController` (`src/skinny/ui/gizmo_input.py`) that owns the
drag-in-progress flag and drives the duck-typed renderer gizmo API
(`gizmo_hit_test` / `gizmo_begin_drag` / `gizmo_update_drag` / `gizmo_end_drag` /
`gizmo.set_hover`). The Qt viewport maps coordinates, holds `self._render_lock`,
and delegates to `self._gizmo.on_press/on_move/on_release`. This mirrors the
existing "pure wiring helpers (testable seams)" pattern from the scene-graph-ui
change and lets the whole interaction be unit-tested with a stub renderer and no
`QApplication`. The parity guard (D6) drives the controller against a recording
stub and asserts the Qt handlers delegate to it.

### D6: Parity guard
A test asserts the Qt viewport's mouse handlers reach the gizmo API on
press/move/release (e.g. by driving the dispatch seam and/or asserting the
renderer gizmo methods are called with a stubbed renderer). The intent: a
render-surface front-end that targets the gizmo must also let the user grab it —
catching the exact regression this change fixes.

## Risks / Open Questions

- **Drag-vs-orbit feel**: with D1, a press that grazes a ring grabs the gizmo
  instead of orbiting. `GIZMO_HIT_TOLERANCE` (6 px) is the same tolerance GLFW
  already ships, so behavior matches the other front-end; no new tuning.
- **Lock contention**: gizmo calls are short CPU math under `_render_lock`; the
  worker already contends on this lock for every frame, so no new contention
  class is introduced.
