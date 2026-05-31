## Why

The viewport gizmo can only rotate, and only about world axes. Users need to
translate instances too, and to manipulate in the instance's own (local) frame,
not just world. A single toggle that flips between the four useful manipulators
keeps the interaction lightweight instead of crowding the keymap.

## What Changes

- Add a **translation gizmo**: three axis arrows (X/Y/Z) drawn as screen-space
  line segments; dragging an arrow moves the instance along that one axis.
- Add a **coordinate-space dimension** to both gizmos: a *world* variant
  (canonical X/Y/Z) and a *local* variant (axes oriented by the instance's
  current rotation).
- Introduce **four gizmo modes** in one enum — rotate-world, rotate-local,
  translate-world, translate-local — cycled by the **space key**
  (`(mode+1) % 4`, grouped by type). Space is ignored while a drag is in flight.
- **BREAKING (keybinding):** space no longer toggles the HUD; **F1 keeps the HUD
  toggle**. Space is reassigned to gizmo-mode cycling in every render-surface
  front-end (GLFW, Qt, debug viewport) for parity.
- Add a **W/L glyph hint** above the gizmo (a few line segments in the existing
  gizmo buffer, billboarded at a fixed pixel offset above the projected pivot):
  `W` for world modes, `L` for local. The glyph alone communicates the space;
  the gizmo shape (rings vs arrows) communicates the type. No HUD text line.
- **Persist** the active gizmo mode in `~/.skinny/settings.json` across sessions.
- Unify rotation drag math so world and local share one path (rotation about a
  world-space axis vector, composed as a matrix, re-decomposed to euler). This
  slightly changes the existing world-rotate feel (more correct) and needs an
  A/B verification.

## Capabilities

### New Capabilities
<!-- none — this extends an existing capability -->

### Modified Capabilities
- `viewport-gizmo-interaction`: grows from rotate-only to a multi-mode transform
  gizmo. New/changed requirements for the translation gizmo, the four-mode space
  cycle, world-vs-local coordinate space, the W/L glyph hint, mode persistence,
  and front-end keybinding parity for the space cycle.

## Impact

- **Code:** `src/skinny/gizmo.py` (rename `RotateGizmo` → `TransformGizmo`; add
  mode enum, arrows, axis-line hit-test, translate drag, glyph, local basis);
  `src/skinny/renderer.py` (`set_gizmo_target` gains instance rotation; mode
  cycle + persistence; segment refresh); `src/skinny/ui/gizmo_input.py`
  (precedence unchanged); `src/skinny/app.py` and `src/skinny/ui/qt/viewport.py`
  (+ debug viewport) — rebind space, keep F1→HUD.
- **APIs:** renderer gizmo API keeps its shape; `set_gizmo_target` signature
  extends with the instance rotation; hit-test still returns an axis string.
- **Shaders:** none. Arrows and glyph reuse the binding-22 `GizmoSegment` buffer
  and the existing line draw in `main_pass.slang` (within the 256-segment cap).
- **Settings:** new persisted `gizmo_mode` field in `settings.json`.
- **Tests:** parity test that each render-surface front-end wires the space
  mode-cycle; gizmo math tests (translate projection, local-basis rotation);
  noted world-rotate behavior-change verification.
