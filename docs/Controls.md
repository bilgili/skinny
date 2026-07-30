# Skinny — Controls

This document covers the interactive bindings: keyboard, mouse, and the
camera debug viewport.

For how to launch the front ends see [Usage.md](Usage.md).

---

## Controls

Keyboard and mouse controls are shown in the on-screen HUD when running the
GLFW debug entry. Qt and web entries use widget-driven input plus the
shortcuts below forwarded to the viewport.

| Input | Action |
|-------|--------|
| Left drag | Orbit camera (orbit mode) / look around (free mode) |
| Right drag | Pan orbit target |
| Scroll | Zoom (orbit) / adjust speed (free) |
| `C` | Toggle orbit / free camera |
| `W A S D` | Move in free-camera mode |
| `Q / E` | Move down / up in free-camera mode |
| `Tab / Shift+Tab` | Next / previous parameter (debug entry) |
| Arrow keys | Adjust selected parameter (debug entry) |
| `1`--`9` | Jump to parameter (debug entry) |
| `F` | Recenter camera |
| `R` | Reset parameters |
| `P` | Print all parameters |
| `H` | Print help |
| `L` | Toggle lens focus overlay |
| `V` | Toggle lens vignette debug (green=ray valid, red=clipped) |
| `Z` | Arm zoom rectangle (drag in viewport, release to apply) |
| `X` | Reset zoom rectangle |
| `F2` | Toggle camera debug viewport dock / window |
| `Space` | Cycle transform gizmo mode (rotate/translate × world/local) |
| `F1` | Toggle HUD |
| `Esc` | Quit |

### Camera Debug viewport

Its own key map, identical in the GLFW window and the `skinny-gui` dock (the
recorded set is asserted by `tests/test_qt_debug_viewport_dock.py`). `W A S D
Q E` move the debug camera in free mode; `D` also toggles the depth-of-field
planes on press, served from a separate channel so a held strafe does not flip
it.

| Input | Action |
|-------|--------|
| `C` | Toggle orbit / free debug camera |
| `F` | Reset debug camera |
| `W A S D Q E` | Move (free mode) |
| `M` | Toggle mesh wireframes (AABBs invert) |
| `G` | Toggle ground grid |
| `P` | Toggle focus plane |
| `D` | Toggle depth-of-field planes |
| `I` | Toggle render-area outline |
| `O` | Toggle orthographic projection |
| `T` / `B` / `L` | Top / back / left view |
| `Space` | Toggle HUD |
| `Esc` | Close (GLFW window only — the Qt dock closes from its title bar) |
