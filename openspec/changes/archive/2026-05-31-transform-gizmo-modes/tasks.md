## 1. Gizmo core: mode + local basis

- [x] 1.1 Rename `RotateGizmo` → `TransformGizmo` in `src/skinny/gizmo.py` and update all imports/usages (renderer.py, gizmo_input.py, tests).
- [x] 1.2 Add a `GizmoMode` enum {`ROT_WORLD`, `ROT_LOCAL`, `TRA_WORLD`, `TRA_LOCAL`} ordered grouped by type; add `self.mode` and a `cycle_mode()` that does `(index+1)%4` and is a no-op while `is_dragging`.
- [x] 1.3 Add helpers `_is_translate(mode)` / `_is_local(mode)` and a `_basis(axis)` that returns the canonical world axis for world modes and `R_inst · axis` (rotation-only, normalized) for local modes.
- [x] 1.4 Extend `set_target(index, pivot_world, rotation)` to stash the instance rotation matrix `R_inst`; update the renderer call site to pass it.

## 2. Gizmo geometry + hit-test per mode

- [x] 2.1 Reorient the rotate rings to use `_basis` so local rings follow `R_inst` (world rings unchanged).
- [x] 2.2 Add translate-arrow geometry: three axis line-arrows (shaft + 2 arrowhead segments) from the pivot along `_basis(axis)`, projected to pixels; respect the segment cap.
- [x] 2.3 Make `build_segments` dispatch on mode (rings vs arrows) and append the active handle's highlight color on hover/drag.
- [x] 2.4 Make `hit_test` dispatch on mode: ring-loop point-distance for rotate, axis-line point-distance for translate; both return an axis string `"x"/"y"/"z"`.

## 3. Drag math per mode

- [x] 3.1 Unify rotation drag: build `R_delta = axisAngle(n, Δθ)` about world-space `n` (`n = _basis(axis)`), compose `R_new = R_delta @ R_start`, decompose to euler, return `(t_start, euler, s_start)`. Keep the existing screen-angle heuristic for `Δθ`.
- [x] 3.2 Add translate drag: project `pivot` and `pivot + _basis(axis)` to screen, take the unit screen direction, `pixels_along = (mouse-start)·dir`, convert px→world via the existing px-per-unit measurement, return `(t_start + dir_world*along, r_start, s_start)`, single-axis constrained.
- [x] 3.3 Make `begin_drag`/`update_drag` dispatch on mode and store the per-mode drag start state (start screen angle for rotate; start mouse + start translate for translate).

## 4. Glyph hint

- [x] 4.1 Add a `_glyph_segments(letter)` that strokes `W` or `L` (~6 segments) in a small fixed-size box, anchored at a fixed pixel offset above the projected pivot in screen space.
- [x] 4.2 Append the glyph (`W` for world modes, `L` for local) to `build_segments` output, drawn on top; verify total segments stay under `GIZMO_SEGMENT_CAPACITY`.

## 5. Renderer wiring + persistence

- [x] 5.1 Add a renderer method to cycle the gizmo mode (e.g. `gizmo_cycle_mode()`), and ensure `_refresh_gizmo_segments` passes `R_inst` into `set_target`.
- [x] 5.2 Persist `gizmo_mode` in the `settings.json` snapshot and restore it at startup (in `app.py`'s settings save/restore path), alongside camera/params.

## 6. Front-end keybinding parity

- [x] 6.1 GLFW (`src/skinny/app.py`): split `F1 or SPACE → show_hud` so F1 keeps the HUD toggle and SPACE calls the gizmo mode-cycle.
- [x] 6.2 Qt (`src/skinny/ui/qt/viewport.py`): same split, calling the cycle under the render lock.
- [x] 6.3 Debug viewport: if it can target the gizmo, bind space to the mode-cycle there too (else document why not). — N/A: the debug viewport is a visualization-only window (AABBs/grid/frustum/camera glyphs); it cannot target the transform gizmo, so it keeps its own `Space`→HUD-toggle. Documented in `test_gizmo_mode_parity.py` and the parity spec.

## 7. Tests + verification

- [x] 7.1 Unit test: `cycle_mode` advances `(index+1)%4` grouped by type and is a no-op mid-drag.
- [x] 7.2 Unit test: translate drag moves only the dragged axis; local translate moves along `R_inst`-rotated axis; world translate along the world axis.
- [x] 7.3 Unit test: local-rotate composes about `R_inst·axis` (compare resulting matrix to the world case under identity `R_inst`).
- [x] 7.4 Parity test: extend the existing front-end parity test so each render-surface front-end's space handler reaches the gizmo mode-cycle (Qt and GLFW), and a display-only / unbound-cycle case fails.
- [x] 7.5 Headless A/B verification of the unified world-rotate vs the old euler-add behavior; confirm the visible result is acceptable and note the behavior change in CHANGELOG/help text.
- [x] 7.6 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest`; no shader recompile expected (confirm `main_pass.slang` untouched).
