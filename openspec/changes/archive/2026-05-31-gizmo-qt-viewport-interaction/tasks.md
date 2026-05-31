## 1. Pure dispatch seam (testable, no Qt)

> Implemented in `src/skinny/ui/gizmo_input.py`. The seam grew from a single
> function into a thin `GizmoMouseController` that owns the drag lifecycle, so
> the parity guard (3.1) can drive press/move/release against a stub renderer
> with no `QApplication`. `gizmo_press_action` remains the pure precedence core.

- [x] 1.1 Add a pure function `gizmo_press_action(*, shift, pick_armed, zoom_arming, ring_hit) -> str` returning one of `"autofocus" | "pick" | "zoom" | "gizmo" | "camera"` in the D1 precedence order (shift → pick → zoom → gizmo → camera)
- [x] 1.2 Unit tests for 1.1: each armed mode wins in order; with nothing armed, `ring_hit=True` → `"gizmo"` and `ring_hit=False` → `"camera"`

## 2. Qt viewport gizmo interaction

> The Qt handlers map widget→render pixels, hold `self._render_lock`, and
> delegate to `self._gizmo` (the controller); the controller's `is_dragging`
> replaces a raw `_gizmo_dragging` flag.

- [x] 2.1 `mousePressEvent`: map the cursor via `_widget_to_render_pixel`, then under `self._render_lock` call `self._gizmo.on_press(...)` with the live shift/pick/zoom flags; dispatch on the returned action (autofocus/pick/zoom keep their existing behavior, `"gizmo"` means a drag began, `"camera"` falls through to `self._left = True`). Precedence runs entirely through `gizmo_press_action`
- [x] 2.2 `mouseMoveEvent`: zoom-drag handled first (early return); otherwise call `self._gizmo.on_move(...)` under the lock — it updates an in-progress drag (consumes the move, skips camera) or, when idle, sets ring hover via `gizmo.set_hover` (clears to `None` when the mapping is `None`)
- [x] 2.3 `mouseReleaseEvent`: on left release call `self._gizmo.on_release(...)` under the lock; if it ended a drag, return; otherwise clear `self._left`. Button-state flags are not left stuck (gizmo drag never sets `self._left`)
- [x] 2.4 Verify priority: shift-autofocus, scene-pick, and zoom-rect arming still win on press (controller only hit-tests the gizmo when no modal mode is armed); camera orbit/pan unchanged on a ring miss — covered by `test_gizmo_input.py` precedence + modal-suppression tests

## 3. Parity guard

- [x] 3.1 `tests/test_gizmo_input.py` drives the controller against a stub renderer recording `gizmo_hit_test` / `gizmo_begin_drag` / `gizmo_update_drag` / `gizmo_end_drag` / `gizmo.set_hover`; `tests/test_qt_gizmo_viewport.py` (PySide6-guarded, source-level — no `QApplication`) asserts each Qt handler delegates to the gizmo controller, so the viewport can't regress to a display-only gizmo

## 4. Manual verification

- [x] 4.1 In `skinny-gui`: load a USD scene, select a mesh instance in the scene-graph dock, confirm the rings draw; hover a ring → it highlights; drag a ring → the instance rotates about that axis; release → camera orbit resumes; a press off the rings orbits the camera as before — confirmed working by user
- [x] 4.2 Confirm shift-click autofocus, scene-pick (BXDF), and `Z` zoom-rect still behave with the gizmo targeted — confirmed working by user
- [x] 4.3 `.venv/bin/ruff check src/skinny/` passes; new tests pass (`test_gizmo_input.py`, `test_qt_gizmo_viewport.py` → 18 passed). NOTE: full `.venv/bin/pytest` has pre-existing, unrelated failures in `.venv` (3.12) — slang harness modules fail to load (`cannot open file 'skin_bssrdf.slang'`) and MaterialX/GPU tests; none reference the gizmo/viewport. Full suite needs the repo-root 3.13 env per CLAUDE.md
