"""Render-thread-safety guard for the Camera Debug viewport dock (Phase 5).

The DebugViewport GPU object lives on the render worker (renderer.debug_viewport);
the worker renders it each frame and emits a DebugFrame. The dock is passive — it
blits the emitted frames and posts camera/display input + lifecycle to the worker,
never touching the GPU object directly. Source-level, mirroring
`test_qt_gizmo_viewport.py` — the actual GPU render needs a real context + GPU.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from skinny.ui.qt.windows import debug_viewport as dv  # noqa: E402

Dock = dv.DebugViewportDock


def test_dock_takes_proxy_and_viewport_not_ctx_or_lock() -> None:
    params = inspect.signature(Dock.__init__).parameters
    assert list(params) == ["self", "renderer", "viewport", "parent"]
    assert "ctx" not in params
    assert "main_lock" not in params


def test_worker_helpers_are_module_level() -> None:
    for fn in ("_worker_debug_create", "_worker_debug_resize",
               "_worker_debug_destroy", "_worker_debug_drag",
               "_worker_debug_wheel", "_worker_debug_move",
               "_worker_debug_call", "_worker_debug_toggle"):
        assert callable(getattr(dv, fn)), fn


def test_input_is_posted_to_the_worker() -> None:
    for method in ("_on_drag", "_on_wheel", "_poll_wasd", "_post_call",
                   "_post_toggle"):
        body = inspect.getsource(getattr(Dock, method))
        assert "self.renderer.post(" in body, method


def test_lifecycle_posts_to_the_worker() -> None:
    for method in ("showEvent", "hideEvent", "resizeEvent", "closeEvent"):
        body = inspect.getsource(getattr(Dock, method))
        assert "self.renderer.post(" in body, method


def test_dock_blits_worker_frames_no_gpu_render_on_gui() -> None:
    src = inspect.getsource(dv)
    # The dock consumes worker-emitted frames; it must not call render_embedded
    # itself (that runs on the worker in viewport.py).
    assert "self.viewport.debug_frame_ready.connect" in src
    assert "render_embedded" not in src


def test_worker_renders_debug_frame_in_viewport_loop() -> None:
    from skinny.ui.qt import viewport as vp
    src = inspect.getsource(vp._RenderWorker._maybe_render_debug)
    assert "render_embedded" in src
    assert "self.debug_frame_ready.emit(" in src
    assert "_debug_viewport_active" in src


# ── Camera Debug key map: reconciled or recorded ──────────────────────
# usd-scene-editing-ui: interaction bindings for a control surface present in
# more than one front-end are reconciled, or each divergence recorded with its
# reason and asserted here.

#: The GLFW `DebugViewport._on_key` chain, transcribed. Key name -> the state it
#: toggles or the method it calls. Asserted against the source below, so a change
#: to one of the two maps without the other fails.
GLFW_DEBUG_KEYS = {
    "C": "_toggle_cam_mode",
    "F": "_reset_debug_camera",
    "M": "show_mesh_wires",
    "G": "show_grid",
    "P": "show_focus_plane",
    "I": "show_render_area",
    "O": "ortho_mode",
    "D": "show_dof_planes",
    "T": "view_top",
    "B": "view_back",
    "L": "view_left",
    "SPACE": "show_hud",
    # Recorded divergence: the GLFW viewport owns an OS window it can close.
    # The Qt surface is a QDockWidget closed by its own title bar / the View
    # menu, so a key binding for it would duplicate a chrome affordance.
    "ESCAPE": "close",
}

#: Movement keys, held rather than pressed. Identical on both front-ends.
MOVEMENT = ("W", "A", "S", "D", "Q", "E")


def _qt_key_name(key) -> str:
    from PySide6.QtCore import Qt

    return {getattr(Qt, f"Key_{n}"): n for n in
            list("CFMGPIOTBLDWASQE") + ["Space"]}[key].upper()


def test_qt_debug_dock_key_map_matches_the_glfw_viewport() -> None:
    qt_map = {_qt_key_name(k): name for k, (_kind, name) in
              dv.PRESS_ACTIONS.items()}
    expected = {k: v for k, v in GLFW_DEBUG_KEYS.items() if k != "ESCAPE"}
    assert qt_map == expected


def test_glfw_debug_key_map_is_what_was_transcribed() -> None:
    from skinny.debug_viewport import DebugViewport

    src = inspect.getsource(DebugViewport._on_key)
    for key, target in GLFW_DEBUG_KEYS.items():
        assert f"glfw.KEY_{key}" in src, key
        assert target in src, target


def test_dof_plane_toggle_is_bound_in_both_channels() -> None:
    """`D` is a movement key *and* the depth-of-field plane toggle on the GLFW
    viewport; the Qt dock returned early for every movement key and dropped the
    toggle (review-surfaced-defects defect 5)."""
    from PySide6.QtCore import Qt

    assert Qt.Key_D in dv.MOVEMENT_KEYS
    assert dv.PRESS_ACTIONS[Qt.Key_D] == ("toggle", "show_dof_planes")

    body = inspect.getsource(Dock.keyPressEvent)
    # Movement keys stop early only when they carry no press action, and a held
    # strafe must not flip the toggle via auto-repeat.
    assert "key not in PRESS_ACTIONS or event.isAutoRepeat()" in body


def test_movement_keys_agree_across_front_ends() -> None:
    from skinny.debug_viewport import DebugViewport

    assert tuple(_qt_key_name(k) for k in dv.MOVEMENT_KEYS) == MOVEMENT
    # GLFW polls the held keys in `update`; Qt polls `_wasd` on a timer.
    src = inspect.getsource(DebugViewport.update)
    for name in MOVEMENT:
        assert f"glfw.KEY_{name}" in src, name
    qt_src = inspect.getsource(Dock._poll_wasd)
    for name in MOVEMENT:
        assert f"Qt.Key_{name if name != 'SPACE' else 'Space'}" in qt_src, name
