"""``skinny-gui`` entry point — single-window Qt application.

Layout:
    QMainWindow
    ├── menu bar (File: Open, Quit)
    ├── central: RenderViewport
    ├── left dock: control sidebar (built from build_main_ui tree)
    └── status bar: GPU, encoder (none in desktop), accum frame counter
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import base64

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QComboBox, QDockWidget, QLineEdit,
    QMainWindow, QScrollArea, QTextEdit, QWidget,
)

import numpy as np

from skinny.params import _apply_saved_params, _snapshot_params, build_all_params
from skinny.renderer import Renderer
from skinny.settings import ensure_dirs, load_settings, save_settings
from skinny.ui.build_app_ui import AppCallbacks, build_main_ui
from skinny.ui.qt.backend import QtTreeBuilder
from skinny.ui.qt.viewport import RenderViewport
from skinny.ui.qt.windows.bxdf import BXDFDock
from skinny.ui.qt.windows.debug_viewport import DebugViewportDock
from skinny.ui.qt.windows.material_graph import MaterialGraphDock
from skinny.ui.qt.windows.scene_graph import SceneGraphDock
from skinny.vk_context import VulkanContext

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self, scene_path: Path | None, gpu_pref: str, use_usd_mtlx: bool,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Skinny")
        self.resize(1600, 900)

        # Renderer setup — synchronous on main thread (no per-user sessions
        # to worry about in the desktop entry). Headless mode: no GLFW
        # window, no surface, no swapchain. DebugViewport renders to an
        # offscreen image and Qt blits it.
        self.ctx = VulkanContext(
            window=None, width=1280, height=720, gpu_preference=gpu_pref,
        )
        log.info("GPU: %s", self.ctx.gpu_info.name)

        repo_root = Path(__file__).resolve().parents[3]
        self.renderer = Renderer(
            vk_ctx=self.ctx,
            shader_dir=Path(__file__).resolve().parents[1].parent / "shaders",
            hdr_dir=repo_root / "hdrs",
            tattoo_dir=repo_root / "tattoos",
            usd_scene_path=scene_path,
            use_usd_mtlx_plugin=use_usd_mtlx,
        )

        # Render viewport: hosted in a dock so the user can detach / re-
        # arrange it alongside the other tool docks. QMainWindow needs a
        # central widget for the layout machinery; a 1px placeholder is
        # enough since every visible surface (render + tool docks) is a
        # QDockWidget.
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.AllowNestedDocks
            | QMainWindow.AllowTabbedDocks
            | QMainWindow.AnimatedDocks,
        )
        placeholder = QWidget()
        placeholder.setFixedSize(0, 0)
        self.setCentralWidget(placeholder)

        self.viewport = RenderViewport(self.renderer, parent=self)
        render_dock = QDockWidget("Render", self)
        # objectName is required by QMainWindow.saveState/restoreState.
        render_dock.setObjectName("render")
        render_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        render_dock.setWidget(self.viewport)
        self.addDockWidget(Qt.RightDockWidgetArea, render_dock)
        self._render_dock = render_dock

        # Debug viewport: embedded dock built on first open. Renders into
        # an offscreen Vulkan image and blits via QImage.
        self._debug_dock: DebugViewportDock | None = None

        # Status bar.
        sb = self.statusBar()
        sb.showMessage(f"GPU: {self.ctx.gpu_info.name}  |  accum: 0")
        self.viewport.accum_changed.connect(
            lambda n: sb.showMessage(
                f"GPU: {self.ctx.gpu_info.name}  |  accum: {n}"
            )
        )

        # Holders for the child docks — instantiated on first open so the
        # tree picks up scene graphs created after startup.
        self._scene_graph_dock: SceneGraphDock | None = None
        self._bxdf_dock: BXDFDock | None = None
        self._material_graph_dock: MaterialGraphDock | None = None

        # Sidebar built from the shared spec tree.
        cb = AppCallbacks(
            open_scene_graph=self._open_scene_graph,
            open_material_graph=self._open_material_graph,
            open_bxdf_visualizer=self._open_bxdf,
            open_debug_viewport=self._toggle_debug_viewport,
        )
        tree = build_main_ui(self.renderer, callbacks=cb)

        sidebar_inner = QWidget()
        self._tree_builder = QtTreeBuilder(tree, sidebar_inner)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar_inner)
        scroll.setMinimumWidth(360)
        dock = QDockWidget("Controls", self)
        dock.setObjectName("controls")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._sidebar_dock = dock

        # Bias the initial split: render gets the lion's share of width,
        # controls sit at 360px.
        self.resizeDocks(
            [dock, self._render_dock], [380, 1200], Qt.Horizontal,
        )

        # Menu bar — window openers + file actions.
        self._build_menu_bar()

        # Restore previous session state (params, camera, dock layout,
        # which child docks were open). Run after every dock holder is
        # initialised so QMainWindow.restoreState can find named docks.
        try:
            ensure_dirs()
            self._saved_settings = load_settings()
        except Exception:  # noqa: BLE001
            self._saved_settings = {}
        self._restore_session_state()

        # Keys the viewport responds to (camera mode toggle, focus reset,
        # HUD toggle, free-cam WASDQE). Forwarded from MainWindow when no
        # text-editing widget is focused.
        self._viewport_keys = {
            Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E,
            Qt.Key_C, Qt.Key_F, Qt.Key_F1, Qt.Key_Space,
        }

        # Hand initial focus to the render viewport so shortcuts work
        # without a click first. Defer to the next event-loop tick so the
        # widget is actually realised.
        QTimer.singleShot(0, lambda: self.viewport.setFocus(Qt.OtherFocusReason))

    # ── Key forwarding ────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if self._should_forward_key(event.key()):
            self.viewport.keyPressEvent(event)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self._should_forward_key(event.key()):
            self.viewport.keyReleaseEvent(event)
            return
        super().keyReleaseEvent(event)

    def _should_forward_key(self, key: int) -> bool:
        if key not in self._viewport_keys:
            return False
        # Don't steal letters from text-edit widgets. ComboBox eats
        # alpha keys for type-ahead search; leave it alone too.
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox)):
            return False
        return True

    def _stub(self, name: str) -> None:
        self.statusBar().showMessage(f"{name}: not yet ported (Phase 7)", 3000)

    def _open_scene_graph(self) -> None:
        if self._scene_graph_dock is None:
            self._scene_graph_dock = SceneGraphDock(self.renderer, parent=self)
            self._scene_graph_dock.setObjectName("scene_graph")
            self.addDockWidget(Qt.BottomDockWidgetArea, self._scene_graph_dock)
        self._scene_graph_dock.show()
        self._scene_graph_dock.raise_()

    def _open_bxdf(self) -> None:
        if self._bxdf_dock is None:
            self._bxdf_dock = BXDFDock(self.renderer, self.viewport, parent=self)
            self._bxdf_dock.setObjectName("bxdf")
            self.addDockWidget(Qt.RightDockWidgetArea, self._bxdf_dock)
        self._bxdf_dock.show()
        self._bxdf_dock.raise_()

    def _open_material_graph(self) -> None:
        if self._material_graph_dock is None:
            self._material_graph_dock = MaterialGraphDock(self.renderer, parent=self)
            self._material_graph_dock.setObjectName("material_graph")
            self.addDockWidget(Qt.BottomDockWidgetArea, self._material_graph_dock)
        self._material_graph_dock.show()
        self._material_graph_dock.raise_()

    def _ensure_debug_dock(self) -> DebugViewportDock:
        if self._debug_dock is not None:
            return self._debug_dock
        self._debug_dock = DebugViewportDock(
            ctx=self.ctx, renderer=self.renderer,
            main_lock=self.viewport._render_lock, parent=self,
        )
        self._debug_dock.setObjectName("debug_viewport")
        self.addDockWidget(Qt.BottomDockWidgetArea, self._debug_dock)
        return self._debug_dock

    def _toggle_debug_viewport(self) -> None:
        try:
            dock = self._ensure_debug_dock()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Debug viewport unavailable: {exc}", 5000)
            return
        if dock.isVisible():
            dock.hide()
        else:
            dock.show()
            dock.raise_()

    # ── Menu bar ──────────────────────────────────────────────────

    def _build_menu_bar(self) -> None:
        """File menu (Open / Quit) + View menu (window openers)."""
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        open_action = QAction("&Open scene…", self)
        open_action.triggered.connect(self._on_menu_open_scene)
        file_menu.addAction(open_action)
        load_hdr_action = QAction("Load &HDR…", self)
        load_hdr_action.triggered.connect(self._on_menu_load_hdr)
        file_menu.addAction(load_hdr_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = bar.addMenu("&View")
        for label, slot in (
            ("&Scene Graph",      self._open_scene_graph),
            ("&Material Graph",   self._open_material_graph),
            ("&BXDF Visualizer",  self._open_bxdf),
            ("&Camera Debug View", self._toggle_debug_viewport),
        ):
            act = QAction(label, self)
            act.triggered.connect(slot)
            view_menu.addAction(act)

    def _on_menu_open_scene(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scene", "",
            "USD scenes (*.usda *.usdc *.usdz);;OBJ (*.obj);;All files (*.*)",
        )
        if path:
            self.renderer.load_model_from_path(Path(path))

    def _on_menu_load_hdr(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load HDR", "",
            "HDR images (*.hdr *.exr *.pfm);;All files (*.*)",
        )
        if path:
            self.renderer.apply_dome_light_texture(0, path)

    # ── State persistence ────────────────────────────────────────

    def _restore_session_state(self) -> None:
        """Apply saved params/camera + reopen previously-open child docks
        + restore dock layout. Tolerant of partial/missing settings.
        """
        data = self._saved_settings or {}

        try:
            _apply_saved_params(
                self.renderer, data.get("params", {}),
                build_all_params(self.renderer),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to apply saved params: %s", exc)

        cam = data.get("camera")
        if isinstance(cam, dict):
            try:
                _apply_camera_snapshot(self.renderer, cam)
                self.renderer._update_light()
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to apply saved camera: %s", exc)

        # Recreate child docks the user had open last session — needs to
        # happen before restoreState so the named docks exist.
        open_docks = data.get("open_docks") or []
        if "scene_graph" in open_docks:
            self._open_scene_graph()
        if "material_graph" in open_docks:
            self._open_material_graph()
        if "bxdf" in open_docks:
            self._open_bxdf()
        if "debug_viewport" in open_docks:
            self._toggle_debug_viewport()

        # Section open/closed state (sidebar QGroupBox checkboxes).
        sec_states = data.get("section_states")
        if isinstance(sec_states, dict):
            try:
                self._tree_builder.apply_section_states(sec_states)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to apply section states: %s", exc)

        geom_b64 = data.get("qt_geometry")
        state_b64 = data.get("qt_dock_state")
        if isinstance(geom_b64, str):
            try:
                self.restoreGeometry(QByteArray(base64.b64decode(geom_b64)))
            except Exception:
                pass
        if isinstance(state_b64, str):
            try:
                self.restoreState(QByteArray(base64.b64decode(state_b64)))
            except Exception:
                pass

    def _snapshot_session_state(self) -> dict:
        """Capture params, camera, open docks, and Qt dock geometry."""
        out: dict = {}
        try:
            out["params"] = _snapshot_params(
                self.renderer, build_all_params(self.renderer),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to snapshot params: %s", exc)
        try:
            out["camera"] = _snapshot_camera(self.renderer)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to snapshot camera: %s", exc)
        open_docks: list[str] = []
        for name, dock in (
            ("scene_graph", self._scene_graph_dock),
            ("material_graph", self._material_graph_dock),
            ("bxdf", self._bxdf_dock),
            ("debug_viewport", self._debug_dock),
        ):
            if dock is not None and dock.isVisible():
                open_docks.append(name)
        out["open_docks"] = open_docks
        try:
            out["section_states"] = self._tree_builder.section_states()
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to snapshot section states: %s", exc)
        try:
            out["qt_geometry"] = base64.b64encode(
                bytes(self.saveGeometry()),
            ).decode("ascii")
            out["qt_dock_state"] = base64.b64encode(
                bytes(self.saveState()),
            ).decode("ascii")
        except Exception:  # noqa: BLE001
            pass
        return out

    def closeEvent(self, event) -> None:
        # Snapshot BEFORE tearing down so we still have a live renderer
        # to read params from.
        try:
            save_settings(self._snapshot_session_state())
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to save session settings: %s", exc)

        self.viewport.shutdown()
        if self._debug_dock is not None:
            try:
                self._debug_dock.close()
            except Exception:
                pass
        try:
            self.renderer.cleanup()
        except Exception:
            pass
        try:
            self.ctx.destroy()
        except Exception:
            pass
        super().closeEvent(event)


def _snapshot_camera(renderer) -> dict:
    orbit = renderer.orbit_camera
    free = renderer.free_camera
    return {
        "mode": renderer.camera_mode,
        "orbit": {
            "yaw": float(orbit.yaw),
            "pitch": float(orbit.pitch),
            "distance": float(orbit.distance),
            "fov": float(orbit.fov),
            "target": [float(orbit.target[0]), float(orbit.target[1]), float(orbit.target[2])],
        },
        "free": {
            "position": [float(free.position[0]), float(free.position[1]), float(free.position[2])],
            "yaw": float(free.yaw),
            "pitch": float(free.pitch),
            "fov": float(free.fov),
            "move_speed": float(free.move_speed),
        },
    }


def _apply_camera_snapshot(renderer, saved_cam) -> None:
    """Restore ``orbit_camera`` + ``free_camera`` from a snapshot dict.
    Out-of-range / missing values fall back to the renderer's defaults.
    """
    if not isinstance(saved_cam, dict):
        return

    def _vec3(raw, fallback):
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            try:
                return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
            except (TypeError, ValueError):
                pass
        return fallback

    def _flt(raw, fallback):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    orbit_raw = saved_cam.get("orbit")
    if isinstance(orbit_raw, dict):
        o = renderer.orbit_camera
        o.yaw = _flt(orbit_raw.get("yaw"), o.yaw)
        o.pitch = float(np.clip(
            _flt(orbit_raw.get("pitch"), o.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        o.distance = float(np.clip(_flt(orbit_raw.get("distance"), o.distance), 0.5, 50.0))
        o.fov = float(np.clip(_flt(orbit_raw.get("fov"), o.fov), 1.0, 170.0))
        o.target = _vec3(orbit_raw.get("target"), o.target)

    free_raw = saved_cam.get("free")
    if isinstance(free_raw, dict):
        f = renderer.free_camera
        f.position = _vec3(free_raw.get("position"), f.position)
        f.yaw = _flt(free_raw.get("yaw"), f.yaw)
        f.pitch = float(np.clip(
            _flt(free_raw.get("pitch"), f.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        f.fov = float(np.clip(_flt(free_raw.get("fov"), f.fov), 1.0, 170.0))
        f.move_speed = float(np.clip(_flt(free_raw.get("move_speed"), f.move_speed), 0.05, 50.0))

    mode = saved_cam.get("mode")
    if mode in ("orbit", "free"):
        renderer.camera_mode = mode


def main() -> None:
    parser = argparse.ArgumentParser(prog="skinny-gui")
    parser.add_argument(
        "scene", nargs="?", type=Path, default=None,
        help="Path to a USD stage (.usda/.usdc/.usdz).",
    )
    parser.add_argument("--gpu", type=str, default="auto",
                        help="GPU preference: intel, nvidia, amd, discrete, auto")
    parser.add_argument("--usdMtlx", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    win = MainWindow(args.scene, args.gpu, args.usdMtlx)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
