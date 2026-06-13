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

from PySide6.QtCore import QByteArray, QEvent, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QDockWidget,
    QLineEdit, QMainWindow, QPlainTextEdit, QScrollArea, QTextEdit, QWidget,
)

import numpy as np

from skinny.cli_common import (
    INTEGRATOR_INDEX,
    add_render_flags,
    resolve_walk,
    validate_render_flags,
)
from skinny.params import _apply_saved_params, _snapshot_params, build_all_params
from skinny.renderer import Renderer
from skinny.settings import (
    ensure_dirs,
    get_last_dir,
    last_dirs_snapshot,
    load_settings,
    record_last_dir,
    save_settings,
)
from skinny.ui.build_app_ui import AppCallbacks, build_main_ui
from skinny.ui.qt.backend import QtTreeBuilder
from skinny.ui.qt.viewport import RenderViewport
from skinny.ui.qt.windows.bxdf import BXDFDock
from skinny.ui.qt.windows.debug_viewport import DebugViewportDock
from skinny.ui.qt.windows.material_graph import MaterialGraphDock
from skinny.ui.qt.windows.python_material_editor import PythonMaterialEditorDock
from skinny.ui.qt.windows.scene_graph import SceneGraphDock
from skinny.backend_select import (
    make_context,
    select_backend,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self, scene_path: Path | None, gpu_pref: str, use_usd_mtlx: bool,
        execution_mode: str = "megakernel", bdpt_walk: str = "fused",
        initial_integrator: str | None = None,
        neural_handoff: str = "file", neural_trainer: str = "auto",
        train_precision: str = "fp32", online_training: bool = False,
        reuse: str | None = None,
        lobe_samplers: str | None = None,
        backend: str = "vulkan",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Skinny")
        self.resize(1600, 900)

        # Resolved GPU backend, persisted in the session snapshot. The Qt GUI is
        # offscreen-rendered (no GLFW window) via make_context(window=None) — the
        # headless path both backends support at full parity, so auto→Metal on
        # Apple Silicon works here too. main() resolves the backend (an explicit,
        # unavailable --backend metal errors) before constructing this window.
        self._backend_name = backend

        # Renderer setup — synchronous on main thread (no per-user sessions
        # to worry about in the desktop entry). Headless mode: no GLFW
        # window, no surface, no swapchain. DebugViewport renders to an
        # offscreen image and Qt blits it.
        self.ctx = make_context(
            backend, window=None, width=1280, height=720, gpu_preference=gpu_pref,
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
            execution_mode=execution_mode,
            bdpt_walk=bdpt_walk,
            neural_handoff=neural_handoff,
            neural_trainer=neural_trainer,
            train_precision=train_precision,
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

        self.viewport = RenderViewport(
            self.renderer, parent=self, online_training=online_training)
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

        # Status bar — GPU + accumulation, plus the online-training state polled
        # from the renderer's lock-free snapshot (change
        # online-training-observability) so training is visible without a console.
        sb = self.statusBar()
        sb.showMessage(f"GPU: {self.ctx.gpu_info.name}  |  accum: 0")
        self.viewport.accum_changed.connect(self._update_status_bar)

        # Holders for the child docks — instantiated on first open so the
        # tree picks up scene graphs created after startup.
        self._scene_graph_dock: SceneGraphDock | None = None
        self._bxdf_dock: BXDFDock | None = None
        self._material_graph_dock: MaterialGraphDock | None = None
        self._python_material_dock: PythonMaterialEditorDock | None = None

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
        # CLI --integrator (when given) wins over the persisted value for this launch.
        if initial_integrator is not None:
            self.renderer.integrator_index = INTEGRATOR_INDEX[initial_integrator]
        # CLI --reuse / --lobe-samplers override the persisted sampling seam
        # (mirrors app.py GLFW). skinny-gui has no --proposals: the Proposals
        # combobox owns proposal selection at runtime, and the online-training
        # gate polls lazily until a neural proposal becomes active, so no
        # startup seed is needed.
        if reuse is not None:
            self.renderer.reuse_index = self.renderer._REUSE_TOKENS.index(reuse)
        if lobe_samplers is not None:
            from skinny.sampling import parse_lobe_samplers

            c, s, d = parse_lobe_samplers(lobe_samplers)
            self.renderer.coat_sampler_index = c
            self.renderer.spec_sampler_index = s
            self.renderer.diff_sampler_index = d

        # Keys the viewport responds to (camera mode toggle, focus reset,
        # HUD toggle, free-cam WASDQE). Forwarded from MainWindow when no
        # text-editing widget is focused.
        self._viewport_keys = {
            Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E,
            Qt.Key_C, Qt.Key_F, Qt.Key_F1, Qt.Key_Space,
            Qt.Key_L, Qt.Key_V, Qt.Key_Z, Qt.Key_X,
        }

        # Hand initial focus to the render viewport so shortcuts work
        # without a click first. Defer to the next event-loop tick so the
        # widget is actually realised.
        QTimer.singleShot(0, lambda: self.viewport.setFocus(Qt.OtherFocusReason))

        # Intercept key events application-wide so shortcuts work even
        # when a sidebar slider/button has keyboard focus. Sliders, dock
        # title bars, etc. otherwise eat WASD/Space before our viewport's
        # keyPressEvent ever fires.
        QApplication.instance().installEventFilter(self)

    # ── Online-training status (change online-training-observability) ──

    def _neural_status_text(self) -> str:
        """One-line online-training state for the status bar, or '' when off."""
        st = self.renderer.online_training_status()
        if not st["armed"]:
            return ""
        if st["active"]:
            loss = st["last_loss"]
            loss_s = f"{loss:.3f}" if loss is not None else "n/a"
            return f"  |  neural: ACTIVE {st['cycles']}cyc loss={loss_s}"
        return "  |  neural: armed (waiting)"

    def _update_status_bar(self, n: int) -> None:
        self.statusBar().showMessage(
            f"GPU: {self.ctx.gpu_info.name}  |  accum: {n}"
            + self._neural_status_text()
        )

    # ── Key forwarding ────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        et = event.type()
        if et == QEvent.KeyPress and self._should_forward_key(
            event.key(), event.modifiers(),
        ):
            self.viewport.keyPressEvent(event)
            return True
        if et == QEvent.KeyRelease and self._should_forward_key(
            event.key(), event.modifiers(),
        ):
            self.viewport.keyReleaseEvent(event)
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if self._should_forward_key(event.key(), event.modifiers()):
            self.viewport.keyPressEvent(event)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self._should_forward_key(event.key(), event.modifiers()):
            self.viewport.keyReleaseEvent(event)
            return
        super().keyReleaseEvent(event)

    def _should_forward_key(self, key: int, modifiers) -> bool:
        if key not in self._viewport_keys:
            return False
        # Reserve Ctrl/Cmd/Alt key combos for app-wide shortcuts (Compile,
        # Undo, Redo, etc.) — they otherwise overlap viewport keys like Z.
        if modifiers & (
            Qt.ControlModifier | Qt.MetaModifier | Qt.AltModifier
        ):
            return False
        # Don't steal keys from text-edit widgets, spin boxes, combo
        # boxes (type-ahead search), or focused buttons (Space activates).
        focus = QApplication.focusWidget()
        if isinstance(focus, (
            QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox,
            QComboBox, QAbstractButton,
        )):
            return False
        return True

    def _stub(self, name: str) -> None:
        self.statusBar().showMessage(f"{name}: not yet ported (Phase 7)", 3000)

    def _open_scene_graph(self) -> None:
        if self._scene_graph_dock is None:
            self._scene_graph_dock = SceneGraphDock(
                self.renderer, parent=self,
                on_open_python_material=self._open_python_material_in_editor,
            )
            self._scene_graph_dock.setObjectName("scene_graph")
            self.addDockWidget(Qt.BottomDockWidgetArea, self._scene_graph_dock)
        self._scene_graph_dock.show()
        self._scene_graph_dock.raise_()

    def _open_python_material_in_editor(self, module_name: str) -> None:
        """Open the editor dock (creating it if needed) and load
        `module_name` into the buffer. Used by Scene Graph's double-click.
        """
        self._open_python_material_editor()
        if self._python_material_dock is not None:
            self._python_material_dock.set_active_module(module_name)

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

    def _open_python_material_editor(self) -> None:
        if self._python_material_dock is None:
            self._python_material_dock = PythonMaterialEditorDock(
                self.renderer, self.viewport._render_lock, parent=self,
            )
            self._python_material_dock.setObjectName("python_material_editor")
            self.addDockWidget(
                Qt.RightDockWidgetArea, self._python_material_dock,
            )
        self._python_material_dock.refresh_from_renderer()
        self._python_material_dock.show()
        self._python_material_dock.raise_()

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

    def _show_render_viewport(self) -> None:
        self._render_dock.show()
        self._render_dock.raise_()

    def _show_sidebar(self) -> None:
        self._sidebar_dock.show()
        self._sidebar_dock.raise_()

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
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = bar.addMenu("&View")
        for label, slot, shortcut in (
            ("&Render",           self._show_render_viewport,    None),
            ("&Controls",         self._show_sidebar,            None),
            ("&Scene Graph",      self._open_scene_graph,        None),
            ("&Material Graph",   self._open_material_graph,     None),
            ("&Python Material Editor", self._open_python_material_editor,
                                                                 "Ctrl+Shift+P"),
            ("&BXDF Visualizer",  self._open_bxdf,               None),
            ("&Camera Debug View", self._toggle_debug_viewport,  None),
        ):
            act = QAction(label, self)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            view_menu.addAction(act)

    def _on_menu_open_scene(self) -> None:
        from skinny.ui.qt.dialogs import get_open_file_name
        path = get_open_file_name(
            self, "Open scene", get_last_dir("model"),
            "USD scenes (*.usda *.usdc *.usdz);;OBJ (*.obj);;All files (*.*)",
        )
        if path:
            record_last_dir("model", Path(path).parent)
            self.renderer.load_model_from_path(Path(path))
            if self._python_material_dock is not None:
                self._python_material_dock.refresh_from_renderer()

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

        gm = data.get("gizmo_mode")
        if gm is not None:
            try:
                from skinny.gizmo import GizmoMode
                self.renderer.gizmo.mode = GizmoMode(int(gm))
            except (TypeError, ValueError):
                pass

        # Recreate child docks the user had open last session — needs to
        # happen before restoreState so the named docks exist.
        open_docks = data.get("open_docks") or []
        if "scene_graph" in open_docks:
            self._open_scene_graph()
        if "material_graph" in open_docks:
            self._open_material_graph()
        if "python_material_editor" in open_docks:
            self._open_python_material_editor()
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
        try:
            out["gizmo_mode"] = int(self.renderer.gizmo.mode)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to snapshot gizmo mode: %s", exc)
        open_docks: list[str] = []
        for name, dock in (
            ("scene_graph", self._scene_graph_dock),
            ("material_graph", self._material_graph_dock),
            ("python_material_editor", self._python_material_dock),
            ("bxdf", self._bxdf_dock),
            ("debug_viewport", self._debug_dock),
        ):
            if dock is not None and dock.isVisible():
                open_docks.append(name)
        out["open_docks"] = open_docks
        out["last_dirs"] = last_dirs_snapshot()
        out["backend"] = self._backend_name
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
    # No --proposals on the interactive front-ends (skinny-gui / skinny-web):
    # the Proposals combobox owns proposal selection at runtime (and persists it).
    add_render_flags(parser, proposals=False)
    args = parser.parse_args()
    # Reject impossible combos (e.g. bdpt + --online-training) up front.
    validate_render_flags(args)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve the GPU backend (precedence: --backend > SKINNY_BACKEND > persisted
    # > auto). auto resolves to Metal on Apple Silicon, else Vulkan; an explicit,
    # unavailable --backend metal errors clearly rather than crashing.
    try:
        backend = select_backend(args.backend, persisted=load_settings().get("backend"))
    except RuntimeError as exc:
        raise SystemExit(f"skinny-gui: {exc}")

    app = QApplication(sys.argv)
    win = MainWindow(args.scene, args.gpu, args.usdMtlx, args.execution_mode,
                     resolve_walk(args.bdpt_walk), args.integrator,
                     neural_handoff=args.neural_handoff,
                     neural_trainer=args.neural_trainer,
                     train_precision=args.train_precision,
                     online_training=args.online_training,
                     reuse=args.reuse,
                     lobe_samplers=args.lobe_samplers,
                     backend=backend)
    # Display-only state for the startup configuration matrix (change
    # online-training-observability).
    win.renderer._requested_backend = args.backend
    win.renderer._online_training_requested = bool(args.online_training)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
