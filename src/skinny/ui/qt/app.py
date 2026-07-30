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
from concurrent.futures import TimeoutError
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

from skinny.bringup import BringupPlan, plan_bringup
from skinny.cli_common import add_render_flags, resolve_mcp_roots
from skinny.session_snapshot import (
    QT_KEYS,
    capture_shared,
    contribute,
    resolve_persisted_flag,
    restore_params,
    restore_shared,
)
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
from skinny.ui.qt.render_session import (
    QtRendererConfig,
    QtRendererProxy,
    RenderCommandQueue,
)
from skinny.ui.qt.windows.bxdf import BXDFDock
from skinny.ui.qt.windows.debug_viewport import DebugViewportDock
from skinny.ui.qt.windows.material_graph import MaterialGraphDock
from skinny.ui.qt.windows.python_material_editor import PythonMaterialEditorDock
from skinny.ui.qt.windows.scene_graph import SceneGraphDock

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self, scene_path: Path | None, gpu_pref: str, use_usd_mtlx: bool,
        execution_mode: str = "megakernel", bdpt_walk: str = "fused",
        initial_integrator: str | None = None,
        plan: BringupPlan | None = None,
        neural_handoff: str = "file", neural_trainer: str = "auto",
        train_precision: str = "fp32", online_training: bool = False,
        reuse: str | None = None,
        lobe_samplers: str | None = None,
        backend: str = "vulkan",
        requested_backend: str = "auto",
        encoding: str = "E0",
        sppm_glossy_roughness: float | None = None,
        width: int = 640,
        height: int = 480,
        spectral: bool = False,
        mcp: bool = False,
        mcp_port: int = 0,
        mcp_roots: "list[str] | None" = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Skinny")
        self.resize(1600, 900)

        # Resolved GPU backend, persisted in the session snapshot. The render
        # worker creates the actual GPU context; MainWindow keeps only a
        # GUI-thread proxy and a command queue.
        self._backend_name = backend
        self._commands = RenderCommandQueue()
        self.renderer = QtRendererProxy(
            self._commands,
            width=width,
            height=height,
            backend=backend,
            encoding=encoding,
            sppm_glossy_roughness=sppm_glossy_roughness,
        )
        # Optional MCP control surface. Handed the GUI-thread proxy, so every
        # edit crosses to the render worker through the same queue the docks
        # use. A bind collision leaves the GUI running without it.
        if mcp:
            from skinny.mcp_server import start as _mcp_start
            _mcp_start(self.renderer, mcp_port, roots=mcp_roots)

        config = QtRendererConfig(
            scene_path=scene_path,
            gpu_pref=gpu_pref,
            use_usd_mtlx=use_usd_mtlx,
            execution_mode=execution_mode,
            bdpt_walk=bdpt_walk,
            initial_integrator=initial_integrator,
            neural_handoff=neural_handoff,
            neural_trainer=neural_trainer,
            train_precision=train_precision,
            online_training=online_training,
            reuse=reuse,
            lobe_samplers=lobe_samplers,
            backend=backend,
            encoding=encoding,
            sppm_glossy_roughness=sppm_glossy_roughness,
            width=width,
            height=height,
            requested_backend=requested_backend,
            spectral=spectral,
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

        # The startup bring-up plan travels *alongside* QtRendererConfig (whose
        # signatures stay Qt-owned) down to the render worker, which runs the
        # deferred `create` step on the render thread.
        self.viewport = RenderViewport(
            config, self.renderer, self._commands, plan=plan, parent=self)
        render_dock = QDockWidget("Render", self)
        # objectName is required by QMainWindow.saveState/restoreState.
        render_dock.setObjectName("render")
        render_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        render_dock.setWidget(self.viewport)
        self.addDockWidget(Qt.RightDockWidgetArea, render_dock)
        self._render_dock = render_dock

        # Debug viewport: embedded dock built on first open. Renders into
        # an offscreen Vulkan image and blits via QImage.
        self._debug_dock = None

        # Status bar — GPU + accumulation, plus the online-training state polled
        # from the renderer's lock-free snapshot (change
        # online-training-observability) so training is visible without a console.
        sb = self.statusBar()
        sb.showMessage("GPU: starting  |  accum: 0")
        self.viewport.accum_changed.connect(self._update_status_bar)

        # Holders for the child docks — instantiated on first open so the
        # tree picks up scene graphs created after startup.
        self._scene_graph_dock = None
        self._bxdf_dock = None
        self._material_graph_dock = None
        self._python_material_dock = None

        # Sidebar built from the shared spec tree.
        cb = AppCallbacks(
            open_scene_graph=self._open_scene_graph,
            open_material_graph=self._open_material_graph,
            open_bxdf_visualizer=self._open_bxdf,
            open_debug_viewport=self._toggle_debug_viewport,
            load_model=self._queue_load_model,
            resize_render_target=self.viewport.request_resize,
            capture_screenshot=self._capture_screenshot,
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
        frame = self.viewport.latest_frame()
        st = frame.online_training if frame is not None else {}
        if not st.get("armed"):
            return ""
        if st.get("active"):
            loss = st.get("last_loss")
            loss_s = f"{loss:.3f}" if loss is not None else "n/a"
            return f"  |  neural: ACTIVE {st.get('cycles', 0)}cyc loss={loss_s}"
        return "  |  neural: armed (waiting)"

    def _update_status_bar(self, n: int) -> None:
        frame = self.viewport.latest_frame()
        gpu_name = frame.gpu_name if frame is not None else self.renderer.gpu_name
        self.statusBar().showMessage(
            f"GPU: {gpu_name}  |  accum: {n}"
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
                self.renderer, parent=self,
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
            self.renderer, self.viewport, parent=self,
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
            self._queue_load_model(Path(path))

    def _queue_load_model(self, path: Path) -> None:
        def load(renderer, path=Path(path)) -> None:
            renderer.load_model_from_path(path)

        self.viewport.post_render_command(load, coalesce_key="load-model")
        if self._python_material_dock is not None:
            QTimer.singleShot(250, self._python_material_dock.refresh_from_renderer)

    def _capture_screenshot(self, fmt: str) -> bytes:
        import io as _io
        buf = _io.BytesIO()
        self.renderer.save_screenshot(buf, fmt)
        return buf.getvalue()

    # ── State persistence ────────────────────────────────────────

    def _restore_session_state(self) -> None:
        """Apply saved params/camera + reopen previously-open child docks
        + restore dock layout. Tolerant of partial/missing settings.
        """
        data = self._saved_settings or {}

        try:
            restore_params(self.renderer, data)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to apply saved params: %s", exc)

        def restore_renderer(renderer, data=data) -> None:
            # Params + camera + gizmo mode through the snapshot owner (change
            # session-settings-owner) — the same rule `skinny` restores by.
            try:
                restore_shared(renderer, data)
                renderer._update_light()
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to restore session state on render thread: %s", exc)

        self.viewport.post_render_command(restore_renderer)

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
        """The full snapshot: the shared renderer-owned section + this
        front-end's own keys.

        The shared section is captured ON the render thread, which owns the live
        renderer. A timeout falls back to the keys this thread knows on its own —
        `backend`, which the window was handed at construction. The rest is left
        out rather than guessed off the proxy: `save_settings` merges, so the
        previous values stay on disk.
        """
        backend = self._backend_name
        shared: dict = {"backend": backend}
        try:
            future = self.renderer.request(
                lambda renderer: capture_shared(renderer, backend=backend),
            )
            shared = future.result(timeout=2.0)
        except TimeoutError:
            log.warning("Timed out waiting for renderer settings snapshot")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to snapshot renderer-owned state: %s", exc)
        return contribute(shared, self._contributed_session_state(), owned=QT_KEYS)

    def _contributed_session_state(self) -> dict:
        """This front-end's own settings keys — must equal
        `session_snapshot.QT_KEYS`, which `contribute` enforces.
        """
        out: dict = {}
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
        super().closeEvent(event)


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

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )

    saved_settings = load_settings()

    # Shared bring-up (change frontend-bringup-builder): resolve the startup
    # integrator, execution mode and backend, and run every refusal guard, in
    # the one canonical order — every flag-level refusal happens before the
    # backend probe touches the GPU. `skinny-gui` persists, so the settings dict
    # is offered: the persisted integrator feeds startup-integrator resolution
    # (a persisted sppm under an explicitly-forced --execution-mode megakernel
    # is refused, which the CLI-keyed validate_render_flags alone cannot see),
    # the persisted backend feeds selection, and the persisted --encoding
    # (a build dim, fixed for the session) feeds the neural build config. Only
    # the plan step runs here — the context and renderer are constructed later,
    # on the Qt render thread, from this plan.
    plan = plan_bringup(args, prog="skinny-gui", persisted=saved_settings)

    # Persisted flags (change session-settings-owner): an explicit CLI flag or
    # env var wins, else the persisted value, else the argparse default. These
    # are the flags `cli_common` documents as persisted on the interactive
    # front-ends; `skinny-gui` used to restore only the SPPM threshold and erase
    # the rest, so the documented behaviour held on `skinny` alone.
    def _persisted(key, cli_value):
        return resolve_persisted_flag(key, cli_value, saved_settings)

    sppm_glossy_roughness_value = _persisted(
        "sppm_glossy_roughness", args.sppm_glossy_roughness)
    neural_handoff_value = _persisted("neural_handoff", args.neural_handoff)
    neural_trainer_value = _persisted("neural_trainer", args.neural_trainer)
    train_precision_value = _persisted("train_precision", args.train_precision)
    online_training_value = bool(_persisted("online_training", args.online_training))

    app = QApplication(sys.argv)
    win = MainWindow(args.scene, args.gpu, args.usdMtlx, plan.execution_mode,
                     plan.bdpt_walk, args.integrator,
                     plan=plan,
                     neural_handoff=neural_handoff_value,
                     neural_trainer=neural_trainer_value,
                     train_precision=train_precision_value,
                     online_training=online_training_value,
                     reuse=args.reuse,
                     lobe_samplers=args.lobe_samplers,
                     backend=plan.backend,
                     requested_backend=args.backend,
                     encoding=plan.encoding,
                     sppm_glossy_roughness=sppm_glossy_roughness_value,
                     width=args.width, height=args.height,
                     spectral=plan.spectral,
                     mcp=bool(getattr(args, "mcp", False)),
                     mcp_port=getattr(args, "mcp_port", 0),
                     mcp_roots=resolve_mcp_roots(args))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
