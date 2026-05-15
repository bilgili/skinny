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

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QMainWindow, QScrollArea, QWidget,
)

from skinny.debug_viewport import DebugViewport
from skinny.renderer import Renderer
from skinny.ui.build_app_ui import AppCallbacks, build_main_ui
from skinny.ui.qt.backend import QtTreeBuilder
from skinny.ui.qt.viewport import RenderViewport
from skinny.ui.qt.windows.bxdf import BXDFDock
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
        # to worry about in the desktop entry).
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

        # Central viewport.
        self.viewport = RenderViewport(self.renderer, parent=self)
        self.setCentralWidget(self.viewport)

        # Debug viewport: standalone GLFW window opened on demand. GLFW is
        # initialised lazily on first open so users who never click the
        # button pay nothing.
        self._debug_viewport: DebugViewport | None = None
        self._glfw_initialised: bool = False
        self._debug_timer: QTimer | None = None

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
            debug_view_top=lambda: self._set_debug_view("top"),
            debug_view_left=lambda: self._set_debug_view("left"),
            debug_view_back=lambda: self._set_debug_view("back"),
        )
        tree = build_main_ui(self.renderer, callbacks=cb)

        sidebar_inner = QWidget()
        self._tree_builder = QtTreeBuilder(tree, sidebar_inner)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar_inner)
        scroll.setMinimumWidth(360)
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _stub(self, name: str) -> None:
        self.statusBar().showMessage(f"{name}: not yet ported (Phase 7)", 3000)

    def _open_scene_graph(self) -> None:
        if self._scene_graph_dock is None:
            self._scene_graph_dock = SceneGraphDock(self.renderer, parent=self)
            self.addDockWidget(Qt.RightDockWidgetArea, self._scene_graph_dock)
        self._scene_graph_dock.show()
        self._scene_graph_dock.raise_()

    def _open_bxdf(self) -> None:
        if self._bxdf_dock is None:
            self._bxdf_dock = BXDFDock(self.renderer, self.viewport, parent=self)
            self.addDockWidget(Qt.RightDockWidgetArea, self._bxdf_dock)
        self._bxdf_dock.show()
        self._bxdf_dock.raise_()

    def _open_material_graph(self) -> None:
        if self._material_graph_dock is None:
            self._material_graph_dock = MaterialGraphDock(self.renderer, parent=self)
            self.addDockWidget(Qt.BottomDockWidgetArea, self._material_graph_dock)
        self._material_graph_dock.show()
        self._material_graph_dock.raise_()

    def _ensure_debug_viewport(self) -> DebugViewport:
        """Lazy-init GLFW + DebugViewport on first request."""
        if self._debug_viewport is not None:
            return self._debug_viewport
        if not self._glfw_initialised:
            import glfw
            if not glfw.init():
                raise RuntimeError("Failed to init GLFW for debug viewport")
            self._glfw_initialised = True
        shader_dir = Path(__file__).resolve().parents[1].parent / "shaders"
        self._debug_viewport = DebugViewport(
            vk_ctx=self.ctx, shader_dir=shader_dir,
        )
        self._debug_viewport.attach_renderer(self.renderer)
        self.renderer.debug_viewport = self._debug_viewport
        # Drive update + render + glfw.poll_events on a Qt timer. 33ms (~30Hz)
        # is enough for a debug overlay and keeps the render-thread Vulkan
        # work uncontended for as much wall-clock as possible.
        self._debug_timer = QTimer(self)
        self._debug_timer.setInterval(33)
        self._debug_timer.timeout.connect(self._debug_tick)
        self._debug_timer.start()
        return self._debug_viewport

    def _toggle_debug_viewport(self) -> None:
        try:
            dv = self._ensure_debug_viewport()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Debug viewport unavailable: {exc}", 5000)
            return
        dv.toggle()

    def _set_debug_view(self, which: str) -> None:
        dv = self._debug_viewport
        if dv is None:
            return
        if not dv.is_open:
            dv.open()
        if which == "top":
            dv.view_top()
        elif which == "left":
            dv.view_left()
        elif which == "back":
            dv.view_back()

    def _debug_tick(self) -> None:
        dv = self._debug_viewport
        if dv is None:
            return
        import glfw
        glfw.poll_events()
        if not dv.is_open:
            return
        # Acquire the main viewport's render lock so GPU work doesn't
        # interleave with renderer.render_headless() on the render thread.
        # The debug viewport reads renderer state (camera, mesh transforms)
        # which the headless render also mutates under that same lock.
        with self.viewport._render_lock:
            dv.update(0.033)
            dv.render(self.renderer)

    def closeEvent(self, event) -> None:
        if self._debug_timer is not None:
            self._debug_timer.stop()
        self.viewport.shutdown()
        if self._debug_viewport is not None:
            try:
                self._debug_viewport.destroy()
            except Exception:
                pass
        if self._glfw_initialised:
            try:
                import glfw
                glfw.terminate()
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
