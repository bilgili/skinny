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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QMainWindow, QScrollArea, QWidget,
)

from skinny.renderer import Renderer
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
        self._sidebar_dock = dock

        # Bias the initial split: render gets the lion's share of width,
        # controls sit at 360px.
        self.resizeDocks(
            [dock, self._render_dock], [380, 1200], Qt.Horizontal,
        )

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

    def _ensure_debug_dock(self) -> DebugViewportDock:
        if self._debug_dock is not None:
            return self._debug_dock
        self._debug_dock = DebugViewportDock(
            ctx=self.ctx, renderer=self.renderer,
            main_lock=self.viewport._render_lock, parent=self,
        )
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

    def _set_debug_view(self, which: str) -> None:
        dock = self._ensure_debug_dock()
        if not dock.isVisible():
            dock.show()
        dock._view(which)

    def closeEvent(self, event) -> None:
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
