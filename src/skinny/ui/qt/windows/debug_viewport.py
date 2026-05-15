"""Embedded Qt dock wrapping ``DebugViewport(embedded=True)``.

Owns a render-to-image debug viewport against the main ``VulkanContext``.
A QTimer drives ``DebugViewport.render_embedded()`` → ``QImage`` blit.
Mouse + key handlers drive the debug viewport's orbit / free camera and
the various toggles the legacy GLFW window exposed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget,
)

from skinny.debug_viewport import DebugViewport


class _DebugCanvas(QWidget):
    """Central widget. Paints the latest RGBA8 frame; forwards mouse
    + wheel events to the parent dock for camera control.
    """

    def __init__(
        self, on_drag, on_wheel, on_press, on_release,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #0d0d12;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._on_drag = on_drag
        self._on_wheel = on_wheel
        self._on_press = on_press
        self._on_release = on_release
        self._image: QImage | None = None
        self._buffer: bytes | None = None
        self._last_pos: tuple[float, float] | None = None
        self._left = self._right = False

    def set_frame(self, pixels: bytes, w: int, h: int) -> None:
        # QImage with a borrowed bytes view; keep ``_buffer`` alive until
        # the next frame replaces it.
        self._buffer = pixels
        self._image = QImage(pixels, w, h, 4 * w, QImage.Format_RGBA8888)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if self._image is None:
            painter.fillRect(self.rect(), Qt.black)
            return
        target = self.rect()
        sw, sh = self._image.width(), self._image.height()
        scale = min(target.width() / sw, target.height() / sh)
        dw, dh = int(sw * scale), int(sh * scale)
        x = (target.width() - dw) // 2
        y = (target.height() - dh) // 2
        painter.fillRect(target, Qt.black)
        painter.drawImage(
            x, y, self._image.scaled(
                dw, dh, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            ),
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._left = True
        elif event.button() == Qt.RightButton:
            self._right = True
        self._last_pos = (event.position().x(), event.position().y())
        self.setFocus(Qt.MouseFocusReason)
        self._on_press(event.button())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._left = False
        elif event.button() == Qt.RightButton:
            self._right = False
        self._on_release(event.button())

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            self._last_pos = (event.position().x(), event.position().y())
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]
        self._last_pos = (x, y)
        if self._left or self._right:
            self._on_drag(dx, dy, self._left, self._right)

    def wheelEvent(self, event) -> None:
        notches = event.angleDelta().y() / 120.0
        if notches != 0:
            self._on_wheel(notches)


class DebugViewportDock(QDockWidget):
    """Qt-embedded debug viewport. Lazy-builds GPU resources when the
    dock is first shown so the main app starts up fast.
    """

    TICK_MS = 33  # ~30 Hz blit; cheap line-rasteriser, plenty.

    def __init__(self, ctx, renderer, main_lock, parent: QWidget | None = None) -> None:
        super().__init__("Camera Debug View", parent)
        self.ctx = ctx
        self.renderer = renderer
        self._main_lock = main_lock  # ``viewport._render_lock`` from RenderViewport
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        shader_dir = Path(__file__).resolve().parents[2].parent / "shaders"
        self._dv = DebugViewport(
            vk_ctx=ctx, shader_dir=shader_dir,
            width=960, height=720, embedded=True,
        )
        self._dv.attach_renderer(renderer)
        # Register so other paths (renderer-side keyboard hooks) that
        # peek at ``renderer.debug_viewport`` find it.
        renderer.debug_viewport = self._dv

        self._wasd: dict[int, bool] = {}
        self._build_widgets()

        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    # ── Layout ────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        bar = QHBoxLayout()
        for label, cb in (
            ("Top",  lambda: self._view("top")),
            ("Left", lambda: self._view("left")),
            ("Back", lambda: self._view("back")),
            ("Reset", self._dv._reset_debug_camera),
            ("Toggle Mode", self._dv._toggle_cam_mode),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(cb)
            bar.addWidget(btn)
        bar.addStretch(1)
        outer.addLayout(bar)

        self._canvas = _DebugCanvas(
            on_drag=self._on_drag,
            on_wheel=self._on_wheel,
            on_press=self._on_press,
            on_release=self._on_release,
        )
        outer.addWidget(self._canvas, stretch=1)

        self.setWidget(host)

    def showEvent(self, event) -> None:
        # Lazy GPU resource build on first show.
        if not self._dv.is_open:
            try:
                self._dv.open()
            except Exception as exc:  # noqa: BLE001
                print(f"[debug viewport] open failed: {exc}")
                return
        # Match GPU image to widget size before the first render.
        self._sync_size()
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_size()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        try:
            self._dv.destroy()
        except Exception:
            pass
        super().closeEvent(event)

    def _sync_size(self) -> None:
        if not self._dv.is_open:
            return
        sz = self._canvas.size()
        w = max(int(sz.width()), 64)
        h = max(int(sz.height()), 64)
        try:
            with self._main_lock:
                self._dv.resize_embedded(w, h)
        except Exception as exc:  # noqa: BLE001
            print(f"[debug viewport] resize failed: {exc}")

    # ── Tick ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self._dv.is_open:
            return
        # Free-cam WASDQE poll.
        if self._dv.camera_mode == "free":
            f = (1.0 if self._wasd.get(Qt.Key_W) else 0.0) - (1.0 if self._wasd.get(Qt.Key_S) else 0.0)
            r = (1.0 if self._wasd.get(Qt.Key_D) else 0.0) - (1.0 if self._wasd.get(Qt.Key_A) else 0.0)
            u = (1.0 if self._wasd.get(Qt.Key_E) else 0.0) - (1.0 if self._wasd.get(Qt.Key_Q) else 0.0)
            if f or r or u:
                self._dv.free_camera.move(float(f), float(r), float(u), self.TICK_MS / 1000.0)
        try:
            with self._main_lock:
                pixels = self._dv.render_embedded(self.renderer)
        except Exception as exc:  # noqa: BLE001
            print(f"[debug viewport] render failed: {exc}")
            return
        if pixels is None:
            return
        self._canvas.set_frame(pixels, self._dv._width, self._dv._height)

    # ── Input forwarding ──────────────────────────────────────────

    def _on_press(self, _button) -> None:
        pass

    def _on_release(self, _button) -> None:
        pass

    def _on_drag(self, dx: float, dy: float, left: bool, right: bool) -> None:
        if self._dv.camera_mode == "orbit":
            if left:
                self._dv.orbit_camera.orbit(dx, dy)
            elif right:
                self._dv.orbit_camera.pan(dx, dy)
        else:
            if left:
                self._dv.free_camera.look(dx, dy)

    def _on_wheel(self, notches: float) -> None:
        cam = (
            self._dv.orbit_camera if self._dv.camera_mode == "orbit"
            else self._dv.free_camera
        )
        cam.zoom(float(notches))

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._wasd[key] = True
            return
        if key == Qt.Key_C:
            self._dv._toggle_cam_mode()
        elif key == Qt.Key_F:
            self._dv._reset_debug_camera()
        elif key == Qt.Key_M:
            self._dv.show_mesh_wires = not self._dv.show_mesh_wires
        elif key == Qt.Key_G:
            self._dv.show_grid = not self._dv.show_grid
        elif key == Qt.Key_P:
            self._dv.show_focus_plane = not self._dv.show_focus_plane
        elif key == Qt.Key_I:
            self._dv.show_render_area = not self._dv.show_render_area
        elif key == Qt.Key_O:
            self._dv.ortho_mode = not self._dv.ortho_mode
        elif key == Qt.Key_D:
            self._dv.show_dof_planes = not self._dv.show_dof_planes
        elif key == Qt.Key_T:
            self._dv.view_top()
        elif key == Qt.Key_B:
            self._dv.view_back()
        elif key == Qt.Key_L:
            self._dv.view_left()
        elif key == Qt.Key_Space:
            self._dv.show_hud = not self._dv.show_hud
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() in self._wasd:
            self._wasd[event.key()] = False
            return
        super().keyReleaseEvent(event)

    # ── View shortcuts ────────────────────────────────────────────

    def _view(self, which: str) -> None:
        if which == "top":
            self._dv.view_top()
        elif which == "left":
            self._dv.view_left()
        elif which == "back":
            self._dv.view_back()
