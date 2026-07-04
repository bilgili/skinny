"""Embedded Qt dock for the Camera Debug viewport, render-thread ownership.

The ``DebugViewport(embedded=True)`` GPU object lives on the render worker (as
``renderer.debug_viewport``); the worker renders it each frame and emits a
``DebugFrame`` the dock blits into a ``QImage``. This dock is passive — mouse +
key handlers post camera/display commands to the worker, and show/hide/resize/
close post the viewport lifecycle. No GPU work runs on the GUI thread.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout,
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
        self._notice: str | None = None
        self._last_pos: tuple[float, float] | None = None
        self._left = self._right = False

    def set_frame(self, pixels: bytes, w: int, h: int) -> None:
        # QImage with a borrowed bytes view; keep ``_buffer`` alive until
        # the next frame replaces it.
        self._buffer = pixels
        self._image = QImage(pixels, w, h, 4 * w, QImage.Format_RGBA8888)
        self.update()

    def set_notice(self, text: str) -> None:
        self._notice = text
        self._image = None
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if self._image is None:
            painter.fillRect(self.rect(), Qt.black)
            if self._notice:
                painter.setPen(Qt.gray)
                painter.drawText(
                    self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._notice,
                )
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


# ── Worker-side DebugViewport helpers ─────────────────────────────
# The DebugViewport owns GPU resources against the worker's VulkanContext, so it
# lives on the render worker as ``renderer.debug_viewport``. These run inside
# ``proxy.post(...)`` closures on the worker; the dock never touches it directly.

def _worker_debug_create(renderer, shader_dir, w, h) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is None:
        dv = DebugViewport(
            vk_ctx=renderer.ctx, shader_dir=shader_dir,
            width=max(w, 64), height=max(h, 64), embedded=True,
        )
        dv.attach_renderer(renderer)
        renderer.debug_viewport = dv
    if not dv.is_open:
        dv.open()
    dv.resize_embedded(max(w, 64), max(h, 64))
    renderer._debug_viewport_active = True


def _worker_debug_set_active(renderer, active: bool) -> None:
    renderer._debug_viewport_active = bool(active)


def _worker_debug_resize(renderer, w, h) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is not None and dv.is_open:
        dv.resize_embedded(max(w, 64), max(h, 64))


def _worker_debug_destroy(renderer) -> None:
    renderer._debug_viewport_active = False
    dv = getattr(renderer, "debug_viewport", None)
    if dv is not None:
        try:
            dv.destroy()
        except Exception:  # noqa: BLE001
            pass
    renderer.debug_viewport = None


def _worker_debug_drag(renderer, dx, dy, left, right) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is None:
        return
    if dv.camera_mode == "orbit":
        if left:
            dv.orbit_camera.orbit(dx, dy)
        elif right:
            dv.orbit_camera.pan(dx, dy)
    elif left:
        dv.free_camera.look(dx, dy)


def _worker_debug_wheel(renderer, notches) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is None:
        return
    cam = dv.orbit_camera if dv.camera_mode == "orbit" else dv.free_camera
    cam.zoom(float(notches))


def _worker_debug_move(renderer, f, r, u, dt) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is None or dv.camera_mode != "free":
        return
    dv.free_camera.move(float(f), float(r), float(u), dt)


def _worker_debug_call(renderer, method: str) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is not None:
        getattr(dv, method)()


def _worker_debug_toggle(renderer, attr: str) -> None:
    dv = getattr(renderer, "debug_viewport", None)
    if dv is not None:
        setattr(dv, attr, not getattr(dv, attr))


class DebugViewportDock(QDockWidget):
    """Qt-embedded Camera Debug viewport. Under render-thread ownership the
    ``DebugViewport`` GPU object lives on the render worker (as
    ``renderer.debug_viewport``); this dock is a passive surface — it blits the
    worker-emitted frames and posts camera/display input + lifecycle commands.
    """

    TICK_MS = 33  # ~30 Hz free-cam WASD move posting.

    def __init__(self, renderer, viewport, parent: QWidget | None = None) -> None:
        super().__init__("Camera Debug View", parent)
        self.renderer = renderer  # QtRendererProxy
        self.viewport = viewport  # RenderViewport (forwards debug_frame_ready)
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self._shader_dir = Path(__file__).resolve().parents[2].parent / "shaders"
        self._created = False
        self._wasd: dict[int, bool] = {}
        self._build_widgets()

        self.viewport.debug_frame_ready.connect(self._on_debug_frame)
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(self.TICK_MS)
        self._move_timer.timeout.connect(self._poll_wasd)

    # ── Layout ────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        bar = QHBoxLayout()
        for label, method in (
            ("Top", "view_top"),
            ("Left", "view_left"),
            ("Back", "view_back"),
            ("Reset", "_reset_debug_camera"),
            ("Toggle Mode", "_toggle_cam_mode"),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, m=method: self._post_call(m),
            )
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

    # ── Lifecycle ─────────────────────────────────────────────────

    def _canvas_size(self) -> tuple[int, int]:
        sz = self._canvas.size()
        return max(int(sz.width()), 64), max(int(sz.height()), 64)

    def showEvent(self, event) -> None:
        # The DebugViewport is a Vulkan graphics rasteriser; the native Metal
        # backend is compute-only (metal-tool-dock-render P2 ports it via a
        # compute rasteriser). Until then, show a notice instead of posting the
        # create/render — which would raise on the worker every frame.
        if getattr(self.renderer, "_backend_name", "") == "metal":
            self._canvas.set_notice(
                "Camera Debug view renders on the Vulkan backend only.\n"
                "(Metal compute-rasteriser port pending.)"
            )
            super().showEvent(event)
            return
        w, h = self._canvas_size()
        sd = self._shader_dir
        self.renderer.post(
            lambda r, sd=sd, w=w, h=h: _worker_debug_create(r, sd, w, h),
            coalesce_key="debug_create",
        )
        self._created = True
        self._move_timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._move_timer.stop()
        self.renderer.post(
            lambda r: _worker_debug_set_active(r, False),
            coalesce_key="debug_active",
        )
        super().hideEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._created:
            return
        w, h = self._canvas_size()
        self.renderer.post(
            lambda r, w=w, h=h: _worker_debug_resize(r, w, h),
            coalesce_key="debug_resize",
        )

    def closeEvent(self, event) -> None:
        self._move_timer.stop()
        self.renderer.post(_worker_debug_destroy)
        self._created = False
        super().closeEvent(event)

    def _on_debug_frame(self, frame) -> None:
        self._canvas.set_frame(frame.pixels, frame.width, frame.height)

    # ── Input forwarding (posted to the worker) ───────────────────

    def _post_call(self, method: str) -> None:
        self.renderer.post(lambda r, m=method: _worker_debug_call(r, m))

    def _post_toggle(self, attr: str) -> None:
        self.renderer.post(lambda r, a=attr: _worker_debug_toggle(r, a))

    def _on_press(self, _button) -> None:
        pass

    def _on_release(self, _button) -> None:
        pass

    def _on_drag(self, dx: float, dy: float, left: bool, right: bool) -> None:
        self.renderer.post(
            lambda r, dx=dx, dy=dy, left=left, right=right:
                _worker_debug_drag(r, dx, dy, left, right),
        )

    def _on_wheel(self, notches: float) -> None:
        self.renderer.post(lambda r, n=notches: _worker_debug_wheel(r, n))

    def _poll_wasd(self) -> None:
        f = ((1.0 if self._wasd.get(Qt.Key_W) else 0.0)
             - (1.0 if self._wasd.get(Qt.Key_S) else 0.0))
        r = ((1.0 if self._wasd.get(Qt.Key_D) else 0.0)
             - (1.0 if self._wasd.get(Qt.Key_A) else 0.0))
        u = ((1.0 if self._wasd.get(Qt.Key_E) else 0.0)
             - (1.0 if self._wasd.get(Qt.Key_Q) else 0.0))
        if not (f or r or u):
            return
        dt = self.TICK_MS / 1000.0
        self.renderer.post(
            lambda rr, f=f, r=r, u=u, dt=dt: _worker_debug_move(rr, f, r, u, dt),
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._wasd[key] = True
            return
        calls = {
            Qt.Key_C: ("call", "_toggle_cam_mode"),
            Qt.Key_F: ("call", "_reset_debug_camera"),
            Qt.Key_M: ("toggle", "show_mesh_wires"),
            Qt.Key_G: ("toggle", "show_grid"),
            Qt.Key_P: ("toggle", "show_focus_plane"),
            Qt.Key_I: ("toggle", "show_render_area"),
            Qt.Key_O: ("toggle", "ortho_mode"),
            Qt.Key_T: ("call", "view_top"),
            Qt.Key_B: ("call", "view_back"),
            Qt.Key_L: ("call", "view_left"),
            Qt.Key_Space: ("toggle", "show_hud"),
        }
        action = calls.get(key)
        if action is None:
            super().keyPressEvent(event)
            return
        kind, name = action
        if kind == "call":
            self._post_call(name)
        else:
            self._post_toggle(name)

    def keyReleaseEvent(self, event) -> None:
        if event.key() in self._wasd:
            self._wasd[event.key()] = False
            return
        super().keyReleaseEvent(event)
