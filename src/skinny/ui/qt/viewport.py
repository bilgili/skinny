"""Central render viewport widget.

Background thread runs ``renderer.render_headless()`` in a tight loop,
emits the latest RGBA8 frame via a Qt signal. Main thread blits the
frame into a ``QImage`` and paints. Mirrors the loop shape used by
``web_app.SkinnySession._render_loop`` minus the H264 encode step.
"""

from __future__ import annotations

import time
from threading import Lock

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from skinny.ui.qt.camera_input import CameraDispatcher


class _RenderWorker(QObject):
    """Lives on a ``QThread``. Loops ``render_headless()`` and signals out
    the most recent frame. Owns no Qt widgets — pure Vulkan + bytes.
    """

    frame_ready = Signal(bytes, int, int, int)  # pixels, w, h, accum_frame
    error = Signal(str)

    def __init__(self, renderer, lock: Lock) -> None:
        super().__init__()
        self.renderer = renderer
        self._lock = lock
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        prev = time.perf_counter()
        try:
            while self._running:
                now = time.perf_counter()
                dt = now - prev
                prev = now

                with self._lock:
                    self.renderer.update(dt)
                    pixels = self.renderer.render_headless()
                    w = int(self.renderer.width)
                    h = int(self.renderer.height)
                    accum = int(self.renderer.accum_frame)

                self.frame_ready.emit(pixels, w, h, accum)

                # Same throttle ladder web_app uses once accumulation
                # converges — keeps GPU + CPU idle when the image is done.
                if accum > 200:
                    time.sleep(0.5)
                elif accum > 50:
                    time.sleep(0.1)
                elif accum > 10:
                    time.sleep(0.03)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(repr(exc))


class RenderViewport(QWidget):
    """Central widget. Holds the latest frame as a ``QImage`` and blits
    it during ``paintEvent``. Camera input goes through ``CameraDispatcher``.
    """

    accum_changed = Signal(int)
    """Emitted every time a new frame arrives so the status bar can
    show the current accumulation count.
    """

    def __init__(self, renderer, parent=None) -> None:
        super().__init__(parent)
        self.renderer = renderer
        self._render_lock = Lock()
        self._image: QImage | None = None
        # Hold the raw bytes so QImage's no-copy view stays valid until the
        # next frame replaces it.
        self._image_buffer: bytes | None = None

        self._camera = CameraDispatcher(renderer)

        self._left = self._right = self._middle = False
        self._last_pos: tuple[float, float] | None = None
        self._wasd: dict[int, bool] = {}

        # Scene-pick arming for the BXDF visualiser. When set, the next
        # left-click captures the shading frame instead of starting a drag.
        self._pick_armed: bool = False
        self._pick_cb = None

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: black;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        # Start the worker thread.
        self._thread = QThread(self)
        self._worker = _RenderWorker(renderer, self._render_lock)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_render_error)
        self._thread.start()

        # Free-cam WASDQE poll: same cadence as the GLFW path's per-frame
        # update(). 16 ms timer keeps movement smooth without spamming.
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(16)
        self._move_timer.timeout.connect(self._poll_wasd)
        self._move_timer.start()

    # ── Frame plumbing ─────────────────────────────────────────────

    def _on_frame_ready(self, pixels: bytes, w: int, h: int, accum: int) -> None:
        # Hold both the bytes and the QImage view; QImage(bytes, …) is a
        # zero-copy view and the buffer must outlive every paint.
        self._image_buffer = pixels
        self._image = QImage(pixels, w, h, 4 * w, QImage.Format_RGBA8888)
        self.accum_changed.emit(accum)
        self.update()

    def _on_render_error(self, msg: str) -> None:
        # Surface to stderr; the status bar can pick this up via a hook later.
        import sys
        print(f"[render error] {msg}", file=sys.stderr)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        if self._image is None:
            painter.fillRect(self.rect(), Qt.black)
            return
        # Letterbox: preserve aspect.
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

    # ── Mouse / wheel / key ────────────────────────────────────────

    def arm_scene_pick(self, callback) -> None:
        """Next left-click in the viewport captures the shading frame and
        forwards it to ``callback(result_dict | None)`` via
        ``Renderer.request_scene_pick``.
        """
        self._pick_armed = True
        self._pick_cb = callback

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if event.button() == Qt.LeftButton and self._pick_armed:
            cb = self._pick_cb
            self._pick_armed = False
            self._pick_cb = None
            with self._render_lock:
                self.renderer.request_scene_pick(
                    float(pos.x()), float(pos.y()), cb,
                )
            self.setFocus(Qt.MouseFocusReason)
            return
        if event.button() == Qt.LeftButton:
            self._left = True
        elif event.button() == Qt.RightButton:
            self._right = True
        elif event.button() == Qt.MiddleButton:
            self._middle = True
        self._last_pos = (pos.x(), pos.y())
        self.setFocus(Qt.MouseFocusReason)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._left = False
        elif event.button() == Qt.RightButton:
            self._right = False
        elif event.button() == Qt.MiddleButton:
            self._middle = False

    def mouseMoveEvent(self, event) -> None:
        if self._last_pos is None:
            self._last_pos = (event.position().x(), event.position().y())
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]
        self._last_pos = (x, y)
        if self._left or self._right or self._middle:
            with self._render_lock:
                self._camera.drag(
                    dx, dy, left=self._left, right=self._right, middle=self._middle,
                )

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Qt wheel deltas are in eighths of a degree; one notch = 120.
        # GLFW yoff is 1/-1 per notch — match that.
        notches = event.angleDelta().y() / 120.0
        with self._render_lock:
            self._camera.zoom(notches)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._wasd[key] = True
            return
        if key == Qt.Key_C:
            with self._render_lock:
                self._camera.toggle_mode()
        elif key == Qt.Key_F:
            with self._render_lock:
                self._camera.reset()
        elif key == Qt.Key_F1 or key == Qt.Key_Space:
            self.renderer.show_hud = not self.renderer.show_hud
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if key in self._wasd:
            self._wasd[key] = False
            return
        super().keyReleaseEvent(event)

    def _poll_wasd(self) -> None:
        if getattr(self.renderer, "camera_mode", "orbit") != "free":
            return
        f = (1.0 if self._wasd.get(Qt.Key_W) else 0.0) - (1.0 if self._wasd.get(Qt.Key_S) else 0.0)
        r = (1.0 if self._wasd.get(Qt.Key_D) else 0.0) - (1.0 if self._wasd.get(Qt.Key_A) else 0.0)
        u = (1.0 if self._wasd.get(Qt.Key_E) else 0.0) - (1.0 if self._wasd.get(Qt.Key_Q) else 0.0)
        if f or r or u:
            with self._render_lock:
                self._camera.move(f, r, u, 0.016)

    # ── Resize / shutdown ──────────────────────────────────────────

    def request_resize(self, w: int, h: int) -> tuple[int, int]:
        """Resize the underlying renderer, return the actual ``(W, H)``
        it settled on after workgroup-multiple rounding.
        """
        with self._render_lock:
            self.renderer.resize(int(w), int(h))
            return int(self.renderer.width), int(self.renderer.height)

    def shutdown(self) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
