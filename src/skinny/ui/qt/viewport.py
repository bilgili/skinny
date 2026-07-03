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

from skinny.ui.gizmo_input import GizmoMouseController
from skinny.ui.qt.camera_input import CameraDispatcher
from skinny.ui.qt.render_session import RenderCommandQueue


class _RenderWorker(QObject):
    """Lives on a ``QThread``. Loops ``render_headless()`` and signals out
    the most recent frame. Owns no Qt widgets — pure Vulkan + bytes.
    """

    frame_ready = Signal(bytes, int, int, int)  # pixels, w, h, accum_frame
    error = Signal(str)

    def __init__(
        self,
        renderer,
        lock: Lock,
        command_queue: RenderCommandQueue,
        online_training: bool = False,
    ) -> None:
        super().__init__()
        self.renderer = renderer
        self._lock = lock
        self._commands = command_queue
        self._running = True
        # --online-training (change online-training-trigger): enable lazily once
        # the scene is built, then drive the per-frame tick on this render thread.
        self._online_training_requested = bool(online_training)
        self._online_training_enabled = False
        self._online_training_refused = False
        self._online_training_waiting = False
        # Mirror the user's intent onto the renderer for the configuration matrix
        # (change online-training-observability); the matrix's online-training row
        # carries the REFUSED/WAITING/APPROVED reason, so the worker no longer
        # prints its own one-shot refused/armed lines.
        if renderer is not None:
            renderer._online_training_requested = bool(online_training)

    def stop(self) -> None:
        self._running = False

    def _maybe_online_training(self, dt: float) -> None:
        """Enable online training once the scene is built and drive its per-frame
        drain — runs under the render lock on the worker (render) thread."""
        if not self._online_training_requested or self.renderer is None:
            return
        if not self._online_training_enabled:
            # Defer the prerequisite check until the scene is built (records can't
            # drain before then). `_backend_render_ready` is backend-aware: the
            # native Metal backend never allocates Vulkan `descriptor_sets` (it
            # binds by name), so gating on those left online training permanently
            # disabled on Metal — use the scene-bindings readiness both set.
            if not self.renderer._backend_render_ready:
                return  # scene not built yet; retry next frame
            ok, _reason = self.renderer.can_online_train()
            if not ok:
                # Two prerequisites with different lifetimes. The execution mode
                # is fixed for the session, so a non-wavefront miss is permanent:
                # give up. A missing *neural proposal* is transient — the
                # Proposals combobox can activate one at runtime — so keep polling
                # and start training the moment it does. The matrix's
                # online-training row surfaces the REFUSED/WAITING reason.
                if not self.renderer.online_train_execution_supported():
                    self._online_training_refused = True
                    self._online_training_requested = False
                    return
                self._online_training_waiting = True
                return  # transient: retry next frame
            self.renderer.enable_online_training()
            self._online_training_enabled = True
        self.renderer.online_training_tick()

    def run(self) -> None:
        prev = time.perf_counter()
        try:
            while self._running:
                now = time.perf_counter()
                dt = now - prev
                prev = now

                with self._lock:
                    for command in self._commands.drain():
                        try:
                            result = command.callback(self.renderer)
                        except Exception as exc:  # noqa: BLE001
                            if command.reply is not None:
                                command.reply.set_exception(exc)
                            self.error.emit(repr(exc))
                        else:
                            if command.reply is not None:
                                command.reply.set_result(result)
                    self.renderer.update(dt)
                    self._maybe_online_training(dt)
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
        finally:
            try:
                self.renderer.disable_online_training()
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

    def __init__(self, renderer, parent=None, online_training: bool = False) -> None:
        super().__init__(parent)
        self.renderer = renderer
        self._render_lock = Lock()
        self._commands = RenderCommandQueue()
        self._image: QImage | None = None
        # Hold the raw bytes so QImage's no-copy view stays valid until the
        # next frame replaces it.
        self._image_buffer: bytes | None = None

        self._camera = CameraDispatcher(renderer)

        # Rotate-gizmo interaction. The controller arbitrates gizmo-vs-camera on
        # press and owns the drag lifecycle; hit-tests are best-effort so the GUI
        # never waits for an in-flight render.
        self._gizmo = GizmoMouseController()

        self._left = self._right = self._middle = False
        self._last_pos: tuple[float, float] | None = None
        self._wasd: dict[int, bool] = {}

        # Scene-pick arming for the BXDF visualiser. When set, the next
        # left-click captures the shading frame instead of starting a drag.
        self._pick_armed: bool = False
        self._pick_cb = None

        # Zoom-rect select. `Z` arms; next left-drag picks a rect (render-
        # pixel space); release commits as the viewport sub-region. `X`
        # resets to full frame.
        self._zoom_arming: bool = False
        self._zoom_dragging: bool = False
        self._zoom_start_px: tuple[float, float] = (0.0, 0.0)

        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: black;")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

        # Start the worker thread.
        self._thread = QThread(self)
        self._worker = _RenderWorker(
            renderer, self._render_lock, self._commands, online_training,
        )
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

    def post_render_command(
        self,
        callback,
        *,
        coalesce_key: str | None = None,
    ) -> None:
        """Run ``callback(renderer)`` on the render worker before a later frame."""
        self._commands.post(callback, coalesce_key=coalesce_key)

    def _try_with_renderer(self, callback, default=None):
        """Best-effort immediate renderer read for hit-tests that need a result.

        These paths used to block the GUI while waiting for the render lock. If a
        frame is in flight, keep the UI responsive and let the gesture miss.
        """
        if not self._render_lock.acquire(blocking=False):
            return default
        try:
            return callback(self.renderer)
        finally:
            self._render_lock.release()

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

    def _widget_to_render_pixel(
        self, wx: float, wy: float,
    ) -> tuple[float, float] | None:
        """Inverse of the letterboxed blit in paintEvent. Returns
        ``(render_x, render_y)`` in pixel coords of the underlying frame,
        or ``None`` when the click falls outside the rendered image.
        """
        if self._image is None:
            return None
        target = self.rect()
        sw, sh = self._image.width(), self._image.height()
        scale = min(target.width() / sw, target.height() / sh)
        if scale <= 0:
            return None
        dw, dh = int(sw * scale), int(sh * scale)
        x_off = (target.width() - dw) // 2
        y_off = (target.height() - dh) // 2
        rx = (wx - x_off) / scale
        ry = (wy - y_off) / scale
        if rx < 0 or ry < 0 or rx >= sw or ry >= sh:
            return None
        return rx, ry

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if event.button() != Qt.LeftButton:
            if event.button() == Qt.RightButton:
                self._right = True
            elif event.button() == Qt.MiddleButton:
                self._middle = True
            self._last_pos = (pos.x(), pos.y())
            self.setFocus(Qt.MouseFocusReason)
            return

        # Left press: one precedence ladder — shift-autofocus → scene-pick →
        # zoom-rect → gizmo → camera — resolved by the gizmo controller so the
        # gizmo is grabbable and never shadowed by the camera (or vice versa).
        mapped = self._widget_to_render_pixel(pos.x(), pos.y())
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        action = self._try_with_renderer(
            lambda renderer: self._gizmo.on_press(
                renderer, mapped,
                shift=shift,
                pick_armed=self._pick_armed,
                zoom_arming=self._zoom_arming,
            ),
            default="camera",
        )

        if action == "autofocus":
            if mapped is not None:
                self.post_render_command(
                    lambda renderer, mapped=mapped: renderer.autofocus_at_pixel(
                        float(mapped[0]), float(mapped[1]),
                    ),
                    coalesce_key="autofocus",
                )
            self.setFocus(Qt.MouseFocusReason)
            return
        if action == "pick":
            if mapped is None:
                # Click outside the rendered image; ignore and stay armed.
                return
            cb = self._pick_cb
            self._pick_armed = False
            self._pick_cb = None
            self.post_render_command(
                lambda renderer, mapped=mapped, cb=cb: renderer.request_scene_pick(
                    float(mapped[0]), float(mapped[1]), cb,
                ),
            )
            self.setFocus(Qt.MouseFocusReason)
            return
        if action == "zoom":
            if mapped is not None:
                self._zoom_dragging = True
                self._zoom_start_px = (float(mapped[0]), float(mapped[1]))
                self.post_render_command(
                    lambda renderer, mapped=mapped: renderer.set_zoom_drag_overlay(
                        (mapped[0], mapped[1], mapped[0], mapped[1]),
                    ),
                    coalesce_key="zoom-overlay",
                )
            self.setFocus(Qt.MouseFocusReason)
            return
        if action == "gizmo":
            # Drag is live (controller began it). Do not arm the camera.
            self._last_pos = (pos.x(), pos.y())
            self.setFocus(Qt.MouseFocusReason)
            return

        # action == "camera"
        self._left = True
        self._last_pos = (pos.x(), pos.y())
        self.setFocus(Qt.MouseFocusReason)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._zoom_dragging:
            pos = event.position()
            mapped = self._widget_to_render_pixel(pos.x(), pos.y())
            def commit(renderer, mapped=mapped) -> None:
                if mapped is not None:
                    renderer.commit_zoom_rect(
                        self._zoom_start_px, (float(mapped[0]), float(mapped[1])),
                    )
                renderer.set_zoom_drag_overlay(None)
            self.post_render_command(commit, coalesce_key="zoom-overlay")
            self._zoom_dragging = False
            self._zoom_arming = False
            return
        if event.button() == Qt.LeftButton:
            ended = self._try_with_renderer(
                lambda renderer: self._gizmo.on_release(renderer),
                default=False,
            )
            if ended:
                return
            self._left = False
        elif event.button() == Qt.RightButton:
            self._right = False
        elif event.button() == Qt.MiddleButton:
            self._middle = False

    def mouseMoveEvent(self, event) -> None:
        if self._zoom_dragging:
            pos = event.position()
            mapped = self._widget_to_render_pixel(pos.x(), pos.y())
            if mapped is not None:
                self.post_render_command(
                    lambda renderer, mapped=mapped: renderer.set_zoom_drag_overlay((
                        self._zoom_start_px[0], self._zoom_start_px[1],
                        float(mapped[0]), float(mapped[1]),
                    )),
                    coalesce_key="zoom-overlay",
                )
            return

        x, y = event.position().x(), event.position().y()
        # Gizmo drag (consumes the move) or, when idle, ring hover highlight.
        mapped = self._widget_to_render_pixel(x, y)
        any_button = self._left or self._right or self._middle
        consumed = self._try_with_renderer(
            lambda renderer: self._gizmo.on_move(
                renderer, mapped,
                any_button_down=any_button, zoom_dragging=False,
            ),
            default=False,
        )
        if consumed:
            self._last_pos = (x, y)
            return

        if self._last_pos is None:
            self._last_pos = (x, y)
            return
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]
        self._last_pos = (x, y)
        if self._left or self._right or self._middle:
            self.post_render_command(
                lambda _renderer, dx=dx, dy=dy, left=self._left,
                right=self._right, middle=self._middle: self._camera.drag(
                    dx, dy, left=left, right=right, middle=middle,
                ),
                coalesce_key="camera-drag",
            )

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Qt wheel deltas are in eighths of a degree; one notch = 120.
        # GLFW yoff is 1/-1 per notch — match that.
        notches = event.angleDelta().y() / 120.0
        self.post_render_command(
            lambda _renderer, notches=notches: self._camera.zoom(notches),
            coalesce_key="camera-zoom",
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._wasd[key] = True
            return
        if key == Qt.Key_C:
            self.post_render_command(lambda _renderer: self._camera.toggle_mode())
        elif key == Qt.Key_F:
            self.post_render_command(lambda _renderer: self._camera.reset())
        elif key == Qt.Key_F1:
            self.post_render_command(
                lambda renderer: setattr(renderer, "show_hud", not renderer.show_hud),
            )
        elif key == Qt.Key_Space:
            def cycle(renderer) -> None:
                mode = renderer.gizmo_cycle_mode()
                print(f"[Gizmo mode: {mode.name}]")
            self.post_render_command(cycle)
        elif key == Qt.Key_L:
            def toggle_focus(renderer) -> None:
                renderer.show_focus_overlay = not renderer.show_focus_overlay
                print(f"[Focus overlay: {'on' if renderer.show_focus_overlay else 'off'}]")
            self.post_render_command(toggle_focus)
        elif key == Qt.Key_V:
            def toggle_vignette(renderer) -> None:
                renderer.lens_vignette_debug = not renderer.lens_vignette_debug
                renderer._material_version += 1
                print(
                    f"[Lens vignette debug: {'on' if renderer.lens_vignette_debug else 'off'}"
                    " — green=ray succeeds, red=clipped]"
                )
            self.post_render_command(toggle_vignette)
        elif key == Qt.Key_Z:
            self._zoom_arming = True
            print("[Zoom: drag a rectangle, release to apply]")
        elif key == Qt.Key_X:
            def reset_zoom(renderer) -> None:
                renderer.reset_zoom_rect()
                renderer.set_zoom_drag_overlay(None)
            self.post_render_command(reset_zoom, coalesce_key="zoom-overlay")
            self._zoom_arming = False
            self._zoom_dragging = False
            print("[Zoom: reset]")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if key in self._wasd:
            self._wasd[key] = False
            return
        super().keyReleaseEvent(event)

    def _poll_wasd(self) -> None:
        f = (1.0 if self._wasd.get(Qt.Key_W) else 0.0) - (1.0 if self._wasd.get(Qt.Key_S) else 0.0)
        r = (1.0 if self._wasd.get(Qt.Key_D) else 0.0) - (1.0 if self._wasd.get(Qt.Key_A) else 0.0)
        u = (1.0 if self._wasd.get(Qt.Key_E) else 0.0) - (1.0 if self._wasd.get(Qt.Key_Q) else 0.0)
        if f or r or u:
            def move(renderer, f=f, r=r, u=u) -> None:
                if getattr(renderer, "camera_mode", "orbit") == "free":
                    self._camera.move(f, r, u, 0.016)
            self.post_render_command(move, coalesce_key="camera-move")

    # ── Resize / shutdown ──────────────────────────────────────────

    def request_resize(self, w: int, h: int) -> tuple[int, int]:
        """Resize the underlying renderer, return the actual ``(W, H)``
        it settled on after workgroup-multiple rounding.
        """
        self.post_render_command(
            lambda renderer, w=int(w), h=int(h): renderer.resize(w, h),
            coalesce_key="resize",
        )
        return int(w), int(h)

    def shutdown(self) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
