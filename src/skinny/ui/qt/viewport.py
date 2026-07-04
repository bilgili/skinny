"""Central render viewport widget.

Background thread runs ``renderer.render_headless()`` in a tight loop,
emits the latest RGBA8 frame via a Qt signal. Main thread blits the
frame into a ``QImage`` and paints. Mirrors the loop shape used by
``web_app.SkinnySession._render_loop`` minus the H264 encode step.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from skinny.backend_select import make_context
from skinny.cli_common import INTEGRATOR_INDEX, apply_sppm_glossy_roughness, resolve_encoding
from skinny.params import _snapshot_params, build_all_params
from skinny.renderer import Renderer
from skinny.sampling.neural_weights import Encoding, NeuralBuildConfig
from skinny.sampling import parse_lobe_samplers
from skinny.ui.gizmo_input import GizmoMouseController
from skinny.ui.qt.camera_input import CameraDispatcher
from skinny.ui.qt.render_session import (
    FrameSnapshot,
    QtRendererConfig,
    QtRendererProxy,
    RenderCommandQueue,
    RendererStateSnapshot,
    choice_names_from_renderer,
    renderer_online_status,
)


class _RenderWorker(QObject):
    """Lives on a ``QThread``. Loops ``render_headless()`` and signals out
    the most recent frame. Owns no Qt widgets — pure Vulkan + bytes.
    """

    frame_ready = Signal(object)  # FrameSnapshot
    state_ready = Signal(object)  # RendererStateSnapshot
    error = Signal(str)

    def __init__(
        self,
        config: QtRendererConfig,
        command_queue: RenderCommandQueue,
    ) -> None:
        super().__init__()
        self.renderer = None
        self.ctx = None
        self._config = config
        self._commands = command_queue
        self._running = True
        # --online-training (change online-training-trigger): enable lazily once
        # the scene is built, then drive the per-frame tick on this render thread.
        self._online_training_requested = bool(config.online_training)
        self._online_training_enabled = False
        self._online_training_refused = False
        self._online_training_waiting = False
        # Mirror the user's intent onto the renderer for the configuration matrix
        # (change online-training-observability); the matrix's online-training row
        # carries the REFUSED/WAITING/APPROVED reason, so the worker no longer
        # prints its own one-shot refused/armed lines.
    def _build_renderer(self):
        cfg = self._config
        ctx = make_context(
            cfg.backend, window=None, width=cfg.width, height=cfg.height,
            gpu_preference=cfg.gpu_pref,
        )
        repo_root = Path(__file__).resolve().parents[3]
        neural_cfg = None
        if resolve_encoding(cfg.encoding) is not Encoding.E0:
            neural_cfg = NeuralBuildConfig(encoding=resolve_encoding(cfg.encoding))
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=Path(__file__).resolve().parents[1].parent / "shaders",
            hdr_dir=repo_root / "hdrs",
            tattoo_dir=repo_root / "tattoos",
            usd_scene_path=cfg.scene_path,
            use_usd_mtlx_plugin=cfg.use_usd_mtlx,
            execution_mode=cfg.execution_mode,
            bdpt_walk=cfg.bdpt_walk,
            neural_handoff=cfg.neural_handoff,
            neural_trainer=cfg.neural_trainer,
            train_precision=cfg.train_precision,
            neural_config=neural_cfg,
        )
        renderer._requested_backend = cfg.requested_backend
        renderer._online_training_requested = bool(cfg.online_training)
        if cfg.initial_integrator is not None:
            renderer.integrator_index = INTEGRATOR_INDEX[cfg.initial_integrator]
        if cfg.reuse is not None:
            renderer.reuse_index = renderer._REUSE_TOKENS.index(cfg.reuse)
        if cfg.lobe_samplers is not None:
            c, s, d = parse_lobe_samplers(cfg.lobe_samplers)
            renderer.coat_sampler_index = c
            renderer.spec_sampler_index = s
            renderer.diff_sampler_index = d
        apply_sppm_glossy_roughness(
            renderer,
            argparse.Namespace(sppm_glossy_roughness=cfg.sppm_glossy_roughness),
        )
        return ctx, renderer

    def _snapshot_state(self) -> RendererStateSnapshot:
        renderer = self.renderer
        if renderer is None or self.ctx is None:
            return RendererStateSnapshot(
                width=self._config.width,
                height=self._config.height,
                gpu_name="starting",
            )
        params = _snapshot_params(renderer, build_all_params(renderer))
        return RendererStateSnapshot(
            width=int(renderer.width),
            height=int(renderer.height),
            gpu_name=self.ctx.gpu_info.name,
            params=params,
            camera={},
            gizmo_mode=int(renderer.gizmo.mode),
            encoding=renderer._neural_config.encoding.value,
            sppm_glossy_roughness=getattr(
                renderer, "_sppm_glossy_roughness_override", None),
            online_training=renderer_online_status(renderer),
            choices=choice_names_from_renderer(renderer),
        )

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
            self.ctx, self.renderer = self._build_renderer()
            self.state_ready.emit(self._snapshot_state())
            while self._running:
                now = time.perf_counter()
                dt = now - prev
                prev = now

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

                self.frame_ready.emit(FrameSnapshot(
                    pixels=pixels,
                    width=w,
                    height=h,
                    accum_frame=accum,
                    gpu_name=self.ctx.gpu_info.name,
                    online_training=renderer_online_status(self.renderer),
                ))

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
            if self.renderer is not None:
                try:
                    self.renderer.disable_online_training()
                except Exception as exc:  # noqa: BLE001
                    self.error.emit(repr(exc))
                try:
                    self.renderer.cleanup()
                except Exception as exc:  # noqa: BLE001
                    self.error.emit(repr(exc))
            if self.ctx is not None:
                try:
                    self.ctx.destroy()
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

    def __init__(
        self,
        config: QtRendererConfig,
        proxy: QtRendererProxy,
        command_queue: RenderCommandQueue,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.renderer = proxy
        self._commands = command_queue
        self._image: QImage | None = None
        self._last_frame: FrameSnapshot | None = None
        # Hold the raw bytes so QImage's no-copy view stays valid until the
        # next frame replaces it.
        self._image_buffer: bytes | None = None

        self._camera = CameraDispatcher(proxy)

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
            config, self._commands,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.state_ready.connect(self._on_state_ready)
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

    def latest_frame(self) -> FrameSnapshot | None:
        return self._last_frame

    def _try_with_renderer(self, callback, default=None):
        """Best-effort immediate renderer read for hit-tests that need a result.

        These paths used to block the GUI while waiting for the render lock. If a
        frame is in flight, keep the UI responsive and let the gesture miss.
        """
        return default

    # ── Frame plumbing ─────────────────────────────────────────────

    def _on_state_ready(self, snapshot: RendererStateSnapshot) -> None:
        self.renderer.apply_snapshot(snapshot)

    def _on_frame_ready(self, frame: FrameSnapshot) -> None:
        # Hold both the bytes and the QImage view; QImage(bytes, …) is a
        # zero-copy view and the buffer must outlive every paint.
        self._last_frame = frame
        self._image_buffer = frame.pixels
        self._image = QImage(
            frame.pixels, frame.width, frame.height,
            4 * frame.width, QImage.Format_RGBA8888,
        )
        self.accum_changed.emit(frame.accum_frame)
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
        if shift:
            action = "autofocus"
        elif self._pick_armed:
            action = "pick"
        elif self._zoom_arming:
            action = "zoom"
        else:
            action = "camera"

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
        # action == "camera": non-modal left-press. Renderer access is async
        # under render-thread ownership (commit 382786f), so we can't decide
        # gizmo-vs-camera synchronously here — post the gizmo controller's
        # on_press to the render thread (mirroring on_move/on_release) and arm
        # the camera as the fallback. A successful ring grab sets
        # `_gizmo.is_dragging`, and the camera-drag command in mouseMoveEvent
        # early-outs on it, so the grab is never shadowed by camera rotation.
        if mapped is not None:
            self.post_render_command(
                lambda renderer, mapped=mapped: self._gizmo.on_press(
                    renderer, mapped,
                    shift=False, pick_armed=False, zoom_arming=False,
                ),
            )
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
            self.post_render_command(lambda renderer: self._gizmo.on_release(renderer))
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
        # Gizmo hover/drag is posted to the render thread; camera input stays
        # responsive even while a frame is in flight.
        mapped = self._widget_to_render_pixel(x, y)
        any_button = self._left or self._right or self._middle
        self.post_render_command(
            lambda renderer, mapped=mapped, any_button=any_button: self._gizmo.on_move(
                renderer, mapped,
                any_button_down=any_button, zoom_dragging=False,
            ),
            coalesce_key="gizmo-move",
        )

        if self._last_pos is None:
            self._last_pos = (x, y)
            return
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]
        self._last_pos = (x, y)
        if self._left or self._right or self._middle:
            def _camera_drag(renderer, dx=dx, dy=dy, left=self._left,
                             right=self._right, middle=self._middle) -> None:
                # A live gizmo grab (begun by on_press on this same thread)
                # consumes the left-drag; don't also rotate the camera.
                if left and self._gizmo.is_dragging:
                    return
                CameraDispatcher(renderer).drag(
                    dx, dy, left=left, right=right, middle=middle,
                )
            self.post_render_command(_camera_drag, coalesce_key="camera-drag")

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Qt wheel deltas are in eighths of a degree; one notch = 120.
        # GLFW yoff is 1/-1 per notch — match that.
        notches = event.angleDelta().y() / 120.0
        self.post_render_command(
            lambda renderer, notches=notches: CameraDispatcher(renderer).zoom(notches),
            coalesce_key="camera-zoom",
        )

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Q, Qt.Key_E):
            self._wasd[key] = True
            return
        if key == Qt.Key_C:
            self.post_render_command(lambda renderer: CameraDispatcher(renderer).toggle_mode())
        elif key == Qt.Key_F:
            self.post_render_command(lambda renderer: CameraDispatcher(renderer).reset())
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
                    CameraDispatcher(renderer).move(f, r, u, 0.016)
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
