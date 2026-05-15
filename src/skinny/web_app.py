"""Panel-based web application for Skinny.

Serves a browser UI built from the shared widget-tree spec
(``skinny.ui.build_app_ui.build_main_ui``) and streams rendered frames
via a custom Tornado WebSocket. Each browser session gets its own
Vulkan renderer and H264 encoder.

Usage::

    skinny-web [--port 8080] [--gpu auto] [--max-sessions 4] [scene.usda]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import struct
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import ClassVar

import panel as pn
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

from skinny.params import _set_nested
from skinny.vk_context import VulkanContext
from skinny.renderer import Renderer
from skinny.video_encoder import VideoEncoder

log = logging.getLogger(__name__)

# ── Module-level config (set by main) ────────────────────────────────

_GPU_PREFERENCE: str = "auto"
_USD_PATH: Path | None = None
_USE_USD_MTLX: bool = False


# ── Session management ───────────────────────────────────────────────

class SkinnySession:
    """Per-user renderer session."""

    MAX_SESSIONS: ClassVar[int] = 4
    _active: ClassVar[dict[str, "SkinnySession"]] = {}

    def __init__(self, session_id: str) -> None:
        if len(self._active) >= self.MAX_SESSIONS:
            raise RuntimeError(
                f"Max sessions ({self.MAX_SESSIONS}) reached. Try again later."
            )

        self.session_id = session_id
        self._running = False
        self.frame_queue: Queue[tuple[int, bytes]] = Queue(maxsize=2)
        self.accum_frame = 0
        self._lock = Lock()
        self._handlers: list["VideoStreamHandler"] = []
        self.ready = False
        self._init_error: Exception | None = None
        self._init_log: list[str] = []
        self.ctx: VulkanContext | None = None
        self.renderer: Renderer | None = None
        self.encoder: VideoEncoder | None = None
        self._active[session_id] = self

    def _log_init(self, msg: str) -> None:
        self._init_log.append(msg)
        log.info("Session %s: %s", self.session_id, msg)

    def initialize(self) -> None:
        """Heavy initialization — run from a background thread."""
        try:
            self._log_init("Creating Vulkan context...")
            self.ctx = VulkanContext(
                window=None, width=1280, height=720,
                gpu_preference=_GPU_PREFERENCE,
            )
            self._log_init(f"GPU: {self.ctx.gpu_info.name}")

            self._log_init("Initializing renderer (shaders, meshes, materials)...")
            repo_root = Path(__file__).resolve().parents[2]
            self.renderer = Renderer(
                vk_ctx=self.ctx,
                shader_dir=Path(__file__).parent / "shaders",
                hdr_dir=repo_root / "hdrs",
                tattoo_dir=repo_root / "tattoos",
                usd_scene_path=_USD_PATH,
                use_usd_mtlx_plugin=_USE_USD_MTLX,
            )

            self._log_init("Setting up video encoder...")
            self.encoder = VideoEncoder(1280, 720, gpu_info=self.ctx.gpu_info)
            self._log_init(f"Encoder: {self.encoder.encoder_name}")

            self._render_thread = Thread(target=self._render_loop, daemon=True)
            self._running = True
            self._render_thread.start()

            self._log_init("Renderer ready")
            self.ready = True
        except Exception as e:
            self._log_init(f"Initialization failed: {e}")
            self._init_error = e
            log.error("Session %s init failed: %s", self.session_id, e,
                       exc_info=True)

    def _render_loop(self) -> None:
        prev = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            with self._lock:
                self.renderer.update(dt)
                raw = self.renderer.render_headless()
                self.accum_frame = self.renderer.accum_frame
                width = self.renderer.width
                height = self.renderer.height
                is_h264 = self.encoder.is_h264
                has_desc = bool(self.encoder.avcc_description)

                if is_h264 and has_desc:
                    results = self.encoder.encode_h264(raw)
                else:
                    quality = 92 if self.accum_frame > 30 else 75
                    results = None
                    jpeg = self.encoder.encode_jpeg(raw, quality=quality)

            if results is not None:
                for is_key, avcc_data in results:
                    self._push_frame(0 if is_key else 1, avcc_data)
            else:
                self._push_frame(2, jpeg)
            # Suppresses an unused-variable warning when not in JPEG path.
            del width, height

            # Adaptive rate: full speed during interaction, throttle when converging
            af = self.accum_frame
            if af > 200:
                time.sleep(0.5)
            elif af > 50:
                time.sleep(0.1)
            elif af > 10:
                time.sleep(0.03)

    def _push_frame(self, frame_type: int, data: bytes) -> None:
        header = struct.pack("!BI", frame_type, self.accum_frame)
        frame = header + data
        try:
            self.frame_queue.put_nowait(frame)
        except Full:
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                pass

    def set_param(self, path: str, value) -> None:
        _set_nested(self.renderer, path, value)

    def handle_camera(self, action: str, data: dict) -> None:
        cam = self.renderer.camera
        if action == "orbit":
            cam.orbit(data.get("dx", 0), data.get("dy", 0))
        elif action == "pan":
            cam.pan(data.get("dx", 0), data.get("dy", 0))
        elif action == "zoom":
            cam.zoom(data.get("delta", 0))
        elif action == "move" and hasattr(cam, "move"):
            cam.move(
                float(data.get("forward", 0)),
                float(data.get("right", 0)),
                float(data.get("up", 0)),
                float(data.get("dt", 0.016)),
            )
        if self.encoder.is_h264:
            self.encoder.force_keyframe()

    # ── Resize + screenshot (called from UI / WS threads) ──────────────

    def resize(self, width: int, height: int) -> tuple[int, int]:
        """Change render + encoder resolution. Returns the actual (W, H)
        the renderer settled on (clamped + rounded to workgroup multiple).
        """
        from skinny.video_encoder import VideoEncoder

        with self._lock:
            self.renderer.resize(width, height)
            actual_w = int(self.renderer.width)
            actual_h = int(self.renderer.height)
            if (actual_w, actual_h) != (self.encoder.width, self.encoder.height):
                self.encoder.close()
                self.encoder = VideoEncoder(
                    actual_w, actual_h, gpu_info=self.ctx.gpu_info,
                )
            self.encoder.force_keyframe()
            # Drop any frames queued at the previous resolution so the
            # browser doesn't try to decode old-dim H264 packets with
            # the new decoder configuration.
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    break

            # Schedule the type=4 (resize) and type=3 (codec config)
            # WebSocket writes BEFORE the render thread is unblocked so
            # the IOLoop sees them ahead of any new H264 frames.
            desc = self.encoder.avcc_description
            for handler in list(self._handlers):
                handler.send_resize(actual_w, actual_h)
                if desc:
                    handler.send_codec_config(desc)
        return actual_w, actual_h

    def screenshot(self, fmt: str) -> bytes:
        """Render a screenshot in the requested format and return its bytes."""
        buf = io.BytesIO()
        with self._lock:
            self.renderer.save_screenshot(buf, fmt)
        return buf.getvalue()

    def register_handler(self, handler: "VideoStreamHandler") -> None:
        self._handlers.append(handler)

    def unregister_handler(self, handler: "VideoStreamHandler") -> None:
        try:
            self._handlers.remove(handler)
        except ValueError:
            pass

    def cleanup(self) -> None:
        log.info("Cleaning up session %s", self.session_id)
        self._running = False
        if hasattr(self, "_render_thread"):
            self._render_thread.join(timeout=5)
        if self.encoder is not None:
            self.encoder.close()
        if self.renderer is not None:
            self.renderer.cleanup()
        if self.ctx is not None:
            self.ctx.destroy()
        self._active.pop(self.session_id, None)

    @classmethod
    def get(cls, session_id: str) -> "SkinnySession | None":
        return cls._active.get(session_id)


# ── Tornado WebSocket for video frames ───────────────────────────────

class VideoStreamHandler(WebSocketHandler):
    """Streams encoded video frames to the browser over a binary WebSocket."""

    def check_origin(self, origin):
        return True

    def open(self, session_id: str):
        self.session = SkinnySession.get(session_id)
        if self.session is None:
            log.warning("Video WS rejected: unknown session %s", session_id)
            self.close(1008, "Unknown session")
            return
        self._streaming = True
        self._ioloop = IOLoop.current()
        self.session.register_handler(self)
        IOLoop.current().spawn_callback(self._stream_frames)
        log.info("Video WS opened for session %s", session_id)

    def send_resize(self, width: int, height: int) -> None:
        """Push a type=4 resize message to the client. Type byte + accum
        placeholder + (uint16 width, uint16 height)."""
        payload = struct.pack("!BIHH", 4, 0, width, height)
        self._write_safely(payload)

    def send_codec_config(self, desc: bytes) -> None:
        header = struct.pack("!BI", 3, 0)
        self._write_safely(header + desc)

    def _write_safely(self, data: bytes) -> None:
        # write_message must run on the IOLoop thread; resize/screenshot
        # run on the Bokeh worker thread, so schedule the write.
        loop = getattr(self, "_ioloop", None)
        if loop is None:
            return
        try:
            loop.add_callback(lambda: self._do_write(data))
        except Exception:
            pass

    def _do_write(self, data: bytes) -> None:
        if self.ws_connection is None:
            return
        try:
            self.write_message(data, binary=True)
        except Exception:
            pass

    async def _stream_frames(self):
        loop = asyncio.get_event_loop()
        while self._streaming and self.ws_connection and not self.session.ready:
            if self.session._init_error:
                return
            await asyncio.sleep(0.3)
        if not self._streaming or not self.ws_connection:
            return
        while not self.session.frame_queue.empty():
            try:
                self.session.frame_queue.get_nowait()
            except Empty:
                break
        self.session.encoder.force_keyframe()
        self.send_resize(self.session.renderer.width, self.session.renderer.height)
        desc = self.session.encoder.avcc_description
        if desc:
            self.send_codec_config(desc)
            log.info("Video WS: sent AVCC description (%d bytes)", len(desc))
        while self._streaming and self.ws_connection:
            try:
                frame = await loop.run_in_executor(
                    None, self.session.frame_queue.get, True, 0.1
                )
            except Empty:
                continue
            try:
                self.write_message(frame, binary=True)
            except Exception:
                break

    def on_message(self, message):
        import json
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        if self.session is None:
            return
        msg_type = data.get("type")
        if msg_type == "camera":
            self.session.handle_camera(data.get("action", ""), data)

    def on_close(self):
        self._streaming = False
        if self.session is not None:
            self.session.unregister_handler(self)
        log.info("Video WS closed")


# ── Standalone video page served via Tornado ────────────────────────

_TEMPLATE_PATH = Path(__file__).parent / "web_templates" / "video_player.html"


class VideoPageHandler(RequestHandler):
    """Serves the video player as a standalone HTML page (loaded via iframe)."""

    def get(self, session_id: str):
        template = _TEMPLATE_PATH.read_text()
        page = (
            template
            .replace("{{SESSION_ID}}", session_id)
            .replace("{{WIDTH}}", "1280")
            .replace("{{HEIGHT}}", "720")
        )
        self.set_header("Content-Type", "text/html")
        self.write(page)


# ── Panel app builder ────────────────────────────────────────────────


def _build_sidebar_widgets(
    session: "SkinnySession",
    child_windows_col: pn.Column,
) -> pn.viewable.Viewable:
    """Build the Panel sidebar from the shared widget-tree spec.

    All layout decisions live in :func:`skinny.ui.build_app_ui.build_main_ui`;
    this function only injects session-scoped callbacks (screenshot/load
    under the session lock, child-window openers that push cards into
    ``child_windows_col``).
    """
    from skinny.ui.build_app_ui import AppCallbacks, build_main_ui
    from skinny.ui.panel.backend import PanelTreeBuilder
    from skinny.ui.panel.windows import (
        build_bxdf_pane, build_debug_viewport_pane,
        build_material_graph_pane, build_scene_graph_pane,
    )

    def _capture(fmt: str) -> bytes:
        return session.screenshot(fmt)

    def _load_model(path):
        with session._lock:
            session.renderer.load_model_from_path(path)

    # Track open panes by name so a second sidebar click brings the
    # existing pane into view rather than spawning a duplicate.
    open_panes: dict[str, pn.Card] = {}

    def _toggle(name: str, builder):
        def _open():
            if name in open_panes:
                # Already open — uncollapse (in case the user collapsed it).
                open_panes[name].collapsed = False
                return
            def _on_close():
                card = open_panes.pop(name, None)
                if card is not None and card in child_windows_col:
                    child_windows_col.remove(card)
            card = builder(session, _on_close)
            open_panes[name] = card
            child_windows_col.append(card)
        return _open

    def _debug_view(which: str):
        def _go():
            # Ensure the debug pane is open, then call its view button.
            _toggle("debug", build_debug_viewport_pane)()
            card = open_panes.get("debug")
            if card is None:
                return
            # Card layout: [close_button, Column(button_row, image, status)].
            # First child of inner Column is the button Row containing the
            # view buttons in fixed order: Top, Left, Back, Reset.
            try:
                btn_row = card.objects[1].objects[0]
                idx = {"top": 0, "left": 1, "back": 2}.get(which)
                if idx is not None:
                    btn_row.objects[idx].clicks += 1
            except (IndexError, AttributeError):
                pass
        return _go

    callbacks = AppCallbacks(
        open_scene_graph=_toggle("scene", build_scene_graph_pane),
        open_material_graph=_toggle("material_graph", build_material_graph_pane),
        open_bxdf_visualizer=_toggle("bxdf", build_bxdf_pane),
        open_debug_viewport=_toggle("debug", build_debug_viewport_pane),
        debug_view_top=_debug_view("top"),
        debug_view_left=_debug_view("left"),
        debug_view_back=_debug_view("back"),
        capture_screenshot=_capture,
        load_model=_load_model,
    )

    tree = build_main_ui(session.renderer, callbacks=callbacks)
    builder = PanelTreeBuilder(tree)
    builder.register_periodic()
    session._panel_builder = builder
    return builder.layout


def create_panel_app() -> pn.viewable.Viewable:
    """Called per browser session by Panel serve."""
    session_id = str(uuid.uuid4())[:8]

    try:
        session = SkinnySession(session_id)
    except RuntimeError as e:
        return pn.pane.Alert(str(e), alert_type="danger")

    _doc = pn.state.curdoc

    def _init_and_signal():
        log.info("Init thread started for session %s", session_id)
        session.initialize()
        log.info("Init thread finished for session %s (ready=%s, error=%s)",
                 session_id, session.ready, session._init_error)

    Thread(
        target=_init_and_signal, daemon=True, name=f"init-{session_id}",
    ).start()

    iframe_html = (
        f'<iframe src="/video_page/{session_id}" '
        f'style="width:100%;height:100%;border:none;display:block;background:#000;" '
        f'allow="autoplay"></iframe>'
    )
    video_pane = pn.pane.HTML(iframe_html, sizing_mode="stretch_both", min_height=500)

    info_bar = pn.pane.Markdown(
        f"**Session:** {session_id} | Initializing...",
        styles={"font-size": "11px", "color": "#888"},
    )

    log_pane = pn.pane.HTML(
        '<pre style="margin:0;padding:4px 8px;font-size:11px;color:#aaa;'
        'background:#1a1a1a;max-height:120px;overflow-y:auto;'
        'white-space:pre-wrap;">Initializing renderer...</pre>',
        sizing_mode="stretch_width",
    )

    sidebar_col = pn.Column(
        pn.indicators.LoadingSpinner(value=True, size=25),
        pn.pane.Markdown("*Initializing renderer...*"),
        width=340, scroll=True,
    )
    # Hosts child-window cards (scene graph, BXDF, material graph, debug
    # viewport). Empty until the user clicks one of the sidebar buttons.
    child_windows_col = pn.Column(width=520, scroll=True)

    layout = pn.Row(
        pn.Column(video_pane, info_bar, log_pane, sizing_mode="stretch_both"),
        child_windows_col,
        sidebar_col,
        sizing_mode="stretch_both",
    )

    _poll_active = [True]
    _prev_log_len = [0]

    def _update_ui():
        if not _poll_active[0]:
            return
        msgs = list(session._init_log)
        if len(msgs) > _prev_log_len[0]:
            _prev_log_len[0] = len(msgs)
            text = "\n".join(msgs)
            log_pane.object = (
                '<pre style="margin:0;padding:4px 8px;font-size:11px;color:#aaa;'
                'background:#1a1a1a;max-height:120px;overflow-y:auto;'
                f'white-space:pre-wrap;">{text}</pre>'
            )

        if session.ready:
            _poll_active[0] = False
            sidebar_col.clear()
            sidebar_col.append(_build_sidebar_widgets(session, child_windows_col))

            encoder_info = session.encoder.encoder_name
            hw_tag = " (HW)" if session.encoder.is_hardware else " (SW)"
            info_bar.object = (
                f"**Session:** {session_id} | **GPU:** {session.ctx.gpu_info.name}"
                f" | **Encoder:** {encoder_info}{hw_tag}"
            )
        elif session._init_error:
            _poll_active[0] = False
            sidebar_col.clear()
            sidebar_col.append(
                pn.pane.Alert(str(session._init_error), alert_type="danger"),
            )
            info_bar.object = (
                f"**Session:** {session_id} | **Error:** {session._init_error}"
            )

    def _poll_loop():
        while _poll_active[0]:
            time.sleep(0.5)
            if _doc:
                try:
                    _doc.add_next_tick_callback(_update_ui)
                except Exception:
                    break

    Thread(target=_poll_loop, daemon=True, name=f"poll-{session_id}").start()

    def on_session_destroyed(session_context):
        _poll_active[0] = False
        session.cleanup()

    pn.state.on_session_destroyed(on_session_destroyed)

    return layout


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="skinny-web")
    parser.add_argument(
        "scene", nargs="?", type=Path, default=None,
        help="Path to a USD stage (.usda/.usdc/.usdz).",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--gpu", type=str, default="auto",
                        help="GPU preference: intel, nvidia, amd, discrete, auto")
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--usd", type=Path, default=None,
                        help="Path to a USD stage (alternative to positional scene arg).")
    parser.add_argument("--usdMtlx", action="store_true", default=False)
    args = parser.parse_args()

    global _GPU_PREFERENCE, _USD_PATH, _USE_USD_MTLX
    _GPU_PREFERENCE = args.gpu
    _USD_PATH = args.scene or args.usd
    _USE_USD_MTLX = args.usdMtlx
    SkinnySession.MAX_SESSIONS = args.max_sessions

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    def modify_doc(doc):
        pass

    extra_patterns = [
        (r"/video_ws/(\w+)", VideoStreamHandler),
        (r"/video_page/(\w+)", VideoPageHandler),
    ]

    server = pn.serve(
        {"skinny": create_panel_app},
        port=args.port,
        address="0.0.0.0",
        allow_websocket_origin=["*"],
        show=False,
        start=False,
    )

    from tornado.routing import Rule, PathMatches
    for pattern, handler in extra_patterns:
        server._tornado.wildcard_router.rules.insert(
            0, Rule(PathMatches(pattern), handler)
        )

    log.info("Skinny web server starting on port %d (GPU: %s)", args.port, args.gpu)
    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    main()
