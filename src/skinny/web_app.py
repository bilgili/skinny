"""Panel-based web application for Skinny.

Serves a browser UI that replicates the Tkinter control panel and streams
rendered frames via a custom Tornado WebSocket. Each browser session gets
its own Vulkan renderer and H264 encoder.

Usage::

    skinny-web [--port 8080] [--gpu auto] [--max-sessions 4] [scene.usda]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import time
import uuid
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from typing import ClassVar

import panel as pn
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

from skinny.params import (
    ParamSpec, build_all_params, _get_nested, _set_nested,
)
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

        usd_scene = None
        if _USD_PATH is not None:
            from skinny.usd_loader import load_scene_from_usd
            usd_scene = load_scene_from_usd(_USD_PATH, use_usd_mtlx_plugin=_USE_USD_MTLX)

        self.ctx = VulkanContext(
            window=None, width=1280, height=720,
            gpu_preference=_GPU_PREFERENCE,
        )
        repo_root = Path(__file__).resolve().parents[2]
        self.renderer = Renderer(
            vk_ctx=self.ctx,
            shader_dir=Path(__file__).parent / "shaders",
            hdr_dir=repo_root / "hdrs",
            head_dir=repo_root / "heads",
            tattoo_dir=repo_root / "tattoos",
            usd_scene=usd_scene,
        )
        self.encoder = VideoEncoder(1280, 720, gpu_info=self.ctx.gpu_info)
        log.info("Session %s: encoder=%s", session_id, self.encoder.encoder_name)

        self._render_thread = Thread(target=self._render_loop, daemon=True)
        self._running = True
        self._render_thread.start()
        self._active[session_id] = self

    def _render_loop(self) -> None:
        prev = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = now - prev
            prev = now

            self.renderer.update(dt)
            raw = self.renderer.render_headless()
            self.accum_frame = self.renderer.accum_frame

            if self.encoder.is_h264 and self.encoder.avcc_description:
                results = self.encoder.encode_h264(raw)
                for is_key, avcc_data in results:
                    self._push_frame(0 if is_key else 1, avcc_data)
            else:
                quality = 92 if self.accum_frame > 30 else 75
                jpeg = self.encoder.encode_jpeg(raw, quality=quality)
                self._push_frame(2, jpeg)

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
        elif action == "move":
            cam.move(
                float(data.get("forward", 0)),
                float(data.get("right", 0)),
                float(data.get("up", 0)),
                float(data.get("dt", 0.016)),
            )

    def cleanup(self) -> None:
        log.info("Cleaning up session %s", self.session_id)
        self._running = False
        self._render_thread.join(timeout=5)
        self.encoder.close()
        self.renderer.cleanup()
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
        # Drain stale frames so browser gets a fresh keyframe first
        while not self.session.frame_queue.empty():
            try:
                self.session.frame_queue.get_nowait()
            except Empty:
                break
        self.session.encoder.force_keyframe()
        desc = self.session.encoder.avcc_description
        if desc:
            header = struct.pack("!BI", 3, 0)
            self.write_message(header + desc, binary=True)
            log.info("Video WS: sent AVCC description (%d bytes)", len(desc))
        IOLoop.current().spawn_callback(self._stream_frames)
        log.info("Video WS opened for session %s", session_id)

    async def _stream_frames(self):
        loop = asyncio.get_event_loop()
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


def _group_params(params: list[ParamSpec]) -> dict[str, list[ParamSpec]]:
    """Group params into UI sections."""
    groups: dict[str, list[ParamSpec]] = {
        "Render": [],
        "Skin": [],
        "Detail": [],
        "Light": [],
    }
    for p in params:
        path = p.path
        if path.startswith("mtlx."):
            groups["Skin"].append(p)
        elif path.startswith("light") or "color" in path:
            groups["Light"].append(p)
        elif path in ("normal_map_strength", "displacement_scale_mm",
                       "detail_maps_index", "subdivision_index"):
            groups["Detail"].append(p)
        else:
            groups["Render"].append(p)
    return {k: v for k, v in groups.items() if v}


def create_panel_app() -> pn.viewable.Viewable:
    """Called per browser session by Panel serve."""
    session_id = str(uuid.uuid4())[:8]

    try:
        session = SkinnySession(session_id)
    except RuntimeError as e:
        return pn.pane.Alert(str(e), alert_type="danger")

    params = build_all_params(session.renderer)
    grouped = _group_params(params)

    widgets_by_path: dict[str, pn.widgets.Widget] = {}

    def make_widget(p: ParamSpec) -> pn.widgets.Widget:
        if p.kind == "continuous":
            w = pn.widgets.FloatSlider(
                name=p.name, start=p.lo, end=p.hi, step=p.step,
                value=float(_get_nested(session.renderer, p.path)),
            )

            def on_change(event, path=p.path):
                session.set_param(path, event.new)
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            w.param.watch(on_change, "value")
        else:
            choices = getattr(session.renderer, p.choice_source, [])
            labels = []
            for i, c in enumerate(choices):
                name = getattr(c, "name", str(c))
                labels.append(name)
            current_idx = int(_get_nested(session.renderer, p.path))
            w = pn.widgets.Select(
                name=p.name,
                options={label: i for i, label in enumerate(labels)},
                value=current_idx if current_idx < len(labels) else 0,
            )

            def on_select(event, path=p.path, source=p.choice_source):
                session.set_param(path, event.new)
                if source == "presets":
                    from skinny.presets import apply_preset
                    presets = getattr(session.renderer, "presets", [])
                    if 0 <= event.new < len(presets):
                        apply_preset(session.renderer, presets[event.new])
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            w.param.watch(on_select, "value")
        widgets_by_path[p.path] = w
        return w

    sections = []
    for group_name, group_params in grouped.items():
        group_widgets = [make_widget(p) for p in group_params]
        sections.append((group_name, pn.Column(*group_widgets)))

    sidebar = pn.Accordion(*sections, active=list(range(len(sections))))

    iframe_html = (
        f'<iframe src="/video_page/{session_id}" '
        f'style="width:100%;aspect-ratio:16/9;border:none;display:block;" '
        f'allow="autoplay"></iframe>'
    )
    video_pane = pn.pane.HTML(iframe_html, sizing_mode="stretch_both", min_height=500)

    encoder_info = session.encoder.encoder_name
    hw_tag = " (HW)" if session.encoder.is_hardware else " (SW)"
    info_bar = pn.pane.Markdown(
        f"**Session:** {session_id} | **GPU:** {session.ctx.gpu_info.name} | "
        f"**Encoder:** {encoder_info}{hw_tag}",
        styles={"font-size": "11px", "color": "#888"},
    )

    layout = pn.Row(
        pn.Column(video_pane, info_bar, sizing_mode="stretch_both"),
        pn.Column(sidebar, width=340, scroll=True),
        sizing_mode="stretch_both",
    )

    def on_session_destroyed(session_context):
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
    parser.add_argument("--usdMtlx", action="store_true", default=False)
    args = parser.parse_args()

    global _GPU_PREFERENCE, _USD_PATH, _USE_USD_MTLX
    _GPU_PREFERENCE = args.gpu
    _USD_PATH = args.scene
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
