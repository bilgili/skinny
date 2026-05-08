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
import io
import logging
import struct
import time
import uuid
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import ClassVar

import panel as pn
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler
from tornado.websocket import WebSocketHandler

from skinny.params import (
    ParamSpec, RESOLUTION_PRESETS, build_all_params,
    find_resolution_preset_index, _get_nested, _set_nested,
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


def _group_params(params: list[ParamSpec]) -> dict[str, list[ParamSpec]]:
    """Group params into UI sections."""
    groups: dict[str, list[ParamSpec]] = {
        "Render": [],
        "Skin": [],
        "Detail": [],
        "IBL": [],
        "Direct Light": [],
    }
    for p in params:
        path = p.path
        if path == "preset_index":
            groups["Skin"].append(p)
        elif path.startswith("mtlx."):
            groups["Skin"].append(p)
        elif path in ("env_index", "env_intensity"):
            groups["IBL"].append(p)
        elif path.startswith("light") or path == "direct_light_index":
            groups["Direct Light"].append(p)
        elif path in ("normal_map_strength", "displacement_scale_mm",
                       "detail_maps_index"):
            groups["Detail"].append(p)
        else:
            groups["Render"].append(p)
    return {k: v for k, v in groups.items() if v}


_CAPTURE_FORMATS = [
    ("PNG",  "png",  "png"),
    ("JPEG", "jpeg", "jpg"),
    ("BMP",  "bmp",  "bmp"),
    ("EXR",  "exr",  "exr"),
    ("HDR",  "hdr",  "hdr"),
]


def _build_resolution_section(session: "SkinnySession") -> pn.viewable.Viewable:
    """Resolution preset + W/H inputs that drive ``session.resize``."""
    cur_w = int(session.renderer.width)
    cur_h = int(session.renderer.height)
    preset_idx = find_resolution_preset_index(cur_w, cur_h)
    preset_names = [name for name, _w, _h in RESOLUTION_PRESETS]

    preset = pn.widgets.Select(
        name="Preset", options=preset_names,
        value=preset_names[preset_idx],
    )
    w_input = pn.widgets.IntInput(
        name="Width", value=cur_w, start=64, end=8192, step=8,
    )
    h_input = pn.widgets.IntInput(
        name="Height", value=cur_h, start=64, end=8192, step=8,
    )
    apply_btn = pn.widgets.Button(
        name="Apply Resolution", button_type="primary",
    )

    # _suppress prevents the preset → W/H sync writes from re-firing the
    # preset watcher (which would loop).
    state = {"suppress": False}

    def do_resize(w: int, h: int) -> None:
        try:
            actual_w, actual_h = session.resize(int(w), int(h))
        except Exception as exc:
            log.error("Resize failed: %s", exc)
            return
        state["suppress"] = True
        try:
            w_input.value = actual_w
            h_input.value = actual_h
            idx = find_resolution_preset_index(actual_w, actual_h)
            preset.value = RESOLUTION_PRESETS[idx][0]
        finally:
            state["suppress"] = False

    def on_preset(event):
        if state["suppress"]:
            return
        for name, w, h in RESOLUTION_PRESETS:
            if name == event.new:
                if w == 0 or h == 0:  # "Custom" — leave entries alone
                    return
                do_resize(w, h)
                return

    def on_apply(_event):
        do_resize(int(w_input.value), int(h_input.value))

    preset.param.watch(on_preset, "value")
    apply_btn.on_click(on_apply)

    return pn.Column(preset, w_input, h_input, apply_btn)


def _build_capture_section(session: "SkinnySession") -> pn.viewable.Viewable:
    """Format select + FileDownload that produces a screenshot on click."""
    fmt_select = pn.widgets.Select(
        name="Format",
        options=[label for label, _fmt, _ext in _CAPTURE_FORMATS],
        value="PNG",
    )

    def screenshot_callback():
        label = fmt_select.value
        for lab, fmt, _ext in _CAPTURE_FORMATS:
            if lab == label:
                data = session.screenshot(fmt)
                return io.BytesIO(data)
        return io.BytesIO(b"")

    download = pn.widgets.FileDownload(
        callback=screenshot_callback,
        filename="skinny.png",
        button_type="primary",
        label="Screenshot",
        embed=False,
    )

    def update_filename(*_):
        for lab, _fmt, ext in _CAPTURE_FORMATS:
            if lab == fmt_select.value:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                download.filename = f"skinny_{ts}.{ext}"
                return

    fmt_select.param.watch(update_filename, "value")
    update_filename()

    return pn.Column(fmt_select, download)


def _build_model_loader_section(
    session: "SkinnySession",
    model_select_widget: pn.widgets.Select | None = None,
) -> pn.viewable.Viewable:
    """FileSelector + Load button for loading models from the server filesystem."""
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    default_dir = str(repo_root / "assets")

    file_sel = pn.widgets.FileSelector(
        default_dir,
        file_pattern="*",
        only_files=True,
        name="Model files",
    )
    load_btn = pn.widgets.Button(name="Load Selected", button_type="primary")
    status = pn.pane.Alert("No model loaded", alert_type="info", visible=True)

    def _refresh_model_select():
        if model_select_widget is None:
            return
        models = session.renderer.models
        opts = {name: i for i, name in enumerate(models)} if models else {"(none)": -1}
        model_select_widget.options = opts
        idx = session.renderer.model_index
        if 0 <= idx < len(models):
            model_select_widget.value = idx

    def on_load(_event):
        selected = file_sel.value
        if not selected:
            status.object = "No file selected"
            status.alert_type = "warning"
            return
        path = _Path(selected[0])
        try:
            session.renderer.load_model_from_path(path)
            status.object = f"Loading: {path.name}"
            status.alert_type = "success"
            pn.state.execute(_refresh_model_select, schedule="later")
        except Exception as exc:
            status.object = f"Error: {exc}"
            status.alert_type = "danger"

    load_btn.on_click(on_load)
    return pn.Column(file_sel, load_btn, status)


def _build_sidebar_widgets(session: "SkinnySession") -> pn.Accordion:
    """Build the full sidebar accordion — requires session.ready."""
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
            opts = {label: i for i, label in enumerate(labels)}
            w = pn.widgets.Select(
                name=p.name,
                options=opts if opts else {"(none)": -1},
                value=current_idx if current_idx < len(labels) else (0 if labels else -1),
            )

            def on_select(event, path=p.path, source=p.choice_source):
                if event.new is None or event.new == -1:
                    return
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

    collapsed_groups = {"Skin"}
    sections = []
    active_indices = []

    sections.append(("Resolution", _build_resolution_section(session)))
    active_indices.append(len(sections) - 1)
    sections.append(("Capture", _build_capture_section(session)))
    active_indices.append(len(sections) - 1)

    for group_name, group_params in grouped.items():
        group_widgets = [make_widget(p) for p in group_params]
        sections.append((group_name, pn.Column(*group_widgets)))
        if group_name not in collapsed_groups:
            active_indices.append(len(sections) - 1)

    model_widget = widgets_by_path.get("model_index")
    sections.insert(0, ("Load Model", _build_model_loader_section(session, model_widget)))
    active_indices = [i + 1 for i in active_indices]
    active_indices.insert(0, 0)

    mat_col = pn.Column()
    sections.append(("Materials", mat_col))
    active_indices.append(len(sections) - 1)

    _last_scene_id = [id(None)]

    def _rebuild_material_col():
        usd_scene = getattr(session.renderer, "_usd_scene", None)
        cur_id = id(usd_scene)
        if cur_id == _last_scene_id[0]:
            return
        _last_scene_id[0] = cur_id
        mat_col.clear()
        if usd_scene is None or not usd_scene.materials:
            return
        mat_widgets = []
        for mat_id, mat in list(enumerate(usd_scene.materials))[1:]:
            mat_section_widgets = []
            for key, lo, hi in (
                ("roughness",      0.04, 1.0),
                ("metallic",       0.0,  1.0),
                ("specular",       0.0,  1.0),
                ("opacity",        0.0,  1.0),
                ("ior",            1.0,  3.0),
                ("coat",           0.0,  1.0),
                ("coat_roughness", 0.0,  1.0),
            ):
                cur = mat.parameter_overrides.get(key)
                try:
                    val = float(cur) if cur is not None else 0.5
                except (TypeError, ValueError):
                    val = 0.5
                val = max(lo, min(hi, val))
                w = pn.widgets.FloatSlider(
                    name=key, start=lo, end=hi, step=0.01, value=val,
                )

                def on_mat_change(event, mid=mat_id, k=key):
                    session.renderer.apply_material_override(mid, k, event.new)
                    if session.encoder.is_h264:
                        session.encoder.force_keyframe()

                w.param.watch(on_mat_change, "value")
                mat_section_widgets.append(w)

            diff = mat.parameter_overrides.get("diffuseColor")
            if diff is not None and "diffuseColor" not in mat.texture_paths:
                try:
                    r, g, b = float(diff[0]), float(diff[1]), float(diff[2])
                except (TypeError, IndexError, ValueError):
                    r, g, b = 0.8, 0.8, 0.8
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    max(0, min(255, int(round(r * 255)))),
                    max(0, min(255, int(round(g * 255)))),
                    max(0, min(255, int(round(b * 255)))),
                )
                cw = pn.widgets.ColorPicker(name="diffuseColor", value=hex_color)

                def on_color(event, mid=mat_id):
                    h = event.new.lstrip("#")
                    rf = int(h[0:2], 16) / 255.0
                    gf = int(h[2:4], 16) / 255.0
                    bf = int(h[4:6], 16) / 255.0
                    session.renderer.apply_material_override(
                        mid, "diffuseColor", (rf, gf, bf)
                    )
                    if session.encoder.is_h264:
                        session.encoder.force_keyframe()

                cw.param.watch(on_color, "value")
                mat_section_widgets.insert(0, cw)

            furnace_cb = pn.widgets.Checkbox(name="Furnace", value=False)

            def on_furnace(event, mid=mat_id):
                session.renderer.toggle_material_furnace(mid, event.new)
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            furnace_cb.param.watch(on_furnace, "value")
            mat_section_widgets.append(furnace_cb)

            mat_widgets.append(
                (mat.name, pn.Column(*mat_section_widgets))
            )

        if mat_widgets:
            mat_col.append(pn.Accordion(*mat_widgets, active=[]))

    _rebuild_material_col()

    session._material_rebuild_cb = _rebuild_material_col

    # ── Scene Graph section ──
    sg_col = pn.Column()
    sections.append(("Scene Graph", sg_col))

    _last_sg_id = [id(None)]

    def _rebuild_scene_graph_col():
        graph = session.renderer.scene_graph
        cur_id = id(graph)
        if cur_id == _last_sg_id[0]:
            return
        _last_sg_id[0] = cur_id
        sg_col.clear()
        if graph is None:
            sg_col.append(pn.pane.Markdown("*No scene loaded*"))
            return
        sg_col.append(_build_scene_graph_section(session, graph))

    _rebuild_scene_graph_col()
    session._scene_graph_rebuild_cb = _rebuild_scene_graph_col

    return pn.Accordion(*sections, active=active_indices)


def _build_scene_graph_section(
    session: "SkinnySession", graph,
) -> pn.viewable.Viewable:
    """Build the scene graph tree + property editor for the web sidebar."""
    from skinny.scene_graph import find_node_by_path, type_icon

    options = {}
    def _collect(node, depth):
        indent = "  " * depth
        icon = type_icon(node.type_name)
        label = f"{indent}{icon} {node.name}  ({node.type_name})"
        options[label] = node.path
        for child in node.children:
            _collect(child, depth + 1)
    _collect(graph, 0)

    selector = pn.widgets.Select(
        name="Node", options=options,
        size=min(20, max(8, len(options))),
    )
    props_col = pn.Column()

    def on_select(event):
        path = event.new
        if not path:
            return
        node = find_node_by_path(graph, path)
        if node is None:
            return
        _build_web_properties(props_col, session, node, graph)

    selector.param.watch(on_select, "value")
    return pn.Column(selector, props_col)




def _build_web_properties(
    props_col: "pn.Column",
    session: "SkinnySession",
    node,
    graph,
) -> None:
    """Rebuild the properties column with Panel widgets for the selected node."""
    from skinny.scene_graph import SceneGraphNode, find_node_by_path
    props_col.clear()

    props_col.append(pn.pane.Markdown(
        f"**{node.name}** `{node.type_name}`\n\n`{node.path}`"
    ))

    if not node.properties:
        props_col.append(pn.pane.Markdown("*No properties*"))
        return

    ref = node.renderer_ref
    # For shader nodes, try to find parent material ref
    if ref is None:
        ref = _find_web_ancestor_material_ref(node, graph)

    for prop in node.properties:
        if prop.type_name == "float" and prop.editable:
            lo = prop.metadata.get("min", 0.0)
            hi = prop.metadata.get("max", 1.0)
            w = pn.widgets.FloatSlider(
                name=prop.display_name, start=lo, end=hi,
                step=0.01, value=float(prop.value),
            )

            def on_change(event, r=ref, p=prop):
                if r is None:
                    return
                if r.kind == "material":
                    session.renderer.apply_material_override(r.index, p.name, event.new)
                elif r.kind in ("light_dir", "light_sphere"):
                    lt = "dir" if r.kind == "light_dir" else "sphere"
                    session.renderer.apply_light_override(lt, r.index, p.name, event.new)
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            w.param.watch(on_change, "value")
            props_col.append(w)

        elif prop.type_name == "color3f" and prop.editable:
            color = prop.value
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            hex_color = "#{:02x}{:02x}{:02x}".format(
                max(0, min(255, int(round(r * 255)))),
                max(0, min(255, int(round(g * 255)))),
                max(0, min(255, int(round(b * 255)))),
            )
            cw = pn.widgets.ColorPicker(name=prop.display_name, value=hex_color)

            def on_color(event, r=ref, p=prop):
                if r is None:
                    return
                h = event.new.lstrip("#")
                rf = int(h[0:2], 16) / 255.0
                gf = int(h[2:4], 16) / 255.0
                bf = int(h[4:6], 16) / 255.0
                if r.kind == "material":
                    session.renderer.apply_material_override(
                        r.index, p.name, (rf, gf, bf))
                elif r.kind in ("light_dir", "light_sphere"):
                    lt = "dir" if r.kind == "light_dir" else "sphere"
                    session.renderer.apply_light_override(
                        lt, r.index, p.name, (rf, gf, bf))
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            cw.param.watch(on_color, "value")
            props_col.append(cw)

        elif prop.type_name == "vec3f" and prop.editable and ref is not None and ref.kind == "instance":
            row_widgets = []
            for i, axis in enumerate(("X", "Y", "Z")):
                w = pn.widgets.FloatInput(
                    name=f"{prop.display_name} {axis}",
                    value=float(prop.value[i]),
                    step=0.1,
                )
                row_widgets.append(w)

            def on_vec3_change(event, r=ref, p=prop, ws=row_widgets):
                vals = tuple(float(w.value) for w in ws)
                # Collect all TRS from the properties
                t = s = (0.0, 0.0, 0.0)
                rot = (0.0, 0.0, 0.0)
                for pp in node.properties:
                    if pp.name == "translate":
                        t = vals if pp is p else pp.value
                    elif pp.name == "rotate":
                        rot = vals if pp is p else pp.value
                    elif pp.name == "scale":
                        s = vals if pp is p else pp.value
                session.renderer.apply_instance_transform(r.index, t, rot, s)
                if session.encoder.is_h264:
                    session.encoder.force_keyframe()

            for w in row_widgets:
                w.param.watch(on_vec3_change, "value")

            props_col.append(pn.Row(*row_widgets))

        elif prop.type_name == "vec3f":
            val = prop.value
            props_col.append(pn.pane.Markdown(
                f"**{prop.display_name}**: ({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f})"
            ))
        elif prop.type_name == "color3f":
            val = prop.value
            r, g, b = float(val[0]), float(val[1]), float(val[2])
            hex_c = "#{:02x}{:02x}{:02x}".format(
                max(0, min(255, int(round(r * 255)))),
                max(0, min(255, int(round(g * 255)))),
                max(0, min(255, int(round(b * 255)))),
            )
            props_col.append(pn.pane.HTML(
                f'<div style="font-size:12px;color:#ccc;padding:2px 0;">'
                f'<b>{prop.display_name}</b>: '
                f'<span style="display:inline-block;width:16px;height:16px;'
                f'background:{hex_c};border:1px solid #555;vertical-align:middle;'
                f'margin:0 4px;"></span>'
                f'({r:.2f}, {g:.2f}, {b:.2f})</div>'
            ))
        elif prop.type_name == "rel":
            props_col.append(pn.pane.Markdown(
                f"**{prop.display_name}**: &rarr; `{prop.value}`"
            ))
        elif prop.type_name == "asset":
            props_col.append(pn.pane.Markdown(
                f"**{prop.display_name}**: `{prop.value}`"
            ))
        else:
            val_str = f"{prop.value:.4f}" if isinstance(prop.value, float) else str(prop.value)
            props_col.append(pn.pane.Markdown(
                f"**{prop.display_name}**: {val_str}"
            ))


def _find_web_ancestor_material_ref(node, graph):
    """Walk up the tree to find an ancestor Material's RendererRef."""
    from skinny.scene_graph import find_node_by_path
    parts = node.path.rstrip("/").split("/")
    for i in range(len(parts) - 1, 0, -1):
        parent_path = "/".join(parts[:i]) or "/"
        parent = find_node_by_path(graph, parent_path)
        if parent is not None and parent.renderer_ref is not None:
            if parent.renderer_ref.kind == "material":
                return parent.renderer_ref
    return None


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

    layout = pn.Row(
        pn.Column(video_pane, info_bar, log_pane, sizing_mode="stretch_both"),
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
            sidebar_col.append(_build_sidebar_widgets(session))

            def _check_materials():
                cb = getattr(session, "_material_rebuild_cb", None)
                if cb is not None:
                    cb()
                sg_cb = getattr(session, "_scene_graph_rebuild_cb", None)
                if sg_cb is not None:
                    sg_cb()

            pn.state.add_periodic_callback(_check_materials, period=1000)

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
