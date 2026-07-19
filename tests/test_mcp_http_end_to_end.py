"""End-to-end over real HTTP: guards, then a tool call through the transport.

Exercises the ASGI stack the client actually talks to, against a fake renderer
driven by a real command queue. No GPU.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from skinny import mcp_server
from skinny.mcp_auth import bind_loopback_socket
from skinny.render_session import RenderCommandQueue
from skinny.scene_graph import RendererRef, SceneGraphNode, SceneGraphProperty

TOKEN = "end-to-end-token"


class FakeRenderer:
    def __init__(self) -> None:
        shader = SceneGraphNode(
            path="/World/mat/surface", name="surface", type_name="Shader",
            children=[], properties=[
                SceneGraphProperty(
                    name="roughness", display_name="roughness", type_name="float",
                    value=0.5, editable=True, metadata={"min": 0.04, "max": 1.0},
                ),
            ], renderer_ref=None,
        )
        material = SceneGraphNode(
            path="/World/mat", name="mat", type_name="Material", children=[shader],
            properties=[], renderer_ref=RendererRef(kind="material", index=0),
        )
        self.scene_graph = SceneGraphNode(
            path="/", name="/", type_name="Stage",
            children=[SceneGraphNode(
                path="/World", name="World", type_name="Xform",
                children=[material], properties=[], renderer_ref=None,
            )],
            properties=[], renderer_ref=None,
        )
        self._material_version = 1
        self._scene_graph_version = 1
        self._usd_stage = None
        self.calls: list = []

    def apply_material_override(self, index, key, value):
        self.calls.append((index, key, value))
        self._material_version += 1


@pytest.fixture
def server():
    renderer = FakeRenderer()
    queue = RenderCommandQueue()
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            queue.run_pending(renderer)
            time.sleep(0.001)

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()

    sock = bind_loopback_socket(0)
    port = sock.getsockname()[1]
    app = mcp_server.build_app(mcp_server.SceneTools(queue, timeout=5.0), TOKEN, port)

    import uvicorn

    config = uvicorn.Config(app, log_level="critical")
    uv = uvicorn.Server(config)
    uv.install_signal_handlers = lambda: None
    threading.Thread(target=lambda: uv.run(sockets=[sock]), daemon=True).start()
    time.sleep(0.8)

    yield port, renderer

    uv.should_exit = True
    stop.set()
    worker.join(timeout=2.0)
    time.sleep(0.3)  # let uvicorn close the socket it now owns


def _post(port, *, token=None, origin=None, host=None, body=b"{}"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    if origin is not None:
        req.add_header("Origin", origin)
    if host is not None:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_unauthenticated_request_refused_over_http(server) -> None:
    port, _renderer = server
    status, body = _post(port)
    assert status == 403
    assert b"token" in body


def test_wrong_token_refused_over_http(server) -> None:
    port, _renderer = server
    status, _body = _post(port, token="wrong")
    assert status == 403


def test_browser_origin_refused_over_http(server) -> None:
    """A page in the operator's browser must not reach a tool."""
    port, _renderer = server
    status, body = _post(port, token=TOKEN, origin="https://evil.example")
    assert status == 403
    assert b"Origin" in body


def test_rebound_host_refused_over_http(server) -> None:
    port, _renderer = server
    status, body = _post(port, token=TOKEN, host="evil.example")
    assert status == 403
    assert b"Host" in body


def test_authenticated_initialize_reaches_the_server(server) -> None:
    """A valid request gets past the guard into the MCP transport itself."""
    port, _renderer = server
    body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }).encode()
    status, payload = _post(port, token=TOKEN, body=body)

    assert status == 200, payload
    assert b"skinny" in payload  # server identifies itself
