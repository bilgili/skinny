"""In-process MCP server exposing the live scene graph.

Three path-addressed tools — ``scene_list`` / ``scene_get`` / ``scene_set`` —
over the scene-graph model the Qt dock already edits. Attaches to a renderer
that is already running; it never builds one.

**Threading invariant (design D2):** this module never touches ``Renderer``
directly. ``Renderer`` has no internal lock, and its scene graph is rebuilt and
swapped by the streaming load thread, so *reads* race too. Every read and every
write goes through the render-thread command queue and is awaited with a
timeout.

Writes await deliberately. Resolving the node, validating the value, and
dispatching all have to happen on the render thread, and a client needs to know
whether its edit was applied or rejected — a fire-and-forget write would report
success for an edit that a bounds, type, or routing check then threw away. The
cost is that writes cannot coalesce (``post_with_reply`` takes no
``coalesce_key``), so a client value-sweep is paced by the round-trip rather
than collapsed into a single edit. That is the right trade for a caller that
must be told what happened; the operator's own slider drags still coalesce
through the dock's proxy verbs.

Review rule: no ``renderer.`` outside a posted closure.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import threading
from copy import deepcopy
from dataclasses import replace
from typing import Any

from skinny.mcp_auth import check_request, load_or_create_token, registration_command
from skinny.scene_graph import find_node_by_path, scene_graph_to_dict
from skinny.ui.scene_edit_actions import apply_scene_property

log = logging.getLogger(__name__)

# A read that cannot complete within this many seconds means the render thread
# is wedged; report it rather than blocking the client forever.
REQUEST_TIMEOUT_S = 10.0

DEFAULT_LIST_DEPTH = 2

# Property types whose displayed value the renderer owns and republishes when it
# rebuilds the node (the camera's lens_file shows a "(load .usda)" placeholder,
# not the loaded path). Writing our value onto the property object would be
# silently discarded, so a later read would contradict what the write reported.
_RENDERER_OWNED_VALUES = frozenset({"lens_file"})


class SceneToolError(Exception):
    """A tool-level failure to report to the client, not a transport error."""


# ── Read helpers (all run on the render thread) ──────────────────────

def _versions(renderer) -> dict[str, int]:
    """Both counters: property edits move only ``_material_version``.

    ``_scene_graph_version`` is bumped on *structural* change only — the
    renderer deliberately leaves it alone on a property edit because the dock's
    widgets are bound to the live property objects. Reporting only that one
    would leave a client unable to observe any material or light edit.
    """
    return {
        "scene_graph_version": int(getattr(renderer, "_scene_graph_version", 0)),
        "material_version": int(getattr(renderer, "_material_version", 0)),
    }


def _summarize(node, depth: int, kind: str | None) -> dict:
    """Structure only — no properties. See design D4."""
    entry: dict[str, Any] = {
        "path": node.path,
        "name": node.name,
        "type": node.type_name,
        "child_count": len(node.children),
    }
    if node.renderer_ref is not None:
        entry["kind"] = node.renderer_ref.kind
    if depth > 0 and node.children:
        children = [_summarize(c, depth - 1, kind) for c in node.children]
        if kind is not None:
            children = [c for c in children if c]
        if children:
            entry["children"] = children
    return entry


def _matches_kind(node, kind: str | None) -> bool:
    if kind is None:
        return True
    ref = node.renderer_ref
    return ref is not None and ref.kind == kind


def _collect_by_kind(node, kind: str, out: list, depth: int) -> None:
    if _matches_kind(node, kind):
        out.append({
            "path": node.path,
            "name": node.name,
            "type": node.type_name,
            "kind": node.renderer_ref.kind,
            "child_count": len(node.children),
        })
    if depth <= 0:
        return
    for child in node.children:
        _collect_by_kind(child, kind, out, depth - 1)


class _QueueProxy:
    """Adapter giving a bare command queue the ``request`` surface.

    The Qt front-end already owns a ``QtRendererProxy``; the GLFW front-end has
    only a queue. Tools use nothing but ``request``, so either works.
    """

    def __init__(self, queue) -> None:
        self._commands = queue

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _as_proxy(proxy_or_queue):
    if hasattr(proxy_or_queue, "request"):
        return proxy_or_queue
    return _QueueProxy(proxy_or_queue)


class SceneTools:
    """Tool bodies, separated from transport so they can be tested headless."""

    def __init__(self, proxy_or_queue, *, timeout: float = REQUEST_TIMEOUT_S) -> None:
        # Normalize here so no caller has to remember: the Qt front-end passes a
        # QtRendererProxy, the GLFW one a bare queue.
        self._proxy = _as_proxy(proxy_or_queue)
        self._timeout = timeout

    def _read(self, callback):
        """Run work on the render thread and wait for its reply.

        On timeout the command is cancelled so a still-queued write cannot
        apply minutes later, after the client already saw an error and possibly
        retried. If it had already started running, cancellation fails and the
        outcome is genuinely unknown — say so rather than implying nothing
        happened.
        """
        future = self._proxy.request(callback)
        try:
            return future.result(timeout=self._timeout)
        except TimeoutError:
            if future.cancel():
                raise SceneToolError(
                    f"render thread did not respond within {self._timeout:g}s; "
                    "the request was cancelled and had no effect"
                )
            raise SceneToolError(
                f"render thread did not respond within {self._timeout:g}s; "
                "the request had already started and its outcome is unknown"
            )

    # ── Tools ────────────────────────────────────────────────────────

    def scene_list(
        self,
        path: str = "/",
        depth: int = DEFAULT_LIST_DEPTH,
        kind: str | None = None,
    ) -> dict:
        """Enumerate scene structure. No property values — use scene_get."""
        def read(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            if kind is not None:
                found: list = []
                _collect_by_kind(node, kind, found, depth)
                return {"nodes": found, **_versions(renderer)}
            return {"root": _summarize(node, depth, None), **_versions(renderer)}

        return self._read(read)

    def scene_get(self, path: str) -> dict:
        """Read one node's full properties, with editable flags and bounds."""
        def read(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            # Serialize this node alone: scene_graph_to_dict recurses, and
            # building a whole subtree only to discard it blocks the render
            # thread between frames. Structure belongs to scene_list.
            detached = replace(node, children=[])
            # deepcopy: scene_graph_to_dict places each property's live
            # `metadata` mapping straight into the result, and `replace` is
            # shallow -- without this, a renderer-owned mutable would cross the
            # future boundary and be serialized off-thread.
            node_dict = deepcopy(scene_graph_to_dict(detached))
            return {"node": node_dict, **_versions(renderer)}

        return self._read(read)

    def scene_set(self, path: str, property: str, value: Any) -> dict:
        """Write one property.

        Resolves the node once on the render thread, validates type then
        bounds, and dispatches through the same function the Qt dock uses.
        Awaited, so the client is told whether the edit applied or why it did
        not; see the module docstring for why that beats coalescing here.
        """
        def write(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            prop = next((p for p in node.properties if p.name == property), None)
            if prop is None:
                names = ", ".join(sorted(p.name for p in node.properties)) or "none"
                raise SceneToolError(
                    f"no property {property!r} on {path!r} (has: {names})"
                )
            if not prop.editable:
                raise SceneToolError(f"{property!r} on {path!r} is not editable")

            checked = _check_bounds(prop, _coerce(prop, value))
            reason = apply_scene_property(renderer, node, prop, checked, graph=graph)
            if reason is not None:
                raise SceneToolError(f"cannot set {property!r} on {path!r}: {reason}")
            if prop.type_name not in _RENDERER_OWNED_VALUES:
                # Types the renderer republishes (it rebuilds the node and
                # restores its own placeholder) must not be written back here —
                # the write would be discarded and a later scene_get would
                # disagree with what this call reported.
                prop.value = checked
            return {"applied": {"path": path, "property": property, "value": checked},
                    **_versions(renderer)}

        return self._read(write)


def _coerce(prop, value):
    """Validate and normalize a written value against the property's type.

    A client can send anything JSON can carry, so type is checked before the
    value reaches renderer code — otherwise ``"false"`` becomes ``True``, a
    two-element list reaches transform recomposition, and a NaN silently
    corrupts a material override that only fails later at upload.
    """
    type_name = getattr(prop, "type_name", "")

    if type_name == "bool":
        if not isinstance(value, bool):
            raise SceneToolError(f"{prop.name} expects a boolean, got {value!r}")
        return value

    if type_name in ("float", "color3f"):
        return _finite(prop, value) if type_name == "float" else _vector(prop, value, 3)

    if type_name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SceneToolError(f"{prop.name} expects an integer, got {value!r}")
        return value

    if type_name == "vec3f":
        return _vector(prop, value, 3)

    if type_name in ("string", "token", "asset", "texture_file", "lens_file"):
        if not isinstance(value, str):
            raise SceneToolError(f"{prop.name} expects a string, got {value!r}")
        return value

    return value


def _finite(prop, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneToolError(f"{prop.name} expects a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SceneToolError(f"{prop.name} must be finite, got {value!r}")
    return number


def _vector(prop, value, length: int) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SceneToolError(
            f"{prop.name} expects {length} numbers, got {value!r}"
        )
    if len(value) != length:
        raise SceneToolError(
            f"{prop.name} expects exactly {length} numbers, got {len(value)}"
        )
    return tuple(_finite(prop, component) for component in value)


def _check_bounds(prop, value):
    """Reject out-of-bounds writes; never clamp. See design D12.

    The published ranges are editor affordances, not legal bounds — the dock
    itself raises a growable property's spin-box maximum far past ``max``.
    Clamping would make a client less capable than the operator and would
    silently alter a render (``roughness=0.0`` quietly becoming ``0.04``).
    """
    meta = prop.metadata or {}
    if meta.get("growable"):
        return value
    lo, hi = meta.get("min"), meta.get("max")
    if lo is None and hi is None:
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if lo is not None and value < lo:
        raise SceneToolError(
            f"{prop.name}={value} is below its minimum {lo} (range {lo}..{hi})"
        )
    if hi is not None and value > hi:
        raise SceneToolError(
            f"{prop.name}={value} is above its maximum {hi} (range {lo}..{hi})"
        )
    return value


# ── Transport ────────────────────────────────────────────────────────

def build_app(tools: SceneTools, token: str, port: int):
    """Build the MCP streamable-HTTP ASGI app with the request guards applied."""
    from mcp.server.fastmcp import FastMCP

    from mcp.server.fastmcp.exceptions import ToolError

    server = FastMCP("skinny")

    def _wrap(fn):
        """Preserve the signature FastMCP reflects on, and signal real errors.

        ``functools.wraps`` alone is not enough: it copies ``__wrapped__``, and
        FastMCP builds the input schema from the *signature*. Without an explicit
        ``__signature__`` the tool is advertised as ``(*args, **kwargs)`` and no
        client can call it.

        Failures raise ``ToolError`` rather than returning an ``{"error": ...}``
        payload, which FastMCP would report as a *successful* call — leaving a
        client unable to tell a failed edit from an applied one.
        """
        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except SceneToolError as exc:
                raise ToolError(str(exc)) from exc
            except (ValueError, RuntimeError) as exc:
                # Renderer exceptions arrive here via Future.result().
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

        call.__signature__ = inspect.signature(fn)
        return call

    for tool in (tools.scene_list, tools.scene_get, tools.scene_set):
        server.tool()(_wrap(tool))

    app = server.streamable_http_app()

    class Guard:
        """Refuse browser and unauthenticated requests before they reach a tool."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                reason = check_request(dict(scope.get("headers") or []), token, port)
                if reason is not None:
                    await send({
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain")],
                    })
                    await send({"type": "http.response.body", "body": reason.encode()})
                    return
            await self.inner(scope, receive, send)

    return Guard(app)


def serve(proxy_or_queue, port: int, sock) -> threading.Thread:
    """Run the MCP server on a daemon thread over an already-bound socket.

    Signal handlers are explicitly not installed. ``uvicorn`` installs SIGINT and
    SIGTERM handlers by default; from a non-main thread that raises, and were it
    to succeed it would overwrite ``MetalContext``'s chained teardown handlers —
    the backstop that stops an abandoned kernel from wedging the GPU until
    reboot.
    """
    import uvicorn

    token = load_or_create_token()
    app = build_app(SceneTools(proxy_or_queue), token, port)
    config = uvicorn.Config(app, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # never touch process signals

    def run() -> None:
        try:
            server.run(sockets=[sock])
        except Exception:  # noqa: BLE001 - a dead server must not kill the render loop
            log.exception("MCP server stopped")

    thread = threading.Thread(target=run, name="skinny-mcp", daemon=True)
    thread.start()
    return thread


def start(proxy_or_queue, port: int) -> threading.Thread | None:
    """Bind, start, and print the registration line. ``None`` if the port is taken.

    A collision leaves the renderer running with MCP disabled — it does not
    exit, and does not silently pick another port.
    """
    from skinny.mcp_auth import bind_loopback_socket

    try:
        sock = bind_loopback_socket(port)
    except OSError as exc:
        log.warning(
            "MCP server unavailable: port %d is already in use (%s). "
            "The renderer is running normally without it.", port, exc,
        )
        return None

    thread = serve(proxy_or_queue, port, sock)
    print(f"\nMCP server on http://127.0.0.1:{port}/mcp — register it with:\n")
    print(registration_command(port))
    print()
    return thread
