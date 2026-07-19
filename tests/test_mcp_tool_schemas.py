"""The tools must advertise real parameters, and signal failure as failure.

Regression test for a bug the rest of the suite missed entirely: every other
test calls SceneTools directly, so nothing exercised what FastMCP actually
advertises. A wrapper that dropped the signature published `(*args, **kwargs)`
and no client could call any tool.
"""

from __future__ import annotations

import asyncio

import pytest

from skinny.mcp_server import SceneTools, build_app
from skinny.render_session import RenderCommandQueue


@pytest.fixture
def server():
    """The FastMCP instance behind the guarded ASGI app."""
    import gc

    from mcp.server.fastmcp import FastMCP

    before = {id(o) for o in gc.get_objects() if isinstance(o, FastMCP)}
    build_app(SceneTools(RenderCommandQueue()), "tok", 1234)
    made = [
        o for o in gc.get_objects()
        if isinstance(o, FastMCP) and id(o) not in before
    ]
    assert made, "build_app did not create a FastMCP server"
    return made[-1]


def _schemas(server) -> dict[str, dict]:
    tools = asyncio.run(server.list_tools())
    return {t.name: t.inputSchema for t in tools}


def test_all_three_tools_are_advertised(server) -> None:
    assert set(_schemas(server)) == {"scene_list", "scene_get", "scene_set"}


def test_no_tool_advertises_args_kwargs(server) -> None:
    """The exact regression: a signature-erasing wrapper breaks every call."""
    for name, schema in _schemas(server).items():
        props = set(schema.get("properties", {}))
        assert "args" not in props, f"{name} advertises *args"
        assert "kwargs" not in props, f"{name} advertises **kwargs"


def test_scene_list_advertises_its_real_parameters(server) -> None:
    props = _schemas(server)["scene_list"]["properties"]
    assert {"path", "depth", "kind"} <= set(props)


def test_scene_get_advertises_path(server) -> None:
    schema = _schemas(server)["scene_get"]
    assert "path" in schema["properties"]
    assert schema.get("required") == ["path"]


def test_scene_set_advertises_path_property_value(server) -> None:
    schema = _schemas(server)["scene_set"]
    assert {"path", "property", "value"} <= set(schema["properties"])
    assert set(schema.get("required", [])) == {"path", "property", "value"}


def test_tool_descriptions_survive_wrapping(server) -> None:
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        assert tool.description, f"{tool.name} lost its docstring"


def test_tool_failure_raises_rather_than_returning_a_success_payload() -> None:
    """A failed edit must not come back as a successful call.

    Returning `{"error": ...}` would be reported by FastMCP as a *successful*
    tool call, leaving a client unable to tell a rejected edit from an applied
    one.
    """
    from mcp.server.fastmcp.exceptions import ToolError

    renderer = type("R", (), {"scene_graph": None, "_material_version": 0,
                              "_scene_graph_version": 0})()
    queue = RenderCommandQueue()
    tools = SceneTools(queue, timeout=1.0)

    import gc

    from mcp.server.fastmcp import FastMCP

    before = {id(o) for o in gc.get_objects() if isinstance(o, FastMCP)}
    build_app(tools, "tok", 1234)
    srv = [o for o in gc.get_objects()
           if isinstance(o, FastMCP) and id(o) not in before][-1]

    async def call_and_drain():
        task = asyncio.create_task(
            asyncio.to_thread(
                lambda: asyncio.run(srv.call_tool("scene_get", {"path": "/nope"}))
            )
        )
        for _ in range(200):  # act as the render thread
            queue.run_pending(renderer)
            await asyncio.sleep(0.005)
            if task.done():
                break
        return await task

    with pytest.raises(ToolError, match="no scene is loaded"):
        asyncio.run(call_and_drain())
