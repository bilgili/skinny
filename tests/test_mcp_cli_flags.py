"""CLI surface for the in-process MCP server.

The port parser is the security-relevant piece here: it must refuse anything
carrying a host component, so the loopback-only bind cannot be widened through
configuration.
"""

from __future__ import annotations

import argparse

import pytest

from skinny.cli_common import (
    DEFAULT_MCP_PORT,
    add_render_flags,
    reject_mcp_unsupported,
)


def _parser(**kwargs) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="test")
    add_render_flags(p, **kwargs)
    return p


def test_mcp_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SKINNY_MCP", raising=False)
    args = _parser().parse_args([])
    assert args.mcp is False
    assert args.mcp_port == DEFAULT_MCP_PORT


def test_env_enables_mcp(monkeypatch) -> None:
    monkeypatch.setenv("SKINNY_MCP", "1")
    assert _parser().parse_args([]).mcp is True


def test_flag_beats_env_for_port(monkeypatch) -> None:
    monkeypatch.setenv("SKINNY_MCP_PORT", "9000")
    assert _parser().parse_args([]).mcp_port == 9000
    assert _parser().parse_args(["--mcp-port", "9100"]).mcp_port == 9100


@pytest.mark.parametrize(
    "value",
    ["0.0.0.0:8765", "127.0.0.1:8765", "localhost:8765", ":8765", "http://x/8765"],
)
def test_port_rejects_a_host_component(value) -> None:
    """A host must be refused outright, not parsed and ignored."""
    with pytest.raises(SystemExit) as excinfo:
        _parser().parse_args(["--mcp-port", value])
    assert "127.0.0.1" in str(excinfo.value) or "port number only" in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "65536", "notaport", "-1"])
def test_port_rejects_out_of_range_and_garbage(value) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--mcp-port", value])


def test_flags_suppressed_for_non_hosting_front_ends() -> None:
    """Headless and web have no queue to host the server on."""
    p = _parser(mcp=False)
    with pytest.raises(SystemExit):
        p.parse_args(["--mcp"])


def test_reject_is_a_noop_when_mcp_is_off() -> None:
    reject_mcp_unsupported(False)  # must not raise even without the extra


def test_reject_names_the_extra_when_dependency_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("mcp"):
            raise ImportError("no mcp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        reject_mcp_unsupported(True)
    assert "[mcp]" in str(excinfo.value)


def test_reject_requires_streamable_http_capable_sdk(monkeypatch) -> None:
    """mcp>=1.8 adds streamable_http_app; an older SDK imports but then fails."""
    import types

    fake = types.ModuleType("mcp.server.fastmcp")
    fake.FastMCP = type("FastMCP", (), {})  # no streamable_http_app
    monkeypatch.setitem(__import__("sys").modules, "mcp.server.fastmcp", fake)

    with pytest.raises(SystemExit, match="1.8"):
        reject_mcp_unsupported(True)
