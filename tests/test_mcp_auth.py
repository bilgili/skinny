"""Security tests for the MCP server's token, bind, and request guards.

Each layer is tested on its own, because each covers a different attacker and
none substitutes for another. The loopback-bind test is the highest-consequence
one in the change: a non-loopback bind turns a local tool into a remotely
reachable scene and filesystem control plane.
"""

from __future__ import annotations

import socket
import stat

import pytest

from skinny import mcp_auth
from skinny.mcp_auth import (
    LOOPBACK_HOST,
    bind_loopback_socket,
    check_request,
    load_or_create_token,
    registration_command,
)

PORT = 8765
TOKEN = "test-token-value"


def _headers(**kwargs) -> dict:
    return {k.replace("_", "-"): v for k, v in kwargs.items()}


# ── Token ────────────────────────────────────────────────────────────

def test_token_created_owner_only(tmp_path) -> None:
    path = tmp_path / "mcp_token"
    token = load_or_create_token(path)

    assert token
    mode = path.stat().st_mode
    assert stat.S_IMODE(mode) == 0o600
    assert not mode & (stat.S_IRWXG | stat.S_IRWXO)


def test_token_persists_across_restarts(tmp_path) -> None:
    """A client's stored registration must stay valid after a restart."""
    path = tmp_path / "mcp_token"
    first = load_or_create_token(path)
    second = load_or_create_token(path)
    assert first == second


def test_token_env_override_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SKINNY_MCP_TOKEN", "from-env")
    assert load_or_create_token(tmp_path / "mcp_token") == "from-env"


def test_group_readable_token_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    path = tmp_path / "mcp_token"
    path.write_text("leaky")
    path.chmod(0o644)

    with pytest.raises(SystemExit) as excinfo:
        load_or_create_token(path)
    assert "chmod 600" in str(excinfo.value)


# ── Bind ─────────────────────────────────────────────────────────────

def test_bind_is_loopback_only() -> None:
    sock = bind_loopback_socket(0)
    try:
        host, _port = sock.getsockname()
        assert host == LOOPBACK_HOST
    finally:
        sock.close()


def test_bind_is_not_reachable_off_host() -> None:
    """Belt and braces: the bound address must not be a wildcard."""
    sock = bind_loopback_socket(0)
    try:
        assert sock.getsockname()[0] not in ("0.0.0.0", "::", "")
    finally:
        sock.close()


def test_port_collision_raises_oserror() -> None:
    """Detected at socket creation, before startup reports success."""
    first = bind_loopback_socket(0)
    try:
        taken = first.getsockname()[1]
        with pytest.raises(OSError):
            bind_loopback_socket(taken)
    finally:
        first.close()


def test_non_loopback_bind_is_refused(monkeypatch) -> None:
    """If the host were ever widened, the socket must refuse to serve."""
    real_socket = socket.socket

    class WildcardSocket(real_socket):
        def getsockname(self):  # pretend the bind landed on a wildcard
            return ("0.0.0.0", PORT)

    monkeypatch.setattr(socket, "socket", WildcardSocket)

    with pytest.raises(RuntimeError, match="loopback only"):
        bind_loopback_socket(0)


# ── Request guard ────────────────────────────────────────────────────

def test_valid_request_allowed() -> None:
    headers = _headers(authorization=f"Bearer {TOKEN}", host=f"127.0.0.1:{PORT}")
    assert check_request(headers, TOKEN, PORT) is None


def test_missing_token_refused() -> None:
    assert check_request(_headers(host=f"127.0.0.1:{PORT}"), TOKEN, PORT) is not None


def test_wrong_token_refused() -> None:
    headers = _headers(authorization="Bearer nope", host=f"127.0.0.1:{PORT}")
    assert check_request(headers, TOKEN, PORT) is not None


def test_origin_refused_even_with_a_valid_token() -> None:
    """The drive-by path: a page in the operator's browser."""
    headers = _headers(
        authorization=f"Bearer {TOKEN}",
        host=f"127.0.0.1:{PORT}",
        origin="https://evil.example",
    )
    reason = check_request(headers, TOKEN, PORT)
    assert reason is not None and "Origin" in reason


def test_rebound_host_refused_without_any_origin() -> None:
    """DNS rebinding reaches us with a valid token shape and no Origin."""
    headers = _headers(authorization=f"Bearer {TOKEN}", host="evil.example")
    reason = check_request(headers, TOKEN, PORT)
    assert reason is not None and "Host" in reason


def test_mismatched_port_in_host_refused() -> None:
    headers = _headers(authorization=f"Bearer {TOKEN}", host="127.0.0.1:9999")
    assert check_request(headers, TOKEN, PORT) is not None


def test_localhost_host_accepted() -> None:
    headers = _headers(authorization=f"Bearer {TOKEN}", host=f"localhost:{PORT}")
    assert check_request(headers, TOKEN, PORT) is None


def test_header_lookup_is_case_insensitive() -> None:
    headers = {"AUTHORIZATION": f"Bearer {TOKEN}", "Host": f"127.0.0.1:{PORT}"}
    assert check_request(headers, TOKEN, PORT) is None


def test_byte_headers_supported() -> None:
    headers = {b"authorization": f"Bearer {TOKEN}".encode(), b"host": b"127.0.0.1:8765"}
    assert check_request(headers, TOKEN, PORT) is None


def test_uses_constant_time_comparison() -> None:
    """Guard against a regression to `==`, which leaks the token by timing."""
    import inspect

    source = inspect.getsource(mcp_auth.check_request)
    assert "compare_digest" in source


# ── Registration line ────────────────────────────────────────────────

def test_registration_line_carries_port_and_no_token_value() -> None:
    line = registration_command(PORT)
    assert str(PORT) in line
    assert "mcp_token" in line  # references the file
    assert TOKEN not in line
