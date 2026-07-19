"""Server lifecycle: signal safety, bind collision, registration output.

The signal test is the important one. uvicorn installs SIGINT/SIGTERM handlers
by default; those would overwrite MetalContext's chained teardown handlers,
which are the backstop that keeps an abandoned GPU kernel from wedging the
device until reboot (CLAUDE.md, metal-dispatch-hygiene).
"""

from __future__ import annotations

import signal
import threading
import time


from skinny import mcp_server
from skinny.mcp_auth import bind_loopback_socket
from skinny.render_session import RenderCommandQueue


class Proxy:
    def __init__(self) -> None:
        self._commands = RenderCommandQueue()

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _free_port() -> int:
    s = bind_loopback_socket(0)
    port = s.getsockname()[1]
    s.close()
    return port


def test_server_does_not_replace_signal_handlers(monkeypatch) -> None:
    """MetalContext's SIGINT/SIGTERM teardown chain must survive startup."""
    sentinel_int = signal.getsignal(signal.SIGINT)
    sentinel_term = signal.getsignal(signal.SIGTERM)

    port = _free_port()
    sock = bind_loopback_socket(port)
    try:
        thread = mcp_server.serve(Proxy(), port, sock)
        time.sleep(0.5)  # let uvicorn reach its serve() call

        assert signal.getsignal(signal.SIGINT) is sentinel_int
        assert signal.getsignal(signal.SIGTERM) is sentinel_term
        assert thread.daemon, "server thread must not delay process exit"
    finally:
        sock.close()


def test_start_returns_none_on_port_collision(caplog) -> None:
    """A collision leaves the renderer running with MCP disabled."""
    port = _free_port()
    holder = bind_loopback_socket(port)
    try:
        result = mcp_server.start(Proxy(), port)
        assert result is None
        assert any("already in use" in r.message for r in caplog.records)
    finally:
        holder.close()


def test_start_prints_registration_line_without_the_token(capsys) -> None:
    port = _free_port()
    thread = mcp_server.start(Proxy(), port)
    assert thread is not None

    out = capsys.readouterr().out
    assert f"127.0.0.1:{port}/mcp" in out
    assert "claude mcp add --transport http" in out
    assert "mcp_token" in out  # references the file...
    # ...and not its contents.
    from skinny.mcp_auth import load_or_create_token
    assert load_or_create_token() not in out


def test_server_thread_is_daemon() -> None:
    """A non-daemon server thread would hang process exit."""
    port = _free_port()
    sock = bind_loopback_socket(port)
    try:
        thread = mcp_server.serve(Proxy(), port, sock)
        assert isinstance(thread, threading.Thread)
        assert thread.daemon
    finally:
        sock.close()
