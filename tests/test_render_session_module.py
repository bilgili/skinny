"""Tests for the front-end-neutral render_session module.

Covers the two properties that let a non-Qt caller (the GLFW loop, the MCP
server thread) use the queue: it imports without a GUI toolkit, and the queue
itself settles reply futures so an awaited command never hangs to its timeout.
"""

from __future__ import annotations

import subprocess
import sys

from skinny.render_session import RenderCommandQueue


def test_module_imports_without_gui_toolkit() -> None:
    """Importing the queue must not drag in a GUI toolkit."""
    code = (
        "import sys; import skinny.render_session; "
        "assert 'PySide6' not in sys.modules, sorted(m for m in sys.modules if 'PySide' in m); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_qt_path_still_re_exports() -> None:
    from skinny.render_session import RenderCommandQueue as Root
    from skinny.ui.qt.render_session import RenderCommandQueue as Shim

    assert Root is Shim


def test_run_pending_executes_in_order() -> None:
    queue = RenderCommandQueue()
    calls: list[str] = []
    queue.post(lambda _r: calls.append("first"))
    queue.post(lambda _r: calls.append("second"))

    queue.run_pending(None)

    assert calls == ["first", "second"]
    assert len(queue) == 0


def test_run_pending_settles_reply_with_result() -> None:
    queue = RenderCommandQueue()
    future = queue.post_with_reply(lambda renderer: renderer["value"] * 2)

    queue.run_pending({"value": 21})

    assert future.done()
    assert future.result(timeout=0) == 42


def test_run_pending_settles_reply_with_exception() -> None:
    """A raising callback must complete its future, not leave it pending."""
    queue = RenderCommandQueue()

    def boom(_renderer):
        raise ValueError("kaboom")

    future = queue.post_with_reply(boom)
    errors: list[str] = []

    queue.run_pending(None, on_error=errors.append)

    assert future.done()
    try:
        future.result(timeout=0)
    except ValueError as exc:
        assert "kaboom" in str(exc)
    else:  # pragma: no cover - the future must carry the exception
        raise AssertionError("expected the callback's exception")
    assert errors and "kaboom" in errors[0]


def test_run_pending_continues_after_a_failing_command() -> None:
    queue = RenderCommandQueue()
    calls: list[str] = []

    def boom(_renderer):
        raise RuntimeError("first fails")

    queue.post(boom)
    queue.post(lambda _r: calls.append("second still runs"))

    queue.run_pending(None, on_error=lambda _msg: None)

    assert calls == ["second still runs"]
