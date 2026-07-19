"""The single-threaded front-end's main loop must run posted commands.

`app.main()` needs a window and a GPU, so this models the loop's contract
instead: commands posted from another thread apply before the frame advances,
and the drain is unconditional.
"""

from __future__ import annotations

import threading
from pathlib import Path

from skinny.render_session import RenderCommandQueue

# Read app.py as text rather than importing it: `skinny.app` pulls in
# `skinny.renderer`, which imports `vulkan` at module load and needs the SDK on
# the dynamic-library path. This keeps the test hostless.
_APP_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "skinny" / "app.py"
).read_text()


class FakeRenderer:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.exposure = 1.0

    def update(self, _dt: float) -> None:
        self.events.append(f"frame(exposure={self.exposure})")


def _one_iteration(queue: RenderCommandQueue, renderer: FakeRenderer) -> None:
    """Mirror of the main loop's ordering: poll -> run_pending -> update."""
    queue.run_pending(renderer)
    renderer.update(1 / 60)


def test_cross_thread_post_applies_before_the_next_frame() -> None:
    queue = RenderCommandQueue()
    renderer = FakeRenderer()

    def poster() -> None:
        def set_exposure(r) -> None:
            r.exposure = 2.5
        queue.post(set_exposure)

    thread = threading.Thread(target=poster)
    thread.start()
    thread.join()

    _one_iteration(queue, renderer)

    assert renderer.events == ["frame(exposure=2.5)"]


def test_empty_queue_does_not_block_the_frame() -> None:
    queue = RenderCommandQueue()
    renderer = FakeRenderer()

    _one_iteration(queue, renderer)

    assert renderer.events == ["frame(exposure=1.0)"]


def test_main_loop_drains_unconditionally() -> None:
    """The drain must not sit behind an `if mcp_enabled` guard."""
    assert "commands.run_pending(renderer)" in _APP_SOURCE

    lines = _APP_SOURCE.splitlines()
    drain_line = next(line for line in lines if "commands.run_pending" in line)
    poll_line = next(line for line in lines if "glfw.poll_events()" in line)
    # Same indent as poll_events -> not nested inside a conditional.
    assert (len(drain_line) - len(drain_line.lstrip())) == (
        len(poll_line) - len(poll_line.lstrip())
    )


def test_drain_precedes_renderer_update_in_the_loop() -> None:
    assert _APP_SOURCE.index("commands.run_pending") < _APP_SOURCE.index(
        "renderer.update(dt)"
    )
