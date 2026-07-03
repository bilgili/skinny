from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from skinny.ui.qt.render_session import RenderCommandQueue


def test_render_command_queue_drains_fifo() -> None:
    queue = RenderCommandQueue()
    calls: list[str] = []

    queue.post(lambda _renderer: calls.append("first"))
    queue.post(lambda _renderer: calls.append("second"))

    for command in queue.drain():
        command.callback(None)

    assert calls == ["first", "second"]
    assert len(queue) == 0


def test_render_command_queue_coalesces_last_write_without_reordering() -> None:
    queue = RenderCommandQueue()
    calls: list[str] = []

    queue.post(lambda _renderer: calls.append("camera"))
    queue.post(lambda _renderer: calls.append("resize-640"), coalesce_key="resize")
    queue.post(lambda _renderer: calls.append("scene"))
    queue.post(lambda _renderer: calls.append("resize-1280"), coalesce_key="resize")

    for command in queue.drain():
        command.callback(None)

    assert calls == ["camera", "resize-1280", "scene"]


def test_render_command_queue_allows_new_command_after_drain() -> None:
    queue = RenderCommandQueue()

    queue.post(lambda _renderer: None, coalesce_key="resize")
    assert len(queue.drain()) == 1

    queue.post(lambda _renderer: None, coalesce_key="resize")
    assert len(queue.drain()) == 1


def test_render_command_queue_supports_one_shot_reply() -> None:
    queue = RenderCommandQueue()
    future = queue.post_with_reply(lambda renderer: renderer["value"] + 1)

    command = queue.drain()[0]
    command.reply.set_result(command.callback({"value": 41}))

    assert future.result(timeout=0) == 42


def test_render_command_queue_accepts_concurrent_posts() -> None:
    queue = RenderCommandQueue()

    def post_one(i: int) -> None:
        queue.post(lambda _renderer, i=i: i)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(post_one, range(64)))

    commands = queue.drain()

    assert len(commands) == 64
    assert len(queue) == 0
