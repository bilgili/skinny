"""Small command queue used by the Qt render worker.

The queue is deliberately free of Qt imports so it can be unit-tested without a
GUI or GPU. GUI code posts renderer mutations; the render worker drains and
executes them on the thread that owns frame rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class RenderCommand:
    """One renderer-thread operation."""

    callback: Callable[[Any], Any]
    coalesce_key: str | None = None
    reply: Future[Any] | None = None


class RenderCommandQueue:
    """Thread-safe FIFO queue with optional last-write-wins coalescing.

    A coalesced command keeps the position of the first pending command with the
    same key, but replaces its callback. That preserves ordering against
    distinct user actions while preventing resize drags or slider streams from
    growing the queue without bound.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: list[RenderCommand] = []

    def post(
        self,
        callback: Callable[[Any], Any],
        *,
        coalesce_key: str | None = None,
    ) -> None:
        command = RenderCommand(callback, coalesce_key)
        with self._lock:
            if coalesce_key is not None:
                for idx, existing in enumerate(self._pending):
                    if existing.coalesce_key == coalesce_key:
                        self._pending[idx] = command
                        return
            self._pending.append(command)

    def post_with_reply(self, callback: Callable[[Any], Any]) -> Future[Any]:
        """Post a command and return a future completed by the render worker."""
        future: Future[Any] = Future()
        command = RenderCommand(callback, reply=future)
        with self._lock:
            self._pending.append(command)
        return future

    def drain(self) -> list[RenderCommand]:
        with self._lock:
            commands = self._pending
            self._pending = []
        return commands

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
