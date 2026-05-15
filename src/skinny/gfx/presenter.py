"""Surface / swapchain abstraction.

A `Presenter` owns the on-screen present surface (Vulkan swapchain or
Metal CAMetalLayer drawable queue). Headless mode uses a `NullPresenter`
that exposes no images; render_headless() routes around it via
``Backend.render_target``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skinny.gfx.types import Extent2D, Format

if TYPE_CHECKING:
    from skinny.gfx.command import Semaphore
    from skinny.gfx.resources import Image


@dataclass
class AcquiredImage:
    """One swapchain image checked out for the current frame.

    `image` is the backend resource the renderer writes/blits into;
    `index` is the swapchain slot (used to match per-image descriptor sets
    or semaphores); `acquire_semaphore` (Vulkan) signals once the image is
    ready to render into — Metal hides this and may set it to None.
    """

    image: "Image"
    index: int
    acquire_semaphore: "Semaphore | None"


class Presenter(ABC):
    extent: Extent2D
    format: Format

    @abstractmethod
    def image_count(self) -> int: ...

    @abstractmethod
    def acquire(self) -> AcquiredImage: ...

    @abstractmethod
    def present(
        self,
        acquired: AcquiredImage,
        wait_semaphore: "Semaphore | None" = None,
    ) -> None: ...

    @abstractmethod
    def recreate(self, width: int, height: int) -> None: ...

    @abstractmethod
    def destroy(self) -> None: ...
