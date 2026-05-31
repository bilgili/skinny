"""Wavefront execution backend — staged compute passes (P1 §P1-F).

Vulkan only. Owns the per-stream path-state + queue buffers and (in later
steps) the stage compute pipelines, dispatched as a bounce loop before the
frame's accumulation step. Modelled on ``vk_skinning.SkinningPasses``: a
self-contained owner of its own GPU resources, driven by the renderer.

Buffer sizes come from ``wavefront_layout.queue_buffer_sizes`` — the single
source of truth shared with the GPU-free layout tests. This module is the
Phase-1 integration scaffold: buffer allocation lands first; the
generate/intersect/logic/shade stage pipelines and the bounce loop follow,
wired to the renderer's shared scene bindings.
"""

from __future__ import annotations

from skinny.vk_compute import StorageBuffer
from skinny.wavefront_layout import queue_buffer_sizes


class WavefrontPasses:
    """Owns the wavefront stage buffers for a given stream size + material count.

    The path-state and queue buffers are sized to the active path stream; the
    per-material counting-sort buffers to the material count. They are
    reallocated when either changes (a scene reload that adds materials, or a
    stream-size retune).
    """

    def __init__(self, ctx, stream_size: int, num_materials: int) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.num_materials = int(num_materials)
        self.buffer_sizes = queue_buffer_sizes(self.stream_size, self.num_materials)
        self.buffers: dict[str, StorageBuffer] = {
            name: StorageBuffer(ctx, size) for name, size in self.buffer_sizes.items()
        }

    def destroy(self) -> None:
        """Release all stage buffers. Idempotent."""
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
