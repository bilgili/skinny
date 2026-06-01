"""Indirect-dispatch correctness for the wavefront shade stage (tasks 3.2 /
3.3 / 9.3).

Verifies the de-risking the design calls for: a per-material-queue shade
dispatched via ``vkCmdDispatchIndirect`` (group count read from the
build_args-shaped indirect-args buffer) produces output identical to a
conservative direct dispatch (worst-case groups + the kernel's empty-lane
early-out). Composes the counting-sort layout (per-material counts → exclusive
offsets → grouped queue → per-material indirect args) that build_args.slang and
scatter.slang produce and that are GPU-verified separately; here the focus is
the indirect dispatch mechanism itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import vulkan as vk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"

pytestmark = pytest.mark.gpu

GROUP = 64


def _u32(values):
    return np.asarray(values, dtype=np.uint32).tobytes()


def _run(ctx, record_fn):
    cmd = vk.vkAllocateCommandBuffers(
        ctx.device, vk.VkCommandBufferAllocateInfo(
            commandPool=ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
        flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
    record_fn(cmd)
    vk.vkEndCommandBuffer(cmd)
    vk.vkQueueSubmit(ctx.compute_queue, 1,
                     [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
                     vk.VK_NULL_HANDLE)
    vk.vkQueueWaitIdle(ctx.compute_queue)
    vk.vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, [cmd])


def test_indirect_dispatch_matches_direct_fallback():
    from skinny.vk_compute import StorageBuffer
    from skinny.vk_context import VulkanContext
    from skinny.vk_wavefront import IndirectPaintPass

    ctx = VulkanContext(window=None, width=8, height=8)
    queue_buf = out_buf = indirect_buf = pass_ = None
    try:
        # Per-material counts (material 1 deliberately empty — exercises the
        # 0-group indirect dispatch). Mirror build_args: exclusive prefix-sum
        # offsets + one (ceil(count/GROUP),1,1) per material.
        counts = [3, 0, 5, 2]
        num_mat = len(counts)
        stream = sum(counts)  # 10
        offsets, running = [], 0
        for c in counts:
            offsets.append(running)
            running += c
        indirect = []
        for c in counts:
            indirect += [(c + GROUP - 1) // GROUP, 1, 1]

        # Identity queue: lane i sits at queue position i, so material m owns
        # the contiguous lane range [offsets[m] .. offsets[m]+counts[m]).
        queue = list(range(stream))

        queue_buf = StorageBuffer(ctx, stream * 4)
        queue_buf.upload_sync(_u32(queue))
        out_buf = StorageBuffer(ctx, stream * 4)
        indirect_buf = StorageBuffer(ctx, num_mat * 12, indirect=True)
        indirect_buf.upload_sync(_u32(indirect))

        pass_ = IndirectPaintPass(ctx, SHADER_DIR,
                                  queue_buf.buffer, queue_buf.size,
                                  out_buf.buffer, out_buf.size)
        slots = [(offsets[m], counts[m], m + 1) for m in range(num_mat)]
        worst_groups = (stream + GROUP - 1) // GROUP  # conservative full-stream

        out_buf.fill_zero_sync()
        _run(ctx, lambda cmd: pass_.record_indirect(cmd, slots, indirect_buf.buffer))
        indirect_out = np.frombuffer(out_buf.download_sync(stream * 4), dtype=np.uint32)

        out_buf.fill_zero_sync()
        _run(ctx, lambda cmd: pass_.record_direct(cmd, slots, worst_groups))
        direct_out = np.frombuffer(out_buf.download_sync(stream * 4), dtype=np.uint32)

        # Indirect (exact group counts from the args buffer) == conservative
        # direct (worst-case groups + early-out).
        assert np.array_equal(indirect_out, direct_out), (
            f"indirect {indirect_out.tolist()} != direct {direct_out.tolist()}"
        )
        # And both paint each lane with its material's value (the queue routing).
        expected = np.zeros(stream, dtype=np.uint32)
        for m in range(num_mat):
            for k in range(counts[m]):
                expected[queue[offsets[m] + k]] = m + 1
        assert np.array_equal(indirect_out, expected), (
            f"painted {indirect_out.tolist()} != expected {expected.tolist()}"
        )
    finally:
        if pass_ is not None:
            pass_.destroy()
        for b in (queue_buf, out_buf, indirect_buf):
            if b is not None:
                b.destroy()
        ctx.destroy()
