"""Integration tests for the wavefront execution backend (P1 §P1-F).

Verifies WavefrontPasses against a real (headless) Vulkan device — first the
stage-buffer allocation, with later stages (pipelines, bounce loop) layered on.
"""

from __future__ import annotations

import pytest

from skinny.wavefront_layout import queue_buffer_sizes

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def headless_ctx():
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=64, height=64)
    yield ctx


def test_allocates_all_stage_buffers(headless_ctx):
    from skinny.vk_wavefront import WavefrontPasses

    stream_size, num_materials = 1024, 4
    wp = WavefrontPasses(headless_ctx, stream_size=stream_size, num_materials=num_materials)
    try:
        expected = queue_buffer_sizes(stream_size, num_materials)
        assert set(wp.buffers) == set(expected)
        # StorageBuffer may round tiny buffers up to a driver minimum, so the
        # device buffer must be at least the requested logical size.
        for name, size in expected.items():
            assert wp.buffers[name].size >= size, name
    finally:
        wp.destroy()


def test_buffers_rescale_with_stream_and_materials(headless_ctx):
    from skinny.vk_wavefront import WavefrontPasses

    small = WavefrontPasses(headless_ctx, stream_size=256, num_materials=2)
    big = WavefrontPasses(headless_ctx, stream_size=4096, num_materials=8)
    try:
        assert big.buffers["path_state"].size > small.buffers["path_state"].size
        assert big.buffers["material_count"].size > small.buffers["material_count"].size
    finally:
        small.destroy()
        big.destroy()


def test_destroy_is_idempotent(headless_ctx):
    from skinny.vk_wavefront import WavefrontPasses

    wp = WavefrontPasses(headless_ctx, stream_size=128, num_materials=1)
    wp.destroy()
    wp.destroy()  # second call must not raise
    assert wp.buffers == {}
