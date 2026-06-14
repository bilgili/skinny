"""MetalContext exposes the duck-typed ``gpu_info`` the front-ends read.

The Qt GUI, the web front-end, and ``VideoEncoder`` all consume
``ctx.gpu_info`` off whichever context ``make_context`` returns —
``.name`` for display, ``.preferred_h264_encoder`` for encoder selection.
``VulkanContext`` builds it from physical-device enumeration; the native
Metal context must expose the same surface or every front-end crashes at
startup under the auto→Metal default
(``AttributeError: 'MetalContext' object has no attribute 'gpu_info'``).
"""

from __future__ import annotations

import pytest

from skinny.backend_select import metal_available


def test_metal_context_exposes_gpu_info():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None)
    try:
        info = ctx.gpu_info
        assert isinstance(info.name, str) and info.name
        assert isinstance(info.preferred_h264_encoder, str)
        assert info.preferred_h264_encoder
        assert info.is_discrete is False  # Apple Silicon is unified-memory
    finally:
        ctx.destroy()
