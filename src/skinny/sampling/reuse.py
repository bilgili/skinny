"""Concrete reuse modes shipped with the seam.

Identity (stock NEE) + ReSTIR DI (reservoir spatiotemporal resampling of
primary-hit direct lighting, wavefront-only).
"""

from __future__ import annotations

from .plugin import ReusePlugin


class IdentityReuse(ReusePlugin):
    """No reuse: stock NEE + stock indirect spawn."""

    name = "none"
    cli_token = "none"
    reuse_mode = 0          # FrameConstants.reuseMode identity


class RestirDiReuse(ReusePlugin):
    """ReSTIR DI: reservoir spatiotemporal resampling of primary-hit direct
    lighting. Wavefront-only — the renderer builds the GPU pass set
    (vk_wavefront.RestirDiPass) and capability-gates to identity on
    megakernel/Metal. The passes/buffers are owned renderer-side; this selector
    carries the reuse_mode uniform value + the name/token.
    """

    name = "restir-di"
    cli_token = "restir-di"
    reuse_mode = 1          # FrameConstants.reuseMode RESTIR_DI
