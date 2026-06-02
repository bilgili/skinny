"""Concrete reuse modes shipped with the seam.

Only the identity baseline ships now: it forwards direct lighting to the stock
NEE and spawns the indirect ray as before (parity). ReSTIR-style reservoir
reuse is a follow-up change that adds a ReusePlugin owning its passes/buffers.
"""

from __future__ import annotations

from .plugin import ReusePlugin


class IdentityReuse(ReusePlugin):
    """No reuse: stock NEE + stock indirect spawn."""

    name = "none"
    cli_token = "none"
    reuse_mode = 0          # FrameConstants.reuseMode identity
