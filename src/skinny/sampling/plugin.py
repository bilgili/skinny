"""SamplingPlugin base classes for the two seams.

A plugin owns its host-side lifecycle and contributes to the FrameConstants
uniform that drives the Slang seam (shaders/sampling/proposal.slang). Cheap
analytic plugins (BSDF, env-importance) own no GPU passes or buffers — they
only set their proposal-mask bit + selection weight. Heavy plugins (a neural
inference pre-pass, ReSTIR reservoirs — later changes) attach passes, buffers,
and descriptor bindings through the same base without changing the interface.
"""

from __future__ import annotations

from enum import Enum


class AttachPoint(Enum):
    """Where in the integrator a plugin attaches."""

    PROPOSAL = "proposal"   # directional proposal at the BSDF bounce
    REUSE = "reuse"         # resampling/reuse around NEE + indirect


class SamplingPlugin:
    """Base for both seams. Lifecycle hooks are no-ops for analytic plugins."""

    name: str = ""
    cli_token: str = ""
    attach_point: AttachPoint

    def reset(self) -> None:
        """Drop any accumulated state (e.g. reservoirs, replay buffer)."""

    # GPU lifecycle — overridden only by plugins that own passes/buffers.
    def build(self, ctx, res) -> None:  # noqa: ANN001 - ctx/res are backend objects
        """Allocate GPU passes/buffers/bindings. No-op for analytic plugins."""

    def destroy(self) -> None:
        """Release anything `build` allocated."""

    def resize(self, width: int, height: int) -> None:
        """React to a framebuffer resize (re-allocate per-pixel buffers)."""


class ProposalPlugin(SamplingPlugin):
    """A directional proposal MIS-mixed at the bounce.

    ``mask_bit`` is this proposal's FrameConstants.proposalMask bit; its slot
    index (bit position) selects its entry in proposalAlpha. ``default_weight``
    is the unnormalized one-sample-MIS selection weight.
    """

    attach_point = AttachPoint.PROPOSAL
    mask_bit: int = 0
    default_weight: float = 1.0


class ReusePlugin(SamplingPlugin):
    """A reuse/resampling mode around NEE + indirect lighting.

    ``reuse_mode`` is the FrameConstants.reuseMode value. Reuse is
    pass-structural — switching it rebuilds the renderer's passes (like the
    execution mode), it is not a per-bounce uniform branch.
    """

    attach_point = AttachPoint.REUSE
    reuse_mode: int = 0
