"""Concrete directional proposals shipped with the seam.

``BsdfProposal`` is the always-on baseline — it delegates to the material's
own importance sampler, so a ``{bsdf}`` set is bit-identical to the pre-seam
renderer. ``EnvImportanceProposal`` is the tiny second proposal that
importance-samples the environment (reusing the existing env-CDF bindings, no
new GPU state). The mask bits here must match the Slang ``PROPOSAL_*``
constants in shaders/sampling/proposal.slang.
"""

from __future__ import annotations

from .plugin import ProposalPlugin


class BsdfProposal(ProposalPlugin):
    """Sample the material's BSDF (VNDF / cosine). Baseline, always available."""

    name = "bsdf"
    cli_token = "bsdf"
    mask_bit = 0x1          # PROPOSAL_BSDF
    default_weight = 1.0


class EnvImportanceProposal(ProposalPlugin):
    """Importance-sample the environment lighting distribution.

    Reuses the env CDF descriptor bindings — no new buffers. The Slang-side
    mixture math + world↔tangent conversion land in the env follow-up commit;
    selecting it before then is a no-op beyond setting the mask bit.
    """

    name = "env"
    cli_token = "env"
    mask_bit = 0x2          # PROPOSAL_ENV
    default_weight = 1.0
