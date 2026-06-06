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


class NeuralProposal(ProposalPlugin):
    """Learned neural spline-flow directional proposal (bit2).

    Frozen, offline-trained, wavefront-only. Like ``RestirDiReuse``, this is a
    thin selector: the GPU state — the weight buffers (scene-set bindings
    33/34/35) and the per-lane neural pre-pass (vk_wavefront.WavefrontNeural
    ProposalPass) — is owned renderer-side and capability-gated off on the
    megakernel. The renderer loads ``weights_path`` (or bakes a dummy net for the
    1a plumbing bring-up). Selecting it sets the mask bit + ``proposalAlpha.z``;
    the seam (shaders/sampling/proposal.slang) mixes it in via one-sample MIS.
    """

    name = "neural"
    cli_token = "neural"
    mask_bit = 0x4          # PROPOSAL_NEURAL — bit2
    default_weight = 1.0

    def __init__(self, weights_path=None):
        # Default None → the renderer resolves a per-scene weights file (or bakes
        # a dummy). Kept on the instance so a CLI/settings override threads here.
        self.weights_path = weights_path
