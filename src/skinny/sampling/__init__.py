"""Pluggable scene-sampling seam (proposal + reuse).

Hosts the directional-proposal mixture at the BSDF bounce and the reuse hook
around NEE/indirect lighting. See docs/superpowers/specs/2026-06-02-pluggable-
scene-sampling-design.md and openspec/changes/pluggable-scene-sampling/.
"""

from .plugin import AttachPoint, ProposalPlugin, ReusePlugin, SamplingPlugin
from .proposals import BsdfProposal, EnvImportanceProposal, NeuralProposal
from .reuse import IdentityReuse
from .registry import (
    PROPOSAL_PLUGINS,
    REUSE_PLUGINS,
    parse_proposals,
    parse_reuse,
    proposal_mask_and_alpha,
)
from .lobe_samplers import (
    LOBE_COAT,
    LOBE_DIFFUSE,
    LOBE_SAMPLERS,
    LOBE_SPEC,
    LobeSamplerStrategy,
    fold_lobe_samplers,
    lobe_sampler_modes,
    parse_lobe_samplers,
    strategies_for_lobe,
)

__all__ = [
    "AttachPoint",
    "SamplingPlugin",
    "ProposalPlugin",
    "ReusePlugin",
    "BsdfProposal",
    "EnvImportanceProposal",
    "NeuralProposal",
    "IdentityReuse",
    "PROPOSAL_PLUGINS",
    "REUSE_PLUGINS",
    "parse_proposals",
    "parse_reuse",
    "proposal_mask_and_alpha",
    "LOBE_COAT",
    "LOBE_SPEC",
    "LOBE_DIFFUSE",
    "LOBE_SAMPLERS",
    "LobeSamplerStrategy",
    "fold_lobe_samplers",
    "lobe_sampler_modes",
    "parse_lobe_samplers",
    "strategies_for_lobe",
]
