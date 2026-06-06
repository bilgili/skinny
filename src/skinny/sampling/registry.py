"""Plugin registry + CLI-token parsing + FrameConstants folding.

Maps plugin names to classes, parses the ``--proposals`` / ``--reuse`` CLI
tokens, and folds an active proposal list into the (proposalMask, proposalAlpha)
pair the renderer packs into FrameConstants.
"""

from __future__ import annotations

from .proposals import BsdfProposal, EnvImportanceProposal, NeuralProposal
from .reuse import IdentityReuse, RestirDiReuse

PROPOSAL_PLUGINS: dict[str, type] = {
    cls.name: cls for cls in (BsdfProposal, EnvImportanceProposal, NeuralProposal)
}
REUSE_PLUGINS: dict[str, type] = {
    cls.name: cls for cls in (IdentityReuse, RestirDiReuse)
}

# Number of proposalAlpha slots in FrameConstants (float4).
NUM_PROPOSAL_SLOTS = 4


def parse_proposals(token: str) -> list:
    """``'bsdf,env'`` → ``[BsdfProposal(), EnvImportanceProposal()]``.

    Empty / falsy input falls back to the BSDF baseline. Raises ValueError on
    an unknown name. Order is preserved but does not affect the mixture (alpha
    is placed by mask-bit slot, not list position).
    """
    names = [t.strip() for t in (token or "").split(",") if t.strip()]
    out: list = []
    for n in names:
        if n not in PROPOSAL_PLUGINS:
            raise ValueError(
                f"unknown proposal '{n}'; known: {sorted(PROPOSAL_PLUGINS)}"
            )
        out.append(PROPOSAL_PLUGINS[n]())
    return out or [BsdfProposal()]


def parse_reuse(token: str):
    """``'none'`` → ``IdentityReuse()``. Raises ValueError on unknown."""
    n = (token or "none").strip() or "none"
    if n not in REUSE_PLUGINS:
        raise ValueError(f"unknown reuse '{n}'; known: {sorted(REUSE_PLUGINS)}")
    return REUSE_PLUGINS[n]()


def proposal_mask_and_alpha(plugins) -> tuple[int, tuple[float, float, float, float]]:
    """Fold active proposals into ``(mask, alpha4)`` for FrameConstants.

    ``mask`` ORs each plugin's ``mask_bit``. ``alpha4`` places each plugin's
    ``default_weight`` at its slot (bit position) and normalizes the active
    slots to sum to 1 — the one-sample-MIS selection distribution.
    """
    mask = 0
    weights = [0.0] * NUM_PROPOSAL_SLOTS
    for p in plugins:
        mask |= p.mask_bit
        slot = p.mask_bit.bit_length() - 1   # 0x1→0, 0x2→1, 0x4→2, 0x8→3
        if 0 <= slot < NUM_PROPOSAL_SLOTS:
            weights[slot] = float(p.default_weight)
    total = sum(weights)
    if total > 0.0:
        weights = [w / total for w in weights]
    return mask, (weights[0], weights[1], weights[2], weights[3])
