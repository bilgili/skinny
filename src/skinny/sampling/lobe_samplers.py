"""Per-lobe sampler registry for the flat / std_surface BSDF.

Mirrors ``sampling/proposals.py``: each :class:`LobeSamplerStrategy` names a
draw/density strategy, the lobes it is valid for, its Slang dispatch id
(``FLAT_SAMPLER_*`` in ``shaders/materials/flat/flat_lobes.slang``), and a CLI
token. The renderer folds the per-lobe selection into the packed
``FrameConstants.flatLobeSamplers`` uint (8 bits/lobe: ``coat | spec<<8 |
diff<<16``); ``flat_material.slang`` unpacks it. No new GPU bindings — the seam
reuses ``FrameConstants``, like the analytic directional proposals.
"""

from __future__ import annotations

from dataclasses import dataclass

# Lobe indices — match LOBE_* in flat_lobes.slang. They are also the byte offset
# of each lobe in the packed field (coat<<0, spec<<8, diff<<16).
LOBE_COAT = 0
LOBE_SPEC = 1
LOBE_DIFFUSE = 2
LOBE_NAMES = ("coat", "spec", "diffuse")

# Shader dispatch ids — match FLAT_SAMPLER_* in flat_lobes.slang.
SAMPLER_NATIVE = 0
SAMPLER_BASIS = 1
SAMPLER_UNIFORM = 2


@dataclass(frozen=True)
class LobeSamplerStrategy:
    """One registered draw/density strategy for one or more lobes."""

    name: str                      # GUI display name
    cli_token: str                 # --lobe-samplers token
    shader_id: int                 # FLAT_SAMPLER_* in flat_lobes.slang
    valid_lobes: tuple[int, ...]   # lobes this strategy may be selected for


# Registry. Native first so it is index 0 — the default — in every lobe's list.
# Native is the 2023 spherical-cap VNDF (coat/spec) / cosine (diffuse); the
# alternates are the Heitz-2018 basis-form VNDF (a different warp of the same GGX
# visible-normal distribution) and uniform-hemisphere.
LOBE_SAMPLERS: tuple[LobeSamplerStrategy, ...] = (
    LobeSamplerStrategy(
        "Native", "native", SAMPLER_NATIVE, (LOBE_COAT, LOBE_SPEC, LOBE_DIFFUSE)
    ),
    LobeSamplerStrategy(
        "Heitz-2018 basis VNDF", "basis", SAMPLER_BASIS, (LOBE_COAT, LOBE_SPEC)
    ),
    LobeSamplerStrategy(
        "Uniform-hemisphere", "uniform", SAMPLER_UNIFORM, (LOBE_DIFFUSE,)
    ),
)

# Lobe key aliases accepted on the command line.
_KEY_TO_LOBE = {
    "coat": LOBE_COAT,
    "spec": LOBE_SPEC,
    "diff": LOBE_DIFFUSE,
    "diffuse": LOBE_DIFFUSE,
}


def strategies_for_lobe(lobe: int) -> list[LobeSamplerStrategy]:
    """Registered strategies valid for ``lobe`` (native first)."""
    return [s for s in LOBE_SAMPLERS if lobe in s.valid_lobes]


def lobe_sampler_modes(lobe: int) -> list[str]:
    """Display names for a lobe's selector — the renderer ``choice_source``."""
    return [s.name for s in strategies_for_lobe(lobe)]


def _shader_id(lobe: int, index: int) -> int:
    """Shader id of the ``index``-th strategy valid for ``lobe`` (clamp→native)."""
    opts = strategies_for_lobe(lobe)
    if 0 <= index < len(opts):
        return opts[index].shader_id
    return SAMPLER_NATIVE


def fold_lobe_samplers(coat_index: int, spec_index: int, diff_index: int) -> int:
    """Three per-lobe selection INDICES → the packed ``flatLobeSamplers`` uint.

    Each index selects into that lobe's :func:`strategies_for_lobe` list; the
    chosen strategy's ``shader_id`` is placed in the lobe's byte (``coat |
    spec<<8 | diff<<16``). Out-of-range indices clamp to native (id 0).
    """
    return (
        _shader_id(LOBE_COAT, int(coat_index))
        | (_shader_id(LOBE_SPEC, int(spec_index)) << 8)
        | (_shader_id(LOBE_DIFFUSE, int(diff_index)) << 16)
    )


def parse_lobe_samplers(token: str) -> tuple[int, int, int]:
    """``'coat=basis,spec=basis,diff=uniform'`` → ``(coat, spec, diff)`` indices.

    Terms name a per-lobe strategy by its ``cli_token``; unspecified lobes stay
    native (index 0). Lobe keys: ``coat``, ``spec``, ``diff``/``diffuse``. Raises
    ``ValueError`` on a malformed term, an unknown lobe, or a strategy not valid
    for the named lobe.
    """
    idx = [0, 0, 0]
    for raw in (token or "").split(","):
        part = raw.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"bad --lobe-samplers term '{part}'; expected lobe=strategy"
            )
        key, _, val = part.partition("=")
        key, val = key.strip().lower(), val.strip().lower()
        if key not in _KEY_TO_LOBE:
            raise ValueError(f"unknown lobe '{key}'; known: coat, spec, diff")
        lobe = _KEY_TO_LOBE[key]
        opts = strategies_for_lobe(lobe)
        match = next((i for i, s in enumerate(opts) if s.cli_token == val), None)
        if match is None:
            valid = [s.cli_token for s in opts]
            raise ValueError(
                f"strategy '{val}' not valid for {LOBE_NAMES[lobe]}; valid: {valid}"
            )
        idx[lobe] = match
    return idx[0], idx[1], idx[2]
