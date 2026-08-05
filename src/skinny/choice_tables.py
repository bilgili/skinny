"""Single owner for each enumerated render axis's values, labels and indices.

``render_envelope`` owns whether a combo is *valid*; this module owns what the
axes are *called*. Each enumerated axis is one ordered tuple of :class:`Axis`
records — the entry's display ``label``, its ``token`` (the CLI / headless /
persisted string, where the axis has one), and the entry's *index* (its position
in the tuple). Every other view is a projection:

* the CLI's ``--integrator`` / ``--tonemap`` / ``--execution-mode`` ``choices``
  are :func:`tokens`;
* the headless ``str -> index`` lookup dicts are :func:`index_by_token`;
* the renderer's display lists (``integrator_modes`` etc.) are :func:`labels`;
* the GUI-thread proxy's placeholder names are :func:`labels`.

No consumer restates an axis's membership, ordering or labels. The module is
dependency-free — it imports nothing from ``skinny`` — so any consumer (the CLI,
the headless driver, the renderer, the render-session proxy, ``render_envelope``,
``frame_plan``) imports it without a cycle. It owns vocabulary, never validity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One entry of an enumerated axis: its display label and its string token.

    ``token`` is ``None`` for a runtime-only axis with no CLI / persisted string
    (reuse and detail-map and ReSTIR-combination modes are selected by index in
    the GUI, not by a flag value). The entry's *index* is its position in the
    axis tuple — the integer the renderer and the headless driver use.
    """

    label: str
    token: str | None = None


# ── the axes ────────────────────────────────────────────────────────────────

#: Light-transport integrator. Index = renderer ``integrator_index``; token =
#: ``--integrator`` value.
INTEGRATOR: tuple[Axis, ...] = (
    Axis("Path", "path"),
    Axis("BDPT", "bdpt"),
    Axis("SPPM", "sppm"),
    Axis("MLT", "mlt"),
)

#: Tonemap operator. Index = renderer ``tonemap_index``; token = ``--tonemap``
#: value on the headless front-end.
TONEMAP: tuple[Axis, ...] = (
    Axis("ACES", "aces"),
    Axis("Reinhard", "reinhard"),
    Axis("Hable", "hable"),
    Axis("Linear", "linear"),
)

#: GPU execution mode. ``auto`` is a resolution sentinel, not a mode, so it is
#: not an entry here — the CLI prepends it to :func:`tokens`.
EXECUTION_MODE: tuple[Axis, ...] = (
    Axis("Megakernel", "megakernel"),
    Axis("Wavefront", "wavefront"),
)

#: Scene-sampling reuse mode. Token = the persisted reuse key.
REUSE: tuple[Axis, ...] = (
    Axis("None", "none"),
    Axis("ReSTIR DI", "restir-di"),
)

#: Detail-map (secondary normal/displacement) toggle. Runtime-only, no token.
DETAIL_MAPS: tuple[Axis, ...] = (
    Axis("On"),
    Axis("Off"),
)

#: ReSTIR spatial-reuse combination. Runtime-only, no token.
RESTIR_COMBINATION: tuple[Axis, ...] = (
    Axis("Unbiased (GRIS)"),
    Axis("Biased (ΣM)"),
)

#: Directional-proposal preset. Token = the comma-separated proposal-token string
#: ``_active_proposals`` resolves to plugin instances.
PROPOSAL_PRESET: tuple[Axis, ...] = (
    Axis("BSDF", "bsdf"),
    Axis("BSDF + Env", "bsdf,env"),
    Axis("Env", "env"),
    Axis("BSDF + Neural", "bsdf,neural"),
    Axis("Neural", "neural"),
)


# ── projections ───────────────────────────────────────────────────────────────

def labels(axis: tuple[Axis, ...]) -> list[str]:
    """The axis's display labels, in index order (the renderer's display list)."""
    return [e.label for e in axis]


def tokens(axis: tuple[Axis, ...]) -> tuple[str, ...]:
    """The axis's string tokens, in index order (an argparse ``choices`` tuple).

    Only for axes whose every entry has a token — integrator, tonemap, execution
    mode, reuse, proposal preset.
    """
    return tuple(e.token for e in axis if e.token is not None)


def index_by_token(axis: tuple[Axis, ...]) -> dict[str, int]:
    """``token -> index`` (the headless / CLI lookup dict)."""
    return {e.token: i for i, e in enumerate(axis) if e.token is not None}


def index_to_token(axis: tuple[Axis, ...]) -> dict[int, str]:
    """``index -> token`` (the inverse lookup ``frame_plan`` uses)."""
    return {i: e.token for i, e in enumerate(axis) if e.token is not None}


def label_token_pairs(axis: tuple[Axis, ...]) -> list[tuple[str, str]]:
    """``(label, token)`` per entry — the renderer's ``_PROPOSAL_PRESETS`` shape."""
    return [(e.label, e.token) for e in axis if e.token is not None]
