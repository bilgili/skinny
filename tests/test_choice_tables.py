"""Gate for the enumerated-axis owner (change choice-table-owners).

`choice_tables.py` owns each axis's values, labels and indices. Two guarantees:

* **Golden content** — the projections still equal the exact literals every
  consumer used before this change, so repointing them changed nothing.
* **Source gate** — no consumer restates an axis's membership as its own list /
  dict literal. A reintroduced mirror fails here rather than silently drifting.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from skinny import choice_tables as ct

SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


# ── golden content (repointing changed nothing) ──────────────────────────────

def test_integrator_golden():
    assert ct.labels(ct.INTEGRATOR) == ["Path", "BDPT", "SPPM", "MLT"]
    assert ct.tokens(ct.INTEGRATOR) == ("path", "bdpt", "sppm", "mlt")
    assert ct.index_by_token(ct.INTEGRATOR) == {"path": 0, "bdpt": 1, "sppm": 2, "mlt": 3}
    assert ct.index_to_token(ct.INTEGRATOR) == {0: "path", 1: "bdpt", 2: "sppm", 3: "mlt"}


def test_tonemap_golden():
    assert ct.labels(ct.TONEMAP) == ["ACES", "Reinhard", "Hable", "Linear"]
    assert ct.tokens(ct.TONEMAP) == ("aces", "reinhard", "hable", "linear")
    assert ct.index_by_token(ct.TONEMAP) == {"aces": 0, "reinhard": 1, "hable": 2, "linear": 3}


def test_execution_mode_golden():
    assert ct.tokens(ct.EXECUTION_MODE) == ("megakernel", "wavefront")


def test_reuse_golden():
    assert ct.labels(ct.REUSE) == ["None", "ReSTIR DI"]
    assert list(ct.tokens(ct.REUSE)) == ["none", "restir-di"]


def test_runtime_only_axes_golden():
    assert ct.labels(ct.DETAIL_MAPS) == ["On", "Off"]
    assert ct.labels(ct.RESTIR_COMBINATION) == ["Unbiased (GRIS)", "Biased (ΣM)"]
    # Runtime-only axes carry no token.
    assert ct.tokens(ct.DETAIL_MAPS) == ()
    assert ct.tokens(ct.RESTIR_COMBINATION) == ()


def test_proposal_preset_golden():
    assert ct.labels(ct.PROPOSAL_PRESET) == [
        "BSDF", "BSDF + Env", "Env", "BSDF + Neural", "Neural"]
    assert ct.label_token_pairs(ct.PROPOSAL_PRESET) == [
        ("BSDF", "bsdf"), ("BSDF + Env", "bsdf,env"), ("Env", "env"),
        ("BSDF + Neural", "bsdf,neural"), ("Neural", "neural")]


def test_consumers_are_projections():
    """The four host-side lookup mirrors are now built from the owner."""
    from skinny import cli_common, frame_plan, headless, render_envelope
    assert cli_common.INTEGRATOR_INDEX == ct.index_by_token(ct.INTEGRATOR)
    assert headless._INTEGRATORS == ct.index_by_token(ct.INTEGRATOR)
    assert headless._TONEMAPS == ct.index_by_token(ct.TONEMAP)
    assert render_envelope.INTEGRATORS == ct.tokens(ct.INTEGRATOR)
    assert render_envelope.EXECUTION_MODES == ct.tokens(ct.EXECUTION_MODE)
    assert frame_plan.INTEGRATOR_NAMES == ct.index_to_token(ct.INTEGRATOR)


def test_proxy_placeholders_match_the_renderer_lists():
    """The six enumerated-axis proxy placeholders that used to drift (missing MLT,
    `['Filmic']`, `['Off']` reuse/detail-maps, `['Unbiased','Biased']`, `['bsdf']`)
    are now the owner's labels."""
    from skinny.render_session import _default_choice_names
    d = _default_choice_names()
    assert d["integrator_modes"] == ct.labels(ct.INTEGRATOR)
    assert d["tonemap_modes"] == ct.labels(ct.TONEMAP)
    assert d["reuse_modes"] == ct.labels(ct.REUSE)
    assert d["detail_maps_modes"] == ct.labels(ct.DETAIL_MAPS)
    assert d["restir_combination_modes"] == ct.labels(ct.RESTIR_COMBINATION)
    assert d["proposal_preset_modes"] == ct.labels(ct.PROPOSAL_PRESET)


# ── the CLI / execution-mode projections that are hostless-checkable ─────────

def test_cli_proposals_choices_project_the_owner():
    import argparse

    from skinny import cli_common
    p = argparse.ArgumentParser()
    cli_common.add_render_flags(p, proposals=True)
    action = next(a for a in p._actions if a.dest == "proposals")
    assert tuple(action.choices) == ct.tokens(ct.PROPOSAL_PRESET)


def test_execution_index_constants_are_pinned_to_the_owner():
    """The `EXECUTION_MEGAKERNEL`/`EXECUTION_WAVEFRONT` named indices are kept as
    separate int constants in four leaf modules (to avoid a GPU import cycle);
    pin every copy to the owner's index so they cannot drift from it."""
    idx = ct.index_by_token(ct.EXECUTION_MODE)
    assert (idx["megakernel"], idx["wavefront"]) == (0, 1)
    from skinny import frame_derive, frame_plan, mlt_chain, params
    # params/frame_plan define both; frame_derive/mlt_chain only need WAVEFRONT.
    for mod in (params, frame_plan, frame_derive, mlt_chain):
        if hasattr(mod, "EXECUTION_MEGAKERNEL"):
            assert mod.EXECUTION_MEGAKERNEL == idx["megakernel"], mod.__name__
        if hasattr(mod, "EXECUTION_WAVEFRONT"):
            assert mod.EXECUTION_WAVEFRONT == idx["wavefront"], mod.__name__


# ── source gate: no axis membership restated outside the owner ────────────────
#
# AST-based, so a multiline literal cannot evade it, and it matches an axis's
# *full membership set* (not a substring) so a generic literal elsewhere is not a
# false positive. `detail-maps` is deliberately NOT gated: its set {"On","Off"}
# is not unique — `direct_light_modes` and `furnace_modes` carry the same pair,
# so a membership-set gate cannot tell an owned mirror from a legitimate sibling.
# `execution tokens` is scoped to the two files that owned it, because
# renderer.py's `("megakernel","wavefront")` legitimately names the record-source
# axis (a different axis) with the same token set.

_CONSUMERS = [
    "cli_common.py", "headless.py", "render_envelope.py", "frame_plan.py",
    "renderer.py", "render_session.py",
]

# (name, membership frozenset, only_files | None)
_GATED_AXES = [
    ("integrator tokens", frozenset(ct.tokens(ct.INTEGRATOR)), None),
    ("integrator labels", frozenset(ct.labels(ct.INTEGRATOR)), None),
    ("tonemap tokens", frozenset(ct.tokens(ct.TONEMAP)), None),
    ("tonemap labels", frozenset(ct.labels(ct.TONEMAP)), None),
    ("execution labels", frozenset(ct.labels(ct.EXECUTION_MODE)), None),
    ("execution tokens", frozenset(ct.tokens(ct.EXECUTION_MODE)),
     {"render_envelope.py", "cli_common.py"}),
    ("reuse labels", frozenset(ct.labels(ct.REUSE)), None),
    ("reuse tokens", frozenset(ct.tokens(ct.REUSE)), None),
    ("restir combination labels", frozenset(ct.labels(ct.RESTIR_COMBINATION)), None),
    ("proposal tokens", frozenset(ct.tokens(ct.PROPOSAL_PRESET)), None),
    ("proposal labels", frozenset(ct.labels(ct.PROPOSAL_PRESET)), None),
]


def _literal_string_sets(text: str) -> list[frozenset[str]]:
    """Every list/tuple/dict literal's set of string constants (its elements, or
    a dict's keys, or a dict's values) — the shapes an axis mirror can take."""
    out: list[frozenset[str]] = []

    def _strs(nodes) -> frozenset[str]:
        return frozenset(
            n.value for n in nodes
            if isinstance(n, ast.Constant) and isinstance(n.value, str))

    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.List, ast.Tuple)):
            s = _strs(node.elts)
            if s:
                out.append(s)
        elif isinstance(node, ast.Dict):
            for group in (node.keys, node.values):
                s = _strs(group)
                if s:
                    out.append(s)
    return out


@pytest.mark.parametrize("name,members,only_files", _GATED_AXES)
def test_no_axis_mirror_outside_the_owner(name, members, only_files):
    offenders = []
    for fname in _CONSUMERS:
        if only_files is not None and fname not in only_files:
            continue
        if members in _literal_string_sets((SRC / fname).read_text(encoding="utf-8")):
            offenders.append(fname)
    assert offenders == [], (
        f"axis '{name}' membership {sorted(members)} restated as a literal in: "
        f"{offenders} — project choice_tables instead")


@pytest.mark.parametrize("name,members,only_files", _GATED_AXES)
def test_gate_detects_a_synthetic_mirror(name, members, only_files):
    """Negative control (git-free): the AST gate must flag a literal of each
    axis's membership, so a green gate means 'no mirror', not 'gate is vacuous'."""
    snippet = "x = [" + ", ".join(repr(m) for m in sorted(members)) + "]\n"
    assert members in _literal_string_sets(snippet)
