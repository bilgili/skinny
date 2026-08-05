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


def test_execution_index_constants_project_the_owner():
    """The `EXECUTION_MEGAKERNEL`/`EXECUTION_WAVEFRONT` named indices are derived
    from the owner in every module that declares them (device-free leaf modules
    keeping the index without a GPU import cycle), so they cannot drift from it."""
    idx = ct.index_by_token(ct.EXECUTION_MODE)
    assert (idx["megakernel"], idx["wavefront"]) == (0, 1)
    from skinny import frame_derive, frame_plan, mlt_chain, params
    # params/frame_plan declare both; frame_derive/mlt_chain only need WAVEFRONT.
    for mod in (params, frame_plan, frame_derive, mlt_chain):
        if hasattr(mod, "EXECUTION_MEGAKERNEL"):
            assert mod.EXECUTION_MEGAKERNEL == idx["megakernel"], mod.__name__
        if hasattr(mod, "EXECUTION_WAVEFRONT"):
            assert mod.EXECUTION_WAVEFRONT == idx["wavefront"], mod.__name__


# ── source gate: no axis membership restated outside the owner ────────────────
#
# Scans EVERY module under src/skinny (not a hand-picked list). AST-based, so a
# multiline literal cannot evade it, and it matches an axis's *full membership
# set*, so a generic literal elsewhere is not a false positive. Two documented
# carve-outs:
#   * `detail-maps` is NOT gated — its set {"On","Off"} is shared by
#     `direct_light_modes` and `furnace_modes` (legitimate sibling axes), so a
#     membership-set gate cannot tell an owned mirror from a sibling.
#   * `execution tokens` skips renderer.py, whose `("megakernel","wavefront")`
#     legitimately names the record-source axis (a different axis) with the same
#     token set.
# Known limit: exact-set matching catches a *complete* reintroduced mirror, not a
# partial one (a 3-of-4 integrator list). Subset matching is rejected because an
# axis's proper subset collides with real sibling literals (e.g. {"sppm","mlt"} is
# render_envelope.WAVEFRONT_ONLY_INTEGRATORS).

_CONSUMERS = sorted(
    str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if p.name != "choice_tables.py")

# (name, membership frozenset, files to SKIP)
_GATED_AXES = [
    ("integrator tokens", frozenset(ct.tokens(ct.INTEGRATOR)), frozenset()),
    ("integrator labels", frozenset(ct.labels(ct.INTEGRATOR)), frozenset()),
    ("tonemap tokens", frozenset(ct.tokens(ct.TONEMAP)), frozenset()),
    ("tonemap labels", frozenset(ct.labels(ct.TONEMAP)), frozenset()),
    ("execution labels", frozenset(ct.labels(ct.EXECUTION_MODE)), frozenset()),
    ("execution tokens", frozenset(ct.tokens(ct.EXECUTION_MODE)), frozenset({"renderer.py"})),
    ("reuse labels", frozenset(ct.labels(ct.REUSE)), frozenset()),
    ("reuse tokens", frozenset(ct.tokens(ct.REUSE)), frozenset()),
    ("restir combination labels", frozenset(ct.labels(ct.RESTIR_COMBINATION)), frozenset()),
    ("proposal tokens", frozenset(ct.tokens(ct.PROPOSAL_PRESET)), frozenset()),
    ("proposal labels", frozenset(ct.labels(ct.PROPOSAL_PRESET)), frozenset()),
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


@pytest.mark.parametrize("name,members,skip_files", _GATED_AXES)
def test_no_axis_mirror_outside_the_owner(name, members, skip_files):
    offenders = []
    for fname in _CONSUMERS:
        if fname in skip_files:
            continue
        if members in _literal_string_sets((SRC / fname).read_text(encoding="utf-8")):
            offenders.append(fname)
    assert offenders == [], (
        f"axis '{name}' membership {sorted(members)} restated as a literal in: "
        f"{offenders} — project choice_tables instead")


@pytest.mark.parametrize("name,members,skip_files", _GATED_AXES)
def test_gate_detects_a_synthetic_mirror(name, members, skip_files):
    """Negative control (git-free): the AST gate must flag a literal of each
    axis's membership, so a green gate means 'no mirror', not 'gate is vacuous'."""
    snippet = "x = [" + ", ".join(repr(m) for m in sorted(members)) + "]\n"
    assert members in _literal_string_sets(snippet)
