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


_EXEC_INDEX_MODULES = ("params.py", "frame_plan.py", "frame_derive.py", "mlt_chain.py")


def _assigned_values(text: str, names: set[str]) -> list[ast.expr]:
    """The RHS expression of every module-level assignment to one of `names`."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    out.append(node.value)
    return out


@pytest.mark.parametrize("module", _EXEC_INDEX_MODULES)
def test_execution_index_constants_are_derived_not_literal(module):
    """The `EXECUTION_MEGAKERNEL`/`EXECUTION_WAVEFRONT` named indices must be
    *derived* from the owner, not restated as a literal `0`/`1` — reverting to a
    literal (even the currently-correct value) fails here, not just a wrong one."""
    values = _assigned_values(
        (SRC / module).read_text(encoding="utf-8"),
        {"EXECUTION_MEGAKERNEL", "EXECUTION_WAVEFRONT"})
    assert values, f"{module} declares no execution-index constant"
    for v in values:
        assert not isinstance(v, ast.Constant), \
            f"{module} restates an execution index as a literal; derive it from choice_tables"


def test_execution_index_values_match_the_owner():
    idx = ct.index_by_token(ct.EXECUTION_MODE)
    assert (idx["megakernel"], idx["wavefront"]) == (0, 1)
    from skinny import frame_derive, frame_plan, mlt_chain, params
    for mod in (params, frame_plan, frame_derive, mlt_chain):
        if hasattr(mod, "EXECUTION_MEGAKERNEL"):
            assert mod.EXECUTION_MEGAKERNEL == idx["megakernel"], mod.__name__
        if hasattr(mod, "EXECUTION_WAVEFRONT"):
            assert mod.EXECUTION_WAVEFRONT == idx["wavefront"], mod.__name__


# ── source gate: no axis membership restated outside the owner ────────────────
#
# Scans EVERY module under src/skinny (not a hand-picked list). AST-based, so a
# multiline literal cannot evade it. A literal's string set *hits* an axis when it
# equals the membership, is a superset of it (axis + an extra sentinel), or — for
# an axis of four or more members — is the membership missing exactly one element
# (the historic "integrator list dropped MLT" failure). Smaller subsets are not
# hits: an axis's proper 1/2-element subset collides with legitimate sibling
# literals (e.g. {"sppm","mlt"} = render_envelope.WAVEFRONT_ONLY_INTEGRATORS,
# {"Off"} everywhere), so only the near-complete band is unambiguous.
#
# Per (axis, file) an allowance records the legitimate hits that are NOT mirrors:
# `renderer.py` carries exactly one execution-token literal — the record-source
# axis `("megakernel","wavefront")`, a different axis — so its allowance is 1 and a
# *second* execution-token literal there is still caught. `detail-maps` is not
# gated at all: its {"On","Off"} set is shared by `direct_light_modes` /
# `furnace_modes`, so even exact matching cannot tell an owned mirror from a
# sibling.

_CONSUMERS = sorted(
    str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if p.name != "choice_tables.py")

# (name, membership frozenset, {file: allowed legitimate hit count})
_GATED_AXES = [
    ("integrator tokens", frozenset(ct.tokens(ct.INTEGRATOR)), {}),
    ("integrator labels", frozenset(ct.labels(ct.INTEGRATOR)), {}),
    ("tonemap tokens", frozenset(ct.tokens(ct.TONEMAP)), {}),
    ("tonemap labels", frozenset(ct.labels(ct.TONEMAP)), {}),
    ("execution labels", frozenset(ct.labels(ct.EXECUTION_MODE)), {}),
    ("execution tokens", frozenset(ct.tokens(ct.EXECUTION_MODE)), {"renderer.py": 1}),
    ("reuse labels", frozenset(ct.labels(ct.REUSE)), {}),
    ("reuse tokens", frozenset(ct.tokens(ct.REUSE)), {}),
    ("restir combination labels", frozenset(ct.labels(ct.RESTIR_COMBINATION)), {}),
    ("proposal tokens", frozenset(ct.tokens(ct.PROPOSAL_PRESET)), {}),
    ("proposal labels", frozenset(ct.labels(ct.PROPOSAL_PRESET)), {}),
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


def _axis_hits(members: frozenset[str], sets: list[frozenset[str]]) -> int:
    """How many literals mirror `members`: exact, superset, or (for a 4+-member
    axis) missing exactly one member. Narrower subsets are not counted."""
    big = len(members) >= 4
    n = 0
    for s in sets:
        if s == members or members < s:
            n += 1
        elif big and s < members and len(s) == len(members) - 1:
            n += 1
    return n


@pytest.mark.parametrize("name,members,allow", _GATED_AXES)
def test_no_axis_mirror_outside_the_owner(name, members, allow):
    offenders = []
    for fname in _CONSUMERS:
        hits = _axis_hits(members, _literal_string_sets((SRC / fname).read_text(encoding="utf-8")))
        if hits > allow.get(fname, 0):
            offenders.append(f"{fname} ({hits} hit(s), {allow.get(fname, 0)} allowed)")
    assert offenders == [], (
        f"axis '{name}' membership {sorted(members)} restated as a literal in: "
        f"{offenders} — project choice_tables instead")


@pytest.mark.parametrize("name,members,allow", _GATED_AXES)
def test_gate_detects_synthetic_mirrors(name, members, allow):
    """Negative control (git-free): the gate must flag an exact mirror, a superset
    (extra sentinel), and — for a 4+-member axis — a missing-one partial. A green
    gate then means 'no mirror', not 'gate is vacuous'."""
    exact = "x = [" + ", ".join(repr(m) for m in sorted(members)) + "]\n"
    superset = "x = [" + ", ".join(repr(m) for m in sorted(members | {"__extra__"})) + "]\n"
    assert _axis_hits(members, _literal_string_sets(exact)) == 1
    assert _axis_hits(members, _literal_string_sets(superset)) == 1
    if len(members) >= 4:
        partial = "x = [" + ", ".join(repr(m) for m in sorted(members)[:-1]) + "]\n"
        assert _axis_hits(members, _literal_string_sets(partial)) == 1
