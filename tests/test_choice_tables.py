"""Gate for the enumerated-axis owner (change choice-table-owners).

`choice_tables.py` owns each axis's values, labels and indices. Two guarantees:

* **Golden content** — the projections still equal the exact literals every
  consumer used before this change, so repointing them changed nothing.
* **Source gate** — no consumer restates an axis's membership as its own list /
  dict literal. A reintroduced mirror fails here rather than silently drifting.
"""

from __future__ import annotations

import re
import subprocess

import pytest

from skinny import choice_tables as ct

SRC = __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "skinny"

# The commit this change branched from — its sources still carry the mirrors, so
# it is the negative control the source gate must fire on.
BASE_COMMIT = "9ffd5b0"


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


# ── source gate: no axis membership restated outside the owner ────────────────

# Each pattern matches a *code* list/tuple/dict literal of an axis's membership —
# anchored on the opening `[` / `(` / `{` so a docstring that merely names the
# tokens in prose ("`integrator` is one of "path", "bdpt", …") does not match.
# Keyed to the file(s) the mirror lived in — the execution-mode pair is not swept
# in renderer.py, where `("megakernel", "wavefront")` legitimately names the
# record-source axis, a different axis this change does not own.
_AXIS_MIRRORS = {
    "integrator tokens": (r'[\[({]\s*"path"\s*[,:].*"bdpt"\s*[,:].*"sppm"', None),
    "integrator index→token": (r'\{\s*0\s*:\s*"path".*1\s*:\s*"bdpt"', None),
    "integrator labels": (r'[\[(]\s*"Path"\s*,\s*"BDPT"', None),
    "tonemap tokens": (r'[\[({]\s*"aces"\s*[,:].*"reinhard"\s*[,:].*"hable"', None),
    "tonemap labels": (r'[\[(]\s*"ACES"\s*,\s*"Reinhard"', None),
    "execution modes": (r'[\[(]\s*"megakernel"\s*,\s*"wavefront"',
                        {"render_envelope.py", "cli_common.py"}),
}

_CONSUMERS = [
    "cli_common.py", "headless.py", "render_envelope.py", "frame_plan.py",
    "renderer.py", "render_session.py",
]


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("name,spec", list(_AXIS_MIRRORS.items()))
def test_no_axis_mirror_outside_the_owner(name, spec):
    pattern, only_files = spec
    rx = re.compile(pattern)
    offenders = []
    for fname in _CONSUMERS:
        if only_files is not None and fname not in only_files:
            continue
        for ln in _code_lines((SRC / fname).read_text(encoding="utf-8")):
            if rx.search(ln):
                offenders.append(f"{fname}: {ln.strip()}")
    assert offenders == [], f"axis mirror '{name}' reappeared:\n" + "\n".join(offenders)


@pytest.mark.parametrize("name,spec", list(_AXIS_MIRRORS.items()))
def test_gate_fires_on_pre_change_sources(name, spec):
    """Negative control: every pattern must have matched the pre-change tree."""
    pattern, only_files = spec
    rx = re.compile(pattern)
    files = only_files if only_files is not None else set(_CONSUMERS)
    matched = False
    for fname in files:
        got = subprocess.run(
            ["git", "-C", str(SRC.parents[1]), "show", f"{BASE_COMMIT}:src/skinny/{fname}"],
            capture_output=True, text=True)
        if got.returncode != 0:
            pytest.skip(f"base commit {BASE_COMMIT} unavailable")
        if any(rx.search(ln) for ln in _code_lines(got.stdout)):
            matched = True
    assert matched, f"pattern for '{name}' never matched the pre-change tree"
