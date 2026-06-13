"""Unit tests for the shared render-selection CLI surface (`skinny.cli_common`).

These are pure-argparse / pure-function tests — no GPU, no window, no heavy
front-end imports — so they run anywhere the package imports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from skinny.cli_common import (
    INTEGRATOR_INDEX,
    WALK_CHOICES,
    add_render_flags,
    resolve_walk,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _parser(**kw) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_render_flags(p, **kw)
    return p


# ── resolve_walk ─────────────────────────────────────────────────────

@pytest.mark.parametrize("value", WALK_CHOICES)
def test_resolve_walk_identity(value):
    assert resolve_walk(value) == value


def test_resolve_walk_megakernel_alias():
    # The deprecated execution-axis word resolves to the fused walk.
    assert resolve_walk("megakernel") == "fused"


@pytest.mark.parametrize("value", ["  Fused ", "MEGAKERNEL", "Eye_Light"])
def test_resolve_walk_normalizes_case_and_whitespace(value):
    assert resolve_walk(value) in WALK_CHOICES


def test_resolve_walk_rejects_unknown():
    with pytest.raises(ValueError, match="unknown bdpt walk"):
        resolve_walk("wavefront")


# ── add_render_flags defaults ────────────────────────────────────────

def test_defaults(monkeypatch):
    # No env overrides in play.
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("SKINNY_BDPT_WALK", raising=False)
    ns = _parser().parse_args([])
    # integrator defaults to None (sentinel = "use persisted / renderer default").
    assert ns.integrator is None
    assert ns.execution_mode == "megakernel"
    assert resolve_walk(ns.bdpt_walk) == "fused"


def test_explicit_values():
    ns = _parser().parse_args(
        ["--integrator", "bdpt", "--execution-mode", "wavefront", "--bdpt-walk", "eye"]
    )
    assert ns.integrator == "bdpt"
    assert ns.execution_mode == "wavefront"
    assert resolve_walk(ns.bdpt_walk) == "eye"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("SKINNY_EXECUTION_MODE", "wavefront")
    monkeypatch.setenv("SKINNY_BDPT_WALK", "eye_light")
    ns = _parser().parse_args([])
    assert ns.execution_mode == "wavefront"
    assert resolve_walk(ns.bdpt_walk) == "eye_light"


def test_megakernel_walk_alias_accepted_on_cli(monkeypatch):
    monkeypatch.delenv("SKINNY_BDPT_WALK", raising=False)
    # `--bdpt-walk` is a free string (not argparse choices) so the alias parses,
    # then resolve_walk maps it to fused.
    ns = _parser().parse_args(["--bdpt-walk", "megakernel"])
    assert resolve_walk(ns.bdpt_walk) == "fused"


# ── choices enforcement ──────────────────────────────────────────────

@pytest.mark.parametrize("flag,bad", [
    ("--integrator", "bidir"),
    ("--execution-mode", "fused"),  # fused is a walk, not an execution mode
])
def test_choice_flags_reject_bad(flag, bad):
    with pytest.raises(SystemExit):
        _parser().parse_args([flag, bad])


# ── --neural-handoff (online-training weight handoff backend) ─────────

def test_neural_handoff_default(monkeypatch):
    monkeypatch.delenv("SKINNY_NEURAL_HANDOFF", raising=False)
    ns = _parser().parse_args([])
    assert ns.neural_handoff == "file"


def test_neural_handoff_explicit():
    ns = _parser().parse_args(["--neural-handoff", "interop"])
    assert ns.neural_handoff == "interop"


def test_neural_handoff_env_fallback(monkeypatch):
    monkeypatch.setenv("SKINNY_NEURAL_HANDOFF", "interop")
    ns = _parser().parse_args([])
    assert ns.neural_handoff == "interop"


def test_neural_handoff_rejects_bad(monkeypatch):
    monkeypatch.delenv("SKINNY_NEURAL_HANDOFF", raising=False)
    with pytest.raises(SystemExit):
        _parser().parse_args(["--neural-handoff", "shared-mem"])


def test_neural_handoff_can_be_suppressed():
    p = _parser(neural_handoff=False)
    ns = p.parse_args([])
    assert not hasattr(ns, "neural_handoff")


# ── suppression kwargs ───────────────────────────────────────────────

def test_flags_can_be_suppressed():
    p = _parser(integrator=False)
    ns = p.parse_args(["--execution-mode", "wavefront"])
    assert not hasattr(ns, "integrator")
    assert ns.execution_mode == "wavefront"


def test_integrator_index_map():
    assert INTEGRATOR_INDEX == {"path": 0, "bdpt": 1}


# ── front-end parity ─────────────────────────────────────────────────

@pytest.mark.parametrize("module", [
    "app.py", "ui/qt/app.py", "web_app.py", "headless.py",
])
def test_every_frontend_uses_shared_helper(module):
    # Identical flags are guaranteed by every front-end deferring to the one
    # add_render_flags definition rather than rolling its own argparse block.
    src = (_SRC / module).read_text()
    assert "add_render_flags(" in src, f"{module} must call add_render_flags"
    # No front-end should re-declare the migrated flags inline.
    assert 'add_argument(\n        "--execution-mode"' not in src
    assert '"--bdpt-walk", choices=' not in src


def test_headless_parser_exposes_all_three_flags(monkeypatch):
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("SKINNY_BDPT_WALK", raising=False)
    from skinny.headless import _build_parser

    ns = _build_parser().parse_args(["scene.usd"])
    assert ns.integrator is None
    assert ns.execution_mode == "megakernel"
    assert resolve_walk(ns.bdpt_walk) == "fused"


# ── validate_render_flags: bdpt × neural/online incompatibility ───────

from skinny.cli_common import validate_render_flags  # noqa: E402


def _ns(**kw):
    base = dict(integrator=None, proposals=None, online_training=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_bdpt_plus_online_training_exits():
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(_ns(integrator="bdpt", online_training=True))
    assert "bdpt" in str(ei.value) and "--integrator path" in str(ei.value)


def test_bdpt_plus_neural_proposal_exits():
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(_ns(integrator="bdpt", proposals="bsdf,neural"))
    assert "bdpt" in str(ei.value)


def test_bdpt_without_neural_is_ok():
    # bdpt alone, and bdpt + a non-neural proposal, are fine.
    validate_render_flags(_ns(integrator="bdpt"))
    validate_render_flags(_ns(integrator="bdpt", proposals="bsdf,env"))


def test_path_plus_neural_is_ok():
    validate_render_flags(_ns(integrator="path", proposals="bsdf,neural",
                              online_training=True))


def test_default_integrator_not_flagged():
    # integrator=None (persisted/default path) must not trip the guard.
    validate_render_flags(_ns(integrator=None, online_training=True))


def test_missing_proposals_attr_ok():
    # Front-ends without --proposals (skinny-gui / skinny-web) pass a Namespace
    # with no `proposals` attribute; the guard must tolerate it.
    validate_render_flags(argparse.Namespace(integrator="bdpt", online_training=False))
