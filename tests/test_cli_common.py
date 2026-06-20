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
    assert INTEGRATOR_INDEX == {"path": 0, "bdpt": 1, "sppm": 2}


def test_integrator_choices_include_sppm():
    ns = _parser().parse_args(["--integrator", "sppm", "--execution-mode", "wavefront"])
    assert ns.integrator == "sppm"


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


# ── validate_render_flags: sppm requires wavefront ────────────────────

def test_sppm_plus_megakernel_exits():
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(_ns(integrator="sppm", execution_mode="megakernel"))
    msg = str(ei.value)
    assert "sppm" in msg.lower() and "wavefront" in msg


def test_sppm_default_execution_mode_exits():
    # `--integrator sppm` with no explicit --execution-mode defaults to megakernel,
    # which SPPM cannot run on → must be refused naming the fix.
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(_ns(integrator="sppm"))
    assert "wavefront" in str(ei.value)


def test_sppm_plus_wavefront_is_ok():
    validate_render_flags(_ns(integrator="sppm", execution_mode="wavefront"))


def test_non_sppm_not_flagged_by_sppm_gate():
    # path/bdpt under megakernel must not trip the sppm gate.
    validate_render_flags(_ns(integrator="path", execution_mode="megakernel"))
    validate_render_flags(_ns(integrator=None, execution_mode="megakernel"))


# ── --width / --height (render-area resolution flags) ────────────────

def test_resolution_defaults(monkeypatch):
    monkeypatch.delenv("SKINNY_WIDTH", raising=False)
    monkeypatch.delenv("SKINNY_HEIGHT", raising=False)
    ns = _parser().parse_args([])
    assert ns.width == 640
    assert ns.height == 480


def test_resolution_explicit(monkeypatch):
    monkeypatch.delenv("SKINNY_WIDTH", raising=False)
    monkeypatch.delenv("SKINNY_HEIGHT", raising=False)
    ns = _parser().parse_args(["--width", "800", "--height", "600"])
    assert ns.width == 800
    assert ns.height == 600


def test_resolution_env_fallback(monkeypatch):
    monkeypatch.setenv("SKINNY_WIDTH", "1024")
    monkeypatch.setenv("SKINNY_HEIGHT", "768")
    ns = _parser().parse_args([])
    assert ns.width == 1024
    assert ns.height == 768


def test_resolution_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("SKINNY_WIDTH", "1024")
    monkeypatch.setenv("SKINNY_HEIGHT", "768")
    ns = _parser().parse_args(["--width", "320", "--height", "240"])
    assert ns.width == 320
    assert ns.height == 240


def test_resolution_malformed_env_errors(monkeypatch):
    monkeypatch.setenv("SKINNY_WIDTH", "wide")
    with pytest.raises(SystemExit, match="SKINNY_WIDTH"):
        _parser()


def test_resolution_can_be_suppressed():
    # skinny-render / skinny-web opt out (resolution=False) so they keep / omit
    # their own width/height — the shared flags are absent.
    p = _parser(resolution=False)
    ns = p.parse_args([])
    assert not hasattr(ns, "width")
    assert not hasattr(ns, "height")


def test_resolution_in_help(monkeypatch):
    monkeypatch.delenv("SKINNY_WIDTH", raising=False)
    monkeypatch.delenv("SKINNY_HEIGHT", raising=False)
    help_text = _parser().format_help()
    assert "--width" in help_text
    assert "--height" in help_text


@pytest.mark.parametrize("flag", ["--width", "--height"])
def test_non_positive_resolution_rejected(flag):
    ns = _ns(**{flag.lstrip("-"): 0})
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(ns)
    assert flag in str(ei.value)


def test_negative_resolution_rejected():
    with pytest.raises(SystemExit):
        validate_render_flags(_ns(width=-1))
    with pytest.raises(SystemExit):
        validate_render_flags(_ns(height=-100))


def test_positive_resolution_ok():
    # A valid render area passes the guard.
    validate_render_flags(_ns(width=640, height=480))


def test_missing_resolution_attrs_ok():
    # Front-ends that suppress the flags (skinny-render / skinny-web) pass a
    # Namespace without width/height; the guard must tolerate it.
    validate_render_flags(argparse.Namespace(integrator=None))


def test_headless_keeps_own_resolution(monkeypatch):
    # skinny-render opts out of the shared flags and keeps its own 1024^2
    # default — no argparse conflict on parser construction.
    monkeypatch.delenv("SKINNY_WIDTH", raising=False)
    monkeypatch.delenv("SKINNY_HEIGHT", raising=False)
    from skinny.headless import _build_parser

    ns = _build_parser().parse_args(["scene.usd"])
    assert ns.width == 1024
    assert ns.height == 1024
