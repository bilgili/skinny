"""Unit tests for the shared render-selection CLI surface (`skinny.cli_common`).

These are pure-argparse / pure-function tests — no GPU, no window, no heavy
front-end imports — so they run anywhere the package imports.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from skinny.cli_common import (
    DEFAULT_EXECUTION_FOR_INTEGRATOR,
    INTEGRATOR_INDEX,
    WALK_CHOICES,
    add_render_flags,
    reject_sppm_without_wavefront,
    resolve_execution_mode,
    resolve_walk,
    startup_integrator_name,
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
    # execution-mode defaults to 'auto' (derives from the integrator; see
    # resolve_execution_mode) — mirroring --backend auto.
    assert ns.execution_mode == "auto"
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


@pytest.mark.parametrize("mode", ["auto", "megakernel", "wavefront"])
def test_execution_mode_accepts_auto_and_pins(mode):
    ns = _parser().parse_args(["--execution-mode", mode])
    assert ns.execution_mode == mode


# ── resolve_execution_mode (integrator → execution mode) ─────────────

def test_default_execution_map():
    assert DEFAULT_EXECUTION_FOR_INTEGRATOR == {
        "path": "megakernel", "bdpt": "megakernel", "sppm": "wavefront"}


@pytest.mark.parametrize("integrator,expected", [
    ("path", "megakernel"),
    ("bdpt", "megakernel"),
    ("sppm", "wavefront"),
])
def test_resolve_auto_derives_from_integrator(integrator, expected):
    assert resolve_execution_mode("auto", integrator) == expected


def test_resolve_auto_none_integrator_is_megakernel():
    # The persisted/default path (integrator=None) derives megakernel.
    assert resolve_execution_mode("auto", None) == "megakernel"


@pytest.mark.parametrize("integrator", ["path", "bdpt", "sppm"])
def test_resolve_explicit_mode_overrides_derivation(integrator):
    # An explicit megakernel/wavefront wins over the integrator-derived default.
    assert resolve_execution_mode("megakernel", integrator) == "megakernel"
    assert resolve_execution_mode("wavefront", integrator) == "wavefront"


def test_resolve_explicit_megakernel_wins_over_sppm():
    # sppm would derive wavefront, but an explicit megakernel pins megakernel
    # (validate_render_flags then rejects that impossible combo separately).
    assert resolve_execution_mode("megakernel", "sppm") == "megakernel"


def test_env_counts_as_explicit(monkeypatch):
    # SKINNY_EXECUTION_MODE=megakernel makes the flag default a concrete mode, so
    # even --integrator sppm resolves to megakernel (and is then rejected).
    monkeypatch.setenv("SKINNY_EXECUTION_MODE", "megakernel")
    ns = _parser().parse_args(["--integrator", "sppm"])
    assert ns.execution_mode == "megakernel"
    assert resolve_execution_mode(ns.execution_mode, ns.integrator) == "megakernel"


# ── startup_integrator_name (interactive persisted-integrator fallback) ──

def test_startup_integrator_cli_wins():
    assert startup_integrator_name("bdpt", 2) == "bdpt"


def test_startup_integrator_uses_persisted_index_when_no_cli():
    # Persisted integrator_index 2 == sppm → drives sppm → wavefront next launch.
    assert startup_integrator_name(None, 2) == "sppm"
    assert resolve_execution_mode("auto", startup_integrator_name(None, 2)) == "wavefront"


@pytest.mark.parametrize("bad", [None, "nope", 99, {}])
def test_startup_integrator_falls_back_to_path(bad):
    # Missing / out-of-range / malformed persisted index → 'path'.
    assert startup_integrator_name(None, bad) == "path"


# ── reject_sppm_without_wavefront (effective-integrator guard) ────────

def test_reject_sppm_megakernel_raises():
    with pytest.raises(SystemExit) as ei:
        reject_sppm_without_wavefront("sppm", "megakernel")
    assert "sppm" in str(ei.value).lower() and "wavefront" in str(ei.value)


def test_reject_sppm_none_mode_raises():
    # A missing resolved mode defaults to megakernel → still refused.
    with pytest.raises(SystemExit):
        reject_sppm_without_wavefront("sppm", None)


@pytest.mark.parametrize("integrator,mode", [
    ("sppm", "wavefront"),
    ("path", "megakernel"),
    ("bdpt", "megakernel"),
    (None, "megakernel"),
])
def test_reject_sppm_noop_on_valid(integrator, mode):
    reject_sppm_without_wavefront(integrator, mode)  # no raise


def test_interactive_persisted_sppm_plus_explicit_megakernel_refused():
    # The escaped combo: no CLI --integrator, persisted integrator_index 2 (sppm),
    # explicit --execution-mode megakernel. validate_render_flags (CLI-keyed) can't
    # see it; the interactive front-ends re-check the effective startup integrator.
    startup = startup_integrator_name(None, 2)          # persisted sppm
    mode = resolve_execution_mode("megakernel", startup)  # explicit wins → megakernel
    assert startup == "sppm" and mode == "megakernel"
    with pytest.raises(SystemExit):
        reject_sppm_without_wavefront(startup, mode)


def test_interactive_persisted_sppm_auto_is_ok():
    # Same persisted sppm, but auto (no explicit mode) → wavefront → no error.
    startup = startup_integrator_name(None, 2)
    mode = resolve_execution_mode("auto", startup)
    assert mode == "wavefront"
    reject_sppm_without_wavefront(startup, mode)  # no raise


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


def test_execution_mode_choices_and_default_in_help(monkeypatch):
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    help_text = _parser().format_help()
    # All three choices advertised, and 'auto' is the default the front-ends share.
    for choice in ("auto", "megakernel", "wavefront"):
        assert choice in help_text
    assert _parser().parse_args([]).execution_mode == "auto"


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
    assert ns.execution_mode == "auto"
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


# ── validate_render_flags: sppm has no megakernel path ────────────────
# The execution mode is resolved (auto → per-integrator) BEFORE validation on
# every front-end, so validate sees a concrete mode. sppm auto-derives wavefront,
# so the guard trips only on an explicit megakernel override.

def test_sppm_plus_explicit_megakernel_exits():
    with pytest.raises(SystemExit) as ei:
        validate_render_flags(_ns(integrator="sppm", execution_mode="megakernel"))
    msg = str(ei.value)
    assert "sppm" in msg.lower() and "wavefront" in msg


def test_sppm_auto_resolves_wavefront_then_validates_ok():
    # `--integrator sppm` with no explicit mode: resolution turns auto → wavefront
    # (before validate on every front-end), so validation passes — no error.
    mode = resolve_execution_mode("auto", "sppm")
    assert mode == "wavefront"
    validate_render_flags(_ns(integrator="sppm", execution_mode=mode))


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


# ── --spectral flag + reject_spectral_unsupported ─────────────────────

def test_spectral_flag_default_off(monkeypatch):
    monkeypatch.delenv("SKINNY_SPECTRAL", raising=False)
    assert _parser().parse_args([]).spectral is False


def test_spectral_flag_set():
    assert _parser().parse_args(["--spectral"]).spectral is True


def test_spectral_env_fallback(monkeypatch):
    monkeypatch.setenv("SKINNY_SPECTRAL", "1")
    assert _parser().parse_args([]).spectral is True


def test_spectral_flag_suppressible():
    # A front-end may omit the flag; the parser then has no --spectral.
    ns = _parser(spectral=False).parse_args([])
    assert not hasattr(ns, "spectral")


def test_reject_spectral_noop_when_off():
    # Not spectral ⇒ never raises, whatever the other axes.
    from skinny.cli_common import reject_spectral_unsupported

    reject_spectral_unsupported(False, "bdpt", "wavefront", "neural", "restir-di")


def test_reject_spectral_not_implemented_refused(monkeypatch):
    # If the capability gate is ever forced back off, an in-envelope --spectral
    # must refuse (don't silently render RGB). Message names it "not yet
    # implemented". (SPECTRAL_IMPLEMENTED ships True; this exercises the guard.)
    from skinny import spectral_capability
    from skinny.cli_common import reject_spectral_unsupported

    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", False)
    with pytest.raises(SystemExit) as ei:
        reject_spectral_unsupported(True, "path", "megakernel", None, None)
    assert "not yet implemented" in str(ei.value)


def test_reject_spectral_envelope_ok_when_wired(monkeypatch):
    # With the capability flag flipped, an in-envelope combo is allowed.
    from skinny import spectral_capability
    from skinny.cli_common import reject_spectral_unsupported

    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", True)
    reject_spectral_unsupported(True, "path", "megakernel", None, None)
    reject_spectral_unsupported(True, "path", "megakernel", "", "none")
    # BDPT is in the megakernel spectral envelope (change spectral-bdpt-megakernel).
    reject_spectral_unsupported(True, "bdpt", "megakernel", None, None)


def test_reject_spectral_sppm_raises():
    # SPPM has no megakernel path, so spectral SPPM is refused at startup.
    from skinny.cli_common import reject_spectral_unsupported

    with pytest.raises(SystemExit):
        reject_spectral_unsupported(True, "sppm", "megakernel", None, None)


def test_reject_spectral_wavefront_raises():
    from skinny.cli_common import reject_spectral_unsupported

    with pytest.raises(SystemExit):
        reject_spectral_unsupported(True, "path", "wavefront", None, None)


def test_reject_spectral_neural_raises():
    from skinny.cli_common import reject_spectral_unsupported

    with pytest.raises(SystemExit):
        reject_spectral_unsupported(True, "path", "megakernel", "bsdf,neural", None)


def test_reject_spectral_reuse_raises():
    from skinny.cli_common import reject_spectral_unsupported

    with pytest.raises(SystemExit):
        reject_spectral_unsupported(True, "path", "megakernel", None, "restir-di")


def test_spectral_auto_execution_mode_allowed(monkeypatch):
    # --spectral with path + auto execution mode resolves to megakernel; with the
    # transport wired (flag flipped) the combo is allowed end-to-end.
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    from skinny import spectral_capability
    from skinny.cli_common import reject_spectral_unsupported

    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", True)
    ns = _parser().parse_args(["--spectral", "--integrator", "path"])
    mode = resolve_execution_mode(ns.execution_mode, ns.integrator or "path")
    assert mode == "megakernel"
    reject_spectral_unsupported(ns.spectral, ns.integrator or "path", mode,
                                getattr(ns, "proposals", None), getattr(ns, "reuse", None))


def test_spectral_flag_refused_when_gate_off_end_to_end(monkeypatch):
    # With the capability gate forced off, a plain in-envelope `--spectral` is
    # refused end-to-end (guard against a silent RGB no-op). Ships True.
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    from skinny import spectral_capability
    from skinny.cli_common import reject_spectral_unsupported

    monkeypatch.setattr(spectral_capability, "SPECTRAL_IMPLEMENTED", False)
    ns = _parser().parse_args(["--spectral"])
    mode = resolve_execution_mode(ns.execution_mode, ns.integrator or "path")
    with pytest.raises(SystemExit) as ei:
        reject_spectral_unsupported(ns.spectral, ns.integrator or "path", mode,
                                    getattr(ns, "proposals", None), getattr(ns, "reuse", None))
    assert "not yet implemented" in str(ei.value)


def test_spectral_flag_accepted_end_to_end():
    # The shipping default: a plain in-envelope `--spectral` is accepted (no exit)
    # now that SPECTRAL_IMPLEMENTED is True.
    import os
    os.environ.pop("SKINNY_EXECUTION_MODE", None)
    from skinny.cli_common import reject_spectral_unsupported

    ns = _parser().parse_args(["--spectral"])
    mode = resolve_execution_mode(ns.execution_mode, ns.integrator or "path")
    reject_spectral_unsupported(ns.spectral, ns.integrator or "path", mode,
                                getattr(ns, "proposals", None), getattr(ns, "reuse", None))


def test_spectral_explicit_wavefront_refused_end_to_end(monkeypatch):
    monkeypatch.delenv("SKINNY_EXECUTION_MODE", raising=False)
    from skinny.cli_common import reject_spectral_unsupported

    ns = _parser().parse_args(["--spectral", "--execution-mode", "wavefront"])
    mode = resolve_execution_mode(ns.execution_mode, ns.integrator or "path")
    with pytest.raises(SystemExit):
        reject_spectral_unsupported(ns.spectral, ns.integrator or "path", mode,
                                    getattr(ns, "proposals", None), getattr(ns, "reuse", None))


def test_all_frontends_wire_the_spectral_gate():
    # Every front-end that exposes --spectral (via add_render_flags) must call
    # reject_spectral_unsupported at startup, else --spectral silently no-ops to
    # RGB on that front-end (regression guard: skinny-gui once bypassed it).
    frontends = [
        _SRC / "app.py",
        _SRC / "headless.py",
        _SRC / "web_app.py",
        _SRC / "ui" / "qt" / "app.py",
    ]
    for fe in frontends:
        src = fe.read_text()
        assert "add_render_flags(" in src, f"{fe.name}: expected to expose the shared flags"
        assert "reject_spectral_unsupported(" in src, (
            f"{fe.name}: exposes --spectral but never calls reject_spectral_unsupported — "
            f"a --spectral run would silently render RGB there"
        )
