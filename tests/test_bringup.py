"""Hostless proof that the shared bring-up sequence is refusal-equivalent to the
four hand-copied front-end sequences it replaces (change
``frontend-bringup-builder``).

No GPU, no ``vulkan`` import: :meth:`BringupPlan.create` takes an injectable
context factory *and* renderer factory, so the construction stage is exercised
against stubs.

The fixture is the pair of ``_legacy_*`` functions below — the two guard orders
transcribed verbatim from the pre-refactor front-ends (``app.py`` /
``ui/qt/app.py``: validate → resolve → rejects; ``headless.py`` /
``web_app.py``: resolve → validate). They are the baseline the canonical order
is diffed against across the whole guard matrix, so a behaviour change on
either side fails here rather than at a user's terminal. Note the interactive
order fed the *unresolved* string ``"auto"`` into ``validate_render_flags``,
where ``_envelope_mode`` maps it to ``"wavefront"`` — the pre-resolution
megakernel check was a silent no-op and the explicit ``reject_*`` re-checks did
the real work; ``MODES`` therefore covers ``"auto"`` explicitly.

``REFUSAL_MESSAGES`` pins the exact strings on top of the differential check, so
a reword that changes both sides at once is still caught.
"""

from __future__ import annotations

import argparse
import itertools

import pytest

from skinny import backend_select
from skinny.bringup import BringupPlan, plan_bringup
from skinny.cli_common import (
    INTEGRATOR_INDEX,
    add_render_flags,
    reject_mcp_unsupported,
    reject_mlt_unsupported,
    reject_spectral_unsupported,
    reject_sppm_without_wavefront,
    resolve_execution_mode,
    startup_integrator_name,
    validate_render_flags,
)


# ── the pre-refactor sequences, transcribed ──────────────────────────

def _legacy_interactive(args, persisted, prog):
    """``app.py`` / ``ui/qt/app.py`` main(), pre-refactor: validate FIRST (with
    the still-unresolved execution mode), then load settings, resolve, and
    re-check the persisted-integrator cases explicitly."""
    validate_render_flags(args)
    startup = startup_integrator_name(
        args.integrator, persisted.get("params", {}).get("integrator_index"))
    args.execution_mode = resolve_execution_mode(args.execution_mode, startup)
    reject_sppm_without_wavefront(startup, args.execution_mode)
    reject_mlt_unsupported(
        startup, args.execution_mode,
        spectral=bool(getattr(args, "spectral", False)),
        proposals=getattr(args, "proposals", None),
        reuse=getattr(args, "reuse", None),
        online_training=bool(getattr(args, "online_training", False)))
    reject_spectral_unsupported(
        getattr(args, "spectral", False), startup, args.execution_mode,
        getattr(args, "proposals", None), getattr(args, "reuse", None))
    reject_mcp_unsupported(bool(getattr(args, "mcp", False)))
    try:
        backend = backend_select.select_backend(
            args.backend, persisted=persisted.get("backend"))
    except RuntimeError as exc:
        raise SystemExit(f"{prog}: {exc}")
    return backend, args.execution_mode, startup


def _legacy_noninteractive(args, persisted, prog):
    """``headless.py`` / ``web_app.py`` main(), pre-refactor: resolve first,
    then validate; only the spectral guard, no persistence."""
    startup = args.integrator or "path"
    args.execution_mode = resolve_execution_mode(args.execution_mode, startup)
    validate_render_flags(args)
    reject_spectral_unsupported(
        getattr(args, "spectral", False), startup, args.execution_mode,
        getattr(args, "proposals", None), getattr(args, "reuse", None))
    try:
        backend = backend_select.select_backend(args.backend)
    except RuntimeError as exc:
        raise SystemExit(f"{prog}: {exc}")
    return backend, args.execution_mode, startup


# Flag-set knobs per front-end, mirroring each main()'s add_render_flags call.
FRONTENDS = {
    "skinny": dict(flags={}, legacy=_legacy_interactive, persists=True),
    "skinny-gui": dict(flags={"proposals": False}, legacy=_legacy_interactive,
                       persists=True),
    "skinny-render": dict(flags={"resolution": False, "mcp": False},
                          legacy=_legacy_noninteractive, persists=False,
                          own_resolution=True),
    "skinny-web": dict(flags={"proposals": False, "resolution": False, "mcp": False},
                       legacy=_legacy_noninteractive, persists=False),
}


def _parser(prog: str, cfg: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    add_render_flags(p, **cfg["flags"])
    if cfg.get("own_resolution"):
        # skinny-render owns its own --width/--height (resolution=False above).
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
    return p


@pytest.fixture(autouse=True)
def _hostless_env(monkeypatch):
    """Keep the sweep off the GPU and off the developer's environment: an
    explicit ``--backend metal`` refuses with a stub reason, ``auto`` falls back
    to vulkan, and no ``SKINNY_*`` variable leaks in."""
    for key in [k for k in __import__("os").environ if k.startswith("SKINNY_")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        backend_select, "metal_available", lambda: (False, "stubbed unavailable"))
    monkeypatch.setattr("sys.argv", ["pytest"])


# ── the guard matrix ─────────────────────────────────────────────────

INTEGRATORS = [None, "path", "bdpt", "sppm", "mlt"]
MODES = [None, "auto", "megakernel", "wavefront"]
SPECTRAL = [False, True]
PROPOSALS = [None, "bsdf", "bsdf,env", "bsdf,neural", "env", "neural"]
REUSE = [None, "none"]
ONLINE = [False, True]
PERSISTED = {
    "none": {},
    "sppm": {"params": {"integrator_index": INTEGRATOR_INDEX["sppm"]}},
    "mlt": {"params": {"integrator_index": INTEGRATOR_INDEX["mlt"]}},
    "backend-metal": {"backend": "metal"},
    "encoding-E1": {"encoding": "E1"},
}


def _argv(cfg, integrator, mode, spectral, proposals, reuse, online):
    """Build an argv for this front-end, or ``None`` when the combination is not
    expressible there (a suppressed flag)."""
    argv = []
    if integrator:
        argv += ["--integrator", integrator]
    if mode:
        argv += ["--execution-mode", mode]
    if spectral:
        argv += ["--spectral"]
    if proposals:
        if cfg["flags"].get("proposals", True) is False:
            return None
        argv += ["--proposals", proposals]
    if reuse:
        argv += ["--reuse", reuse]
    if online:
        argv += ["--online-training"]
    return argv


def _outcome(fn, parser, argv, persisted, prog):
    """Run a sequence over a freshly parsed Namespace; return a comparable
    ``("refuse", message)`` or ``("accept", backend, mode, startup)``."""
    args = parser.parse_args(argv)
    try:
        backend, mode, startup = fn(args, persisted, prog)
    except SystemExit as exc:
        return ("refuse", str(exc))
    return ("accept", backend, mode, startup)


def _plan_outcome(parser, argv, persisted, prog, persists):
    args = parser.parse_args(argv)
    try:
        plan = plan_bringup(args, prog, persisted=persisted if persists else None)
    except SystemExit as exc:
        return ("refuse", str(exc))
    return ("accept", plan.backend, plan.execution_mode, plan.startup_integrator)


@pytest.mark.parametrize("prog", sorted(FRONTENDS))
def test_canonical_order_matches_the_legacy_sequence(prog):
    """Every guard-matrix combination yields the same accept/refuse outcome —
    and the same message — as the front-end's pre-refactor sequence."""
    cfg = FRONTENDS[prog]
    parser = _parser(prog, cfg)
    persisted_cases = PERSISTED if cfg["persists"] else {"none": {}}
    checked = 0
    for combo in itertools.product(
            INTEGRATORS, MODES, SPECTRAL, PROPOSALS, REUSE, ONLINE):
        argv = _argv(cfg, *combo)
        if argv is None:
            continue
        for pname, persisted in persisted_cases.items():
            legacy = _outcome(cfg["legacy"], parser, argv, persisted, prog)
            actual = _plan_outcome(parser, argv, persisted, prog, cfg["persists"])
            assert actual == legacy, f"{prog} {argv} persisted={pname}"
            checked += 1
    assert checked > 100, f"{prog}: matrix collapsed to {checked} cases"


@pytest.mark.parametrize("prog", sorted(FRONTENDS))
@pytest.mark.parametrize("backend_flag", [None, "vulkan", "metal", "auto"])
def test_backend_selection_matches_the_legacy_sequence(prog, backend_flag):
    """The backend axis, including the ``{prog}:``-prefixed refusal for an
    explicit but unavailable ``--backend metal``."""
    cfg = FRONTENDS[prog]
    parser = _parser(prog, cfg)
    argv = ["--backend", backend_flag] if backend_flag else []
    persisted_cases = PERSISTED if cfg["persists"] else {"none": {}}
    for persisted in persisted_cases.values():
        legacy = _outcome(cfg["legacy"], parser, argv, persisted, prog)
        actual = _plan_outcome(parser, argv, persisted, prog, cfg["persists"])
        assert actual == legacy


@pytest.mark.parametrize("prog", ["skinny", "skinny-gui"])
def test_unavailable_metal_refuses_with_the_frontend_prefix(prog):
    parser = _parser(prog, FRONTENDS[prog])
    with pytest.raises(SystemExit) as excinfo:
        plan_bringup(parser.parse_args(["--backend", "metal"]), prog, persisted={})
    assert str(excinfo.value) == (
        f"{prog}: --backend metal requested but native Metal is unavailable: "
        "stubbed unavailable. Use --backend vulkan (or 'auto')."
    )


# Exact refusal strings, pinned **in full** as independent goldens. The
# differential check above compares the canonical order against the transcribed
# legacy orders — both of which call the same live guard functions, so a reword
# would move both sides together and pass. These literals are the second side
# of that: they fail on any change to the user-visible text.
#
# Note the guards print a fixed `skinny:` prefix on *every* front-end; only
# select_backend's failure carries the caller's `{prog}`. That asymmetry is
# pre-existing behaviour this change deliberately preserved (rewording it would
# break users' scripts and greps), so it is pinned here rather than fixed.
REFUSAL_MESSAGES = {
    "sppm-megakernel": (
        ["--integrator", "sppm", "--execution-mode", "megakernel"],
        "skinny: --integrator sppm has no megakernel path — SPPM (Stochastic "
        "Progressive Photon Mapping) shares a global visible-point / photon-grid "
        "structure across pixels. Drop the explicit --execution-mode megakernel "
        "(sppm auto-selects wavefront) or pass --execution-mode wavefront.",
    ),
    "mlt-megakernel": (
        ["--integrator", "mlt", "--execution-mode", "megakernel"],
        "skinny: --integrator mlt has no megakernel path — MLT mutates Markov "
        "chains through the staged wavefront BDPT kernels and shares "
        "bootstrap/normalization state across pixels. Drop the explicit "
        "--execution-mode megakernel (mlt auto-selects wavefront) or pass "
        "--execution-mode wavefront.",
    ),
    "bdpt-neural": (
        ["--integrator", "bdpt", "--proposals", "bsdf,neural"],
        "skinny: the neural proposal (--proposals …,neural) is incompatible with "
        "--integrator bdpt — BDPT does not consume the neural directional "
        "proposal (it samples directions with native BSDF sampling, on every "
        "backend and execution mode). Use --integrator path for neural guiding / "
        "online training.",
    ),
    "bdpt-online-training": (
        ["--integrator", "bdpt", "--online-training"],
        "skinny: --online-training is incompatible with --integrator bdpt — BDPT "
        "does not consume the neural directional proposal (it samples directions "
        "with native BSDF sampling, on every backend and execution mode). Use "
        "--integrator path for neural guiding / online training.",
    ),
    "mlt-online-training": (
        ["--integrator", "mlt", "--online-training"],
        "skinny: --online-training is incompatible with --integrator mlt — MLT "
        "does not consume the neural directional proposal the training loop "
        "exists to improve.",
    ),
    "mlt-proposals": (
        ["--integrator", "mlt", "--proposals", "neural"],
        "skinny: --integrator mlt supports only the BSDF proposal (got "
        "--proposals neural) — a non-BSDF directional proposal inside a "
        "Markov-chain mutation would change the target function MLT normalizes "
        "against.",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSAL_MESSAGES))
def test_refusal_messages_are_verbatim(case):
    argv, expected = REFUSAL_MESSAGES[case]
    parser = _parser("skinny", FRONTENDS["skinny"])
    with pytest.raises(SystemExit) as excinfo:
        plan_bringup(parser.parse_args(argv), "skinny", persisted={})
    assert str(excinfo.value) == expected


def test_spectral_neural_refusal_message_is_verbatim():
    """Kept separate: the spectral guard interpolates the proposal token, so the
    message is checked for both a bare-neural and a mixed-proposal spelling."""
    parser = _parser("skinny", FRONTENDS["skinny"])
    for token in ("neural", "bsdf,neural"):
        with pytest.raises(SystemExit) as excinfo:
            plan_bringup(parser.parse_args(["--spectral", "--proposals", token]),
                         "skinny", persisted={})
        assert str(excinfo.value).startswith(
            f"skinny: --spectral supports only the analytic BSDF/environment "
            f"proposals (got --proposals {token}). The neural")


# ── the MCP axis ─────────────────────────────────────────────────────

@pytest.mark.parametrize("prog", ["skinny", "skinny-gui"])
def test_mcp_guard_runs_for_every_frontend_that_exposes_the_flag(prog, monkeypatch):
    """`--mcp` is refused when the optional server dependency is missing. Before
    this change only the interactive pair called this guard at all; it is now
    part of the shared sequence, so it must fire from there."""
    import builtins

    real_import = builtins.__import__

    def no_mcp(name, *a, **kw):
        if name.startswith("mcp"):
            raise ImportError("stubbed missing")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_mcp)
    parser = _parser(prog, FRONTENDS[prog])
    with pytest.raises(SystemExit) as excinfo:
        plan_bringup(parser.parse_args(["--mcp"]), prog, persisted={})
    assert str(excinfo.value).startswith("--mcp needs the optional MCP server dependency.")


def test_mcp_guard_is_a_noop_without_the_flag():
    """The non-interactive front-ends suppress `--mcp` entirely, so the shared
    sequence must tolerate a Namespace with no `mcp` attribute."""
    parser = _parser("skinny-web", FRONTENDS["skinny-web"])
    args = parser.parse_args([])
    assert not hasattr(args, "mcp")
    assert plan_bringup(args, "skinny-web").backend == "vulkan"


# ── refusals happen before the GPU is touched ────────────────────────

def test_no_backend_probe_when_a_guard_refuses(monkeypatch):
    """Every flag-level refusal must land before `select_backend` probes for a
    Metal device — constructing one is the expensive, machine-affecting step."""
    probed = []
    monkeypatch.setattr(backend_select, "metal_available",
                        lambda: (probed.append(1), (False, "stub"))[1])
    parser = _parser("skinny", FRONTENDS["skinny"])
    with pytest.raises(SystemExit):
        plan_bringup(parser.parse_args(
            ["--integrator", "sppm", "--execution-mode", "megakernel"]),
            "skinny", persisted={})
    assert probed == []


def test_bad_bdpt_walk_is_rejected_before_the_backend_probe(monkeypatch):
    """`--bdpt-walk` has no argparse `choices=` and takes its default from
    SKINNY_BDPT_WALK, which argparse never validates — so a bad value reaches
    the plan. It must be rejected before the GPU probe."""
    probed = []
    monkeypatch.setattr(backend_select, "metal_available",
                        lambda: (probed.append(1), (False, "stub"))[1])
    monkeypatch.setenv("SKINNY_BDPT_WALK", "not-a-walk")
    parser = _parser("skinny", FRONTENDS["skinny"])
    with pytest.raises(ValueError, match="unknown bdpt walk"):
        plan_bringup(parser.parse_args([]), "skinny", persisted={})
    assert probed == []


def test_deprecated_bdpt_walk_alias_still_normalizes():
    parser = _parser("skinny", FRONTENDS["skinny"])
    plan = plan_bringup(parser.parse_args(["--bdpt-walk", "megakernel"]),
                        "skinny", persisted={})
    assert plan.bdpt_walk == "fused"


# ── persisted precedence ─────────────────────────────────────────────

def test_persisted_sppm_under_forced_megakernel_is_refused():
    """The interactive case ``validate_render_flags`` cannot see: no
    ``--integrator``, sppm from the persisted settings, megakernel forced."""
    parser = _parser("skinny", FRONTENDS["skinny"])
    args = parser.parse_args(["--execution-mode", "megakernel"])
    with pytest.raises(SystemExit, match="sppm has no megakernel path"):
        plan_bringup(args, "skinny", persisted=PERSISTED["sppm"])


def test_persisted_integrator_is_ignored_without_a_settings_dict():
    """``skinny-render`` / ``skinny-web`` stay persistence-free: the same
    settings dict must not reach resolution when it is not offered."""
    parser = _parser("skinny-render", FRONTENDS["skinny-render"])
    plan = plan_bringup(
        parser.parse_args(["--execution-mode", "megakernel"]), "skinny-render")
    assert plan.startup_integrator == "path"
    assert plan.execution_mode == "megakernel"


def test_persisted_backend_feeds_selection_only_when_offered(monkeypatch):
    parser = _parser("skinny", FRONTENDS["skinny"])
    monkeypatch.setattr(backend_select, "metal_available", lambda: (True, ""))
    with_persist = plan_bringup(parser.parse_args([]), "skinny",
                                persisted={"backend": "vulkan"})
    without = plan_bringup(parser.parse_args([]), "skinny-web")
    assert with_persist.backend == "vulkan"   # persisted wins over auto
    assert without.backend == "metal"         # auto, persistence not offered


def test_flag_beats_env_beats_persisted(monkeypatch):
    parser = _parser("skinny", FRONTENDS["skinny"])
    settings = {"backend": "metal"}
    assert plan_bringup(parser.parse_args(["--backend", "vulkan"]), "skinny",
                        persisted=settings).backend == "vulkan"
    monkeypatch.setenv("SKINNY_BACKEND", "vulkan")
    assert plan_bringup(parser.parse_args([]), "skinny",
                        persisted=settings).backend == "vulkan"


def test_persisted_encoding_restores_only_when_offered(monkeypatch):
    parser = _parser("skinny", FRONTENDS["skinny"])
    settings = {"encoding": "E1"}
    assert plan_bringup(parser.parse_args([]), "skinny",
                        persisted=settings).encoding == "E1"
    # Not offered → the flag default survives, the persisted E1 never applies.
    assert plan_bringup(parser.parse_args([]), "skinny-web").encoding == "E0"
    # An explicit --encoding wins over the persisted value.
    monkeypatch.setattr("sys.argv", ["skinny", "--encoding", "E3"])
    assert plan_bringup(parser.parse_args(["--encoding", "E3"]), "skinny",
                        persisted=settings).encoding == "E3"


def test_default_encoding_keeps_the_renderer_default_neural_config():
    """E0 (the default) must stay ``None`` so the shipped neural ``.spv`` is
    reused byte-identically."""
    parser = _parser("skinny", FRONTENDS["skinny"])
    assert plan_bringup(parser.parse_args([]), "skinny", persisted={}).neural_config is None
    plan = plan_bringup(parser.parse_args([]), "skinny", persisted={"encoding": "E1"})
    assert plan.neural_config is not None
    assert plan.neural_config.encoding.value == "E1"


# ── the create stage ─────────────────────────────────────────────────

class _StubContext:
    def __init__(self, **kw):
        self.kw = kw
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _stub_plan(**overrides) -> BringupPlan:
    fields = dict(prog="skinny", backend="vulkan", execution_mode="wavefront",
                  startup_integrator="path", spectral=True, bdpt_walk="eye",
                  encoding="E1", neural_config=object())
    fields.update(overrides)
    return BringupPlan(**fields)


def test_create_passes_plan_fields_and_forwards_kwargs():
    seen = {}

    def context_factory(backend, **kw):
        seen["ctx"] = dict(backend=backend, **kw)
        return _StubContext()

    def renderer_factory(**kw):
        seen["renderer"] = kw
        return "renderer"

    plan = _stub_plan()
    ctx, renderer = plan.create(
        window="win", width=320, height=240, gpu_preference="discrete",
        context_factory=context_factory, renderer_factory=renderer_factory,
        usd_scene_path="/scene.usda", use_usd_mtlx_plugin=True,
        neural_handoff="shared",
    )
    assert renderer == "renderer"
    assert seen["ctx"] == dict(backend="vulkan", window="win", width=320,
                               height=240, gpu_preference="discrete")
    # Plan-carried, guard-vetted fields come from the plan …
    assert seen["renderer"]["vk_ctx"] is ctx
    assert seen["renderer"]["execution_mode"] == "wavefront"
    assert seen["renderer"]["bdpt_walk"] == "eye"
    assert seen["renderer"]["spectral"] is True
    assert seen["renderer"]["neural_config"] is plan.neural_config
    # … front-end constructor inputs are forwarded verbatim.
    assert seen["renderer"]["usd_scene_path"] == "/scene.usda"
    assert seen["renderer"]["use_usd_mtlx_plugin"] is True
    assert seen["renderer"]["neural_handoff"] == "shared"


@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_create_destroys_the_context_when_the_renderer_raises(exc):
    """Including the BaseException cases: renderer construction compiles
    shaders and runs for seconds, so a Ctrl-C inside it is ordinary — and a
    leaked Metal context wedges GPU capacity until reboot."""
    ctx = _StubContext()

    def boom(**kw):
        raise exc("renderer exploded")

    with pytest.raises(exc):
        _stub_plan().create(context_factory=lambda *a, **k: ctx,
                            renderer_factory=boom)
    assert ctx.destroyed


def test_create_defaults_are_windowless_720p():
    seen = {}
    _stub_plan().create(
        context_factory=lambda backend, **kw: seen.update(kw) or _StubContext(),
        renderer_factory=lambda **kw: None)
    assert seen == dict(window=None, width=1280, height=720, gpu_preference=None)


def test_plan_is_frozen():
    with pytest.raises(Exception):
        _stub_plan().backend = "metal"
