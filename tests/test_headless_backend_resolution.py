"""Hostless proof that the headless Python API resolves its backend.

The direct-API path (`HeadlessRenderer` with no `BringupPlan`) used to hard-code
`backend="vulkan"`: it never reached `backend_select.select_backend`, so it
ignored `SKINNY_BACKEND`, could not be given `auto` at all, and put an
Apple-Silicon caller on MoltenVK by default. Change `headless-backend-auto`
routes it through the shared selector.

No GPU here: `metal_available` is stubbed (so no device is probed) and
`BringupPlan.create` is intercepted (so no context is built). What is under test
is the resolution and its precedence, which is entirely host-independent.
"""

from __future__ import annotations

import pytest

from skinny import backend_select
from skinny.bringup import BringupPlan


class _StubContext:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The selector reads SKINNY_BACKEND; never inherit the caller's."""
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)


@pytest.fixture
def captured(monkeypatch):
    """Intercept `BringupPlan.create`; return the list of plans it received."""
    plans: list[BringupPlan] = []

    def create(self, **kw):
        plans.append(self)
        return _StubContext(), object()

    monkeypatch.setattr(BringupPlan, "create", create)
    return plans


def _metal(monkeypatch, available: bool):
    monkeypatch.setattr(
        backend_select, "metal_available",
        lambda: (True, "") if available else (False, "no Metal device here"),
    )


def _construct(captured, **kw) -> BringupPlan:
    from skinny.headless import HeadlessRenderer

    HeadlessRenderer(64, 64, **kw)
    assert len(captured) == 1, "expected exactly one plan.create call"
    return captured[0]


# ── the default: resolve, do not assume Vulkan ────────────────────────

def test_default_resolves_to_metal_where_a_metal_device_constructs(
        monkeypatch, captured):
    _metal(monkeypatch, True)
    assert _construct(captured).backend == "metal"


def test_default_falls_back_to_vulkan_without_a_metal_device(
        monkeypatch, captured):
    _metal(monkeypatch, False)
    assert _construct(captured).backend == "vulkan"


# ── precedence: argument > SKINNY_BACKEND > auto ───────────────────────

def test_environment_metal_is_honoured_by_the_default(monkeypatch, captured):
    """The discriminating direction: the pre-change code returned "vulkan" here
    no matter what SKINNY_BACKEND said. Asserting env=vulkan → vulkan alone
    would pass vacuously against the old hard-coded default."""
    _metal(monkeypatch, True)
    monkeypatch.setenv("SKINNY_BACKEND", "metal")
    assert _construct(captured).backend == "metal"


def test_environment_vulkan_is_honoured_by_the_default(monkeypatch, captured):
    """The default must be *unset*, not the literal "auto".

    `select_backend` reads `prefer or env or persisted or "auto"`, so passing
    the string "auto" as `prefer` would win over SKINNY_BACKEND. `None` is what
    lets the environment participate — the same thing argparse hands the four
    front-ends (`--backend … default=None`).
    """
    _metal(monkeypatch, True)          # Metal is available and still not chosen
    monkeypatch.setenv("SKINNY_BACKEND", "vulkan")
    assert _construct(captured).backend == "vulkan"


def test_explicit_argument_outranks_the_environment(monkeypatch, captured):
    _metal(monkeypatch, True)
    monkeypatch.setenv("SKINNY_BACKEND", "vulkan")
    assert _construct(captured, backend="metal").backend == "metal"


def test_explicit_auto_outranks_the_environment(monkeypatch, captured):
    """`backend="auto"` is an explicit choice, exactly like `--backend auto`."""
    _metal(monkeypatch, True)
    monkeypatch.setenv("SKINNY_BACKEND", "vulkan")
    assert _construct(captured, backend="auto").backend == "metal"


def test_explicit_vulkan_is_preserved(monkeypatch, captured):
    """The pre-change default, now opt-in — the documented migration."""
    _metal(monkeypatch, True)
    assert _construct(captured, backend="vulkan").backend == "vulkan"


# ── the token vocabulary ──────────────────────────────────────────────

def test_auto_is_resolved_before_the_context_factory_sees_it(
        monkeypatch, captured):
    """`auto` used to reach `make_context` unresolved and raise
    `unknown backend 'auto' (expected 'vulkan' or 'metal')`."""
    _metal(monkeypatch, False)
    assert _construct(captured, backend="auto").backend in ("metal", "vulkan")


def test_unknown_token_is_refused(monkeypatch, captured):
    from skinny.headless import HeadlessRenderer

    with pytest.raises(ValueError, match="unknown backend"):
        HeadlessRenderer(64, 64, backend="opengl")
    assert not captured, "no context may be constructed for a bad token"


def test_unavailable_explicit_metal_refuses_without_building_a_context(
        monkeypatch, captured):
    from skinny.headless import HeadlessRenderer

    _metal(monkeypatch, False)
    with pytest.raises(RuntimeError, match="no Metal device here"):
        HeadlessRenderer(64, 64, backend="metal")
    assert not captured


# ── persistence and the CLI path ──────────────────────────────────────

def test_no_persisted_setting_participates(monkeypatch, captured):
    """The direct API is non-interactive, like `skinny-render`: it must not
    offer a persisted backend to the selector."""
    seen = {}
    import skinny.headless as headless

    def spy(prefer=None, *, persisted=None):
        seen["prefer"], seen["persisted"] = prefer, persisted
        return "vulkan"

    monkeypatch.setattr(headless, "select_backend", spy)
    _construct(captured)
    assert seen == {"prefer": None, "persisted": None}


def test_a_given_plan_is_not_resolved_again(monkeypatch, captured):
    """`skinny-render`'s path: `plan_bringup` already ran `select_backend`."""
    import skinny.headless as headless

    def boom(*a, **kw):  # pragma: no cover — must never run
        raise AssertionError("select_backend re-ran on the plan path")

    monkeypatch.setattr(headless, "select_backend", boom)
    plan = BringupPlan(
        prog="skinny-render", backend="vulkan", execution_mode="megakernel",
        startup_integrator="path", spectral=False, bdpt_walk="fused",
        encoding=None, neural_config=None,
    )
    assert _construct(captured, plan=plan).backend == "vulkan"
    assert captured[0] is plan
