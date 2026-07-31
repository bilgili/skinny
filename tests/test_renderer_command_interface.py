"""Hostless tests for the one command interface (change renderer-command-interface).

Every front-end that owns a renderer drives it the same way: post a command,
drain it on the owning thread. These tests assert that shape against a stub
renderer, with no GPU device present — which is the point. The web session's
mutation path had no coverage precisely because exercising it used to mean
constructing a real context.

Four front-ends, one interface:

* ``skinny-web`` — :class:`~skinny.web_app.SkinnySession`, background render
  thread, commands posted from the IOLoop and Bokeh threads;
* ``skinny-gui`` — :class:`~skinny.render_session.QtRendererProxy`;
* the shared web sidebar — :class:`~skinny.render_session.MarshalledRenderer`;
* ``skinny-render`` — :class:`~skinny.headless.HeadlessRenderer`, which posts
  and drains synchronously.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from concurrent.futures import Future
from pathlib import Path

import pytest

from skinny.render_session import (
    MarshalledRenderer,
    QtRendererProxy,
    RenderCommandQueue,
)
from skinny.web_app import SkinnySession

SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


class _StubCamera:
    def __init__(self) -> None:
        self.log: list[tuple] = []

    def orbit(self, dx, dy) -> None:
        self.log.append(("orbit", dx, dy))

    def pan(self, dx, dy) -> None:
        self.log.append(("pan", dx, dy))

    def zoom(self, d) -> None:
        self.log.append(("zoom", d))

    def move(self, f, r, u, dt) -> None:
        self.log.append(("move", f, r, u, dt))


class _StubRenderer:
    """Records every mutation, so a test can assert both *that* and *when*."""

    def __init__(self) -> None:
        self.camera = _StubCamera()
        self.camera_mode = "orbit"
        self.mtlx_overrides: dict[str, float] = {}
        self.show_hud = True
        self.show_focus_overlay = False
        self.lens_vignette_debug = False
        self._material_version = 0
        self.log: list[tuple] = []

    def autofocus_at_pixel(self, x, y) -> None:
        self.log.append(("autofocus", x, y))

    def toggle_camera_mode(self) -> None:
        self.log.append(("toggle_camera",))

    def reset_camera(self) -> None:
        self.log.append(("reset_camera",))

    def save_screenshot(self, buf, fmt) -> None:
        self.log.append(("screenshot", fmt))
        buf.write(b"PNGBYTES")

    def load_model_from_path(self, path) -> None:
        self.log.append(("load_model", str(path)))

    def load_environment_from_path(self, path) -> None:
        self.log.append(("load_hdr", str(path)))

    def toggle_material_furnace(self, index, enabled) -> None:
        self.log.append(("furnace", index, enabled))


class _StubEncoder:
    is_h264 = True

    def __init__(self) -> None:
        self.keyframes = 0

    def force_keyframe(self) -> None:
        self.keyframes += 1


@pytest.fixture
def session():
    """A session with no GPU behind it. `__init__` builds only the queue, lock
    and frame queue; `initialize()` is what needs a device."""
    s = SkinnySession("rci-test")
    s.renderer = _StubRenderer()
    s.encoder = _StubEncoder()
    try:
        yield s
    finally:
        SkinnySession._active.pop(s.session_id, None)


def _drain(session) -> None:
    session._commands.run_pending(session.renderer)


# ── skinny-web: every mutation path posts ────────────────────────────────


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda s: s.handle_camera("orbit", {"dx": 3, "dy": 4}),
         ("orbit", 3.0, 4.0)),
        (lambda s: s.handle_camera("pan", {"dx": 1, "dy": 2}),
         ("pan", 1.0, 2.0)),
        (lambda s: s.handle_camera("zoom", {"delta": -2}), ("zoom", -2.0)),
        (lambda s: s.handle_camera("move", {"forward": 1, "dt": 0.5}),
         ("move", 1.0, 0.0, 0.0, 0.5)),
    ],
)
def test_camera_gestures_are_posted_not_applied_inline(session, call, expected):
    call(session)
    assert session.renderer.camera.log == [], "camera moved on the caller's thread"
    _drain(session)
    assert session.renderer.camera.log == [expected]


@pytest.mark.parametrize(
    ("action", "check"),
    [
        ("toggle_camera", lambda r: ("toggle_camera",) in r.log),
        ("reset_camera", lambda r: ("reset_camera",) in r.log),
        ("toggle_hud", lambda r: r.show_hud is False),
        ("toggle_focus_overlay", lambda r: r.show_focus_overlay is True),
        ("toggle_lens_vignette", lambda r: r.lens_vignette_debug is True),
    ],
)
def test_control_actions_are_posted_not_applied_inline(session, action, check):
    before = dict(vars(session.renderer))
    session.handle_control(action)
    assert vars(session.renderer)["show_hud"] == before["show_hud"]
    assert session.renderer.log == [], "control ran on the caller's thread"
    _drain(session)
    assert check(session.renderer)


def test_autofocus_is_posted_not_applied_inline(session):
    session.handle_autofocus(12.0, 34.0)
    assert session.renderer.log == []
    _drain(session)
    assert session.renderer.log == [("autofocus", 12.0, 34.0)]


def test_set_param_is_posted_not_applied_inline(session):
    session.set_param("mtlx.base", 0.25)
    assert session.renderer.mtlx_overrides == {}
    _drain(session)
    assert session.renderer.mtlx_overrides["base"] == 0.25


def test_camera_gestures_are_not_coalesced(session):
    """A drag is a stream of deltas the camera integrates; last-write-wins
    would rotate less than the mouse did."""
    for _ in range(5):
        session.handle_camera("orbit", {"dx": 1, "dy": 0})
    _drain(session)
    assert session.renderer.camera.log == [("orbit", 1.0, 0.0)] * 5


def test_control_toggles_are_not_coalesced(session):
    """Collapsing two presses of a toggle into one inverts the result."""
    session.handle_control("toggle_hud")
    session.handle_control("toggle_hud")
    _drain(session)
    assert session.renderer.show_hud is True


def test_posted_commands_keep_their_order(session):
    session.handle_control("toggle_hud")
    session.handle_autofocus(1.0, 2.0)
    session.handle_camera("zoom", {"delta": 1})
    session.handle_control("reset_camera")
    _drain(session)
    assert [e[0] for e in session.renderer.log] == ["autofocus", "reset_camera"]
    assert session.renderer.camera.log == [("zoom", 1.0)]


def test_keyframe_is_forced_with_the_mutation_not_before_it(session):
    session.handle_camera("orbit", {"dx": 1, "dy": 1})
    assert session.encoder.keyframes == 0, "re-keyed before the camera moved"
    _drain(session)
    assert session.encoder.keyframes == 1


# ── skinny-web: awaited commands carry a reply ───────────────────────────


def _reply_runner(session):
    """Settle awaited commands as if a render thread were draining."""
    original = session._commands.post_with_reply

    def post_and_run(callback):
        future = original(callback)
        session._commands.run_pending(session.renderer)
        return future

    session._commands.post_with_reply = post_and_run


def test_screenshot_runs_on_the_render_thread_and_returns_bytes(session):
    _reply_runner(session)
    assert session.screenshot("png") == b"PNGBYTES"
    assert session.renderer.log == [("screenshot", "png")]


def test_loads_run_on_the_render_thread(session):
    _reply_runner(session)
    session.load_model("/tmp/a.usda")
    session.load_hdr("/tmp/b.hdr")
    assert session.renderer.log == [
        ("load_model", "/tmp/a.usda"), ("load_hdr", "/tmp/b.hdr"),
    ]


def test_a_failing_command_reports_its_error_to_the_caller(session):
    """The reply contract the tool surface already has, on the web too."""
    _reply_runner(session)

    def boom(_r):
        raise ValueError("no such scene")

    with pytest.raises(ValueError, match="no such scene"):
        session.run_on_render_thread(boom)


def test_debug_view_without_an_open_pane_is_a_no_op(session):
    _reply_runner(session)
    session.debug_view("top")  # pane never built → nothing published
    assert session.renderer.log == []


def test_debug_view_posts_the_panes_published_action(session):
    _reply_runner(session)
    seen: list[tuple] = []
    session._debug_view_action = lambda r, which: seen.append((r, which))
    session.debug_view("left")
    assert seen == [(session.renderer, "left")]


# ── the shared control tree's proxies ────────────────────────────────────


def test_marshalled_renderer_posts_the_furnace_toggle():
    """`toggle_material_furnace` is a renderer mutation the shared tree calls.
    It is not spelled `apply_*`, so the prefix rule let it through to the live
    renderer on the caller's thread."""
    live = _StubRenderer()
    queue = RenderCommandQueue()
    MarshalledRenderer(queue, lambda: live).toggle_material_furnace(2, True)
    assert live.log == [], "the furnace toggle ran inline"
    queue.run_pending(live)
    assert live.log == [("furnace", 2, True)]


def test_marshalled_renderer_refuses_an_unmarshalled_mutation_verb():
    live = _StubRenderer()
    proxy = MarshalledRenderer(RenderCommandQueue(), lambda: live)
    MarshalledRenderer._MUTATION_VERBS = (
        *MarshalledRenderer._MUTATION_VERBS, "reset_camera",
    )
    try:
        with pytest.raises(AttributeError, match="no marshalled verb"):
            proxy.reset_camera
    finally:
        MarshalledRenderer._MUTATION_VERBS = tuple(
            v for v in MarshalledRenderer._MUTATION_VERBS if v != "reset_camera"
        )


def test_qt_proxy_marshals_the_furnace_toggle_too():
    """The Qt front-end mounts the same tree. Without the verb the proxy's
    `__getattr__` raises, because it serves mirrored values and a verb is not
    a value."""
    posted: list = []
    proxy = QtRendererProxy.__new__(QtRendererProxy)
    object.__setattr__(proxy, "post", lambda cb, coalesce_key=None: posted.append(cb))
    QtRendererProxy.toggle_material_furnace(proxy, 3, False)
    live = _StubRenderer()
    posted[0](live)
    assert live.log == [("furnace", 3, False)]


def test_shared_tree_reaches_only_marshalled_mutation_verbs():
    """Whatever `build_main_ui` binds to is a marshalling proxy on the web, so
    every renderer mutation the shared tree calls must have a marshalled verb.
    Fails loudly instead of silently running one on the caller's thread."""
    import re

    from skinny.ui import build_app_ui

    src = inspect.getsource(build_app_ui)
    reached = {
        v for v in re.findall(r"renderer\.([a-z_]+)\(", src)
        if MarshalledRenderer._is_mutation(v)
    }
    assert reached, "expected the tree to reach at least one mutation verb"
    assert reached <= set(MarshalledRenderer._MARSHALLED_VERBS), (
        f"shared tree reaches unmarshalled verbs: "
        f"{sorted(reached - set(MarshalledRenderer._MARSHALLED_VERBS))}"
    )


# ── skinny-render: post and drain synchronously ──────────────────────────


def test_headless_posts_and_drains_in_order():
    """The degenerate case of the same interface: no second thread, so the
    drain happens under the post. Order must match the direct-call sequence."""
    from skinny.headless import HeadlessRenderer

    driver = HeadlessRenderer.__new__(HeadlessRenderer)
    driver._commands = RenderCommandQueue()
    driver.renderer = _StubRenderer()

    driver._run(lambda r: r.log.append(("one",)))
    driver._run(lambda r: r.log.append(("two",)))
    assert driver.renderer.log == [("one",), ("two",)]
    assert len(driver._commands) == 0, "a command was left pending"


def test_headless_run_returns_the_commands_result():
    from skinny.headless import HeadlessRenderer

    driver = HeadlessRenderer.__new__(HeadlessRenderer)
    driver._commands = RenderCommandQueue()
    driver.renderer = _StubRenderer()
    assert driver._run(lambda r: 41 + 1) == 42


def test_headless_run_raises_at_the_call_site():
    """A direct write that raised, raised here. Posting must not swallow it."""
    from skinny.headless import HeadlessRenderer

    driver = HeadlessRenderer.__new__(HeadlessRenderer)
    driver._commands = RenderCommandQueue()
    driver.renderer = _StubRenderer()

    def boom(_r):
        raise RuntimeError("scene has no usable materials")

    with pytest.raises(RuntimeError, match="no usable materials"):
        driver._run(boom)


def test_headless_prepare_applies_the_scene_before_the_options(monkeypatch):
    """Options are applied AFTER the scene swap so they win over anything
    `_apply_usd_lights` seeded. Posting must not reorder that."""
    from skinny import headless as headless_mod
    from skinny.headless import HeadlessRenderer, RenderOptions

    monkeypatch.setattr(headless_mod, "_load_scene", lambda src, t: object())
    order: list[str] = []

    class _Recorder:
        def set_usd_scene(self, scene) -> None:
            order.append("scene")

        def __setattr__(self, name, value) -> None:
            order.append(name)
            object.__setattr__(self, name, value)

    driver = HeadlessRenderer.__new__(HeadlessRenderer)
    driver._commands = RenderCommandQueue()
    driver.renderer = _Recorder()
    HeadlessRenderer._prepare(
        driver, "scene.usda", RenderOptions(samples=1, env_intensity=2.0),
    )
    assert order[0] == "scene"
    assert order[1:] == [
        "integrator_index", "tonemap_index", "exposure",
        "direct_light_index", "env_intensity",
    ]


# ── 4.4 source gate ──────────────────────────────────────────────────────


#: Front-end modules whose code runs on a thread that does not own the
#: renderer. A `with session._lock:` here used to serialise a mutation against
#: the render without moving it onto the owning thread — the distinction the
#: whole change rests on (design D1).
_OFF_THREAD_MODULES = (
    "web_app.py",
    "ui/panel/windows.py",
    "ui/panel/backend.py",
)


@pytest.mark.parametrize("rel", _OFF_THREAD_MODULES)
def test_no_front_end_mutates_the_renderer_under_the_session_lock(rel):
    """`session._lock` is the render+encode lock. Taking it around a renderer
    call is the pattern this change removes; the render loop's own drain is the
    one legitimate holder, and it lives on the session, not here."""
    source = (SRC / rel).read_text()
    assert "session._lock" not in source, (
        f"{rel} still mutates the renderer under the session lock; post it"
    )


def test_the_debug_shortcut_does_not_poke_a_widget_tree():
    source = (SRC / "web_app.py").read_text()
    assert ".clicks +=" not in source, (
        "the Camera-Debug shortcut still synthesises a click by widget index"
    )


def test_the_render_loop_still_drains_before_advancing_the_renderer():
    """The one place the lock legitimately wraps renderer work — and the drain
    must come first, or a posted command lands after the frame it was meant for.
    """
    source = textwrap.dedent(inspect.getsource(SkinnySession._render_iteration))
    tree = ast.parse(source)
    calls = [
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert calls.index("run_pending") < calls.index("update"), (
        "the render loop advances the renderer before draining its commands"
    )


def test_every_awaited_session_command_has_a_timeout():
    """A reply future waited on without a timeout hangs the IOLoop thread for
    good if the render thread dies before settling it."""
    tree = ast.parse((SRC / "web_app.py").read_text())
    awaited = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "result"
    ]
    assert awaited, "expected the session to await at least one reply"
    for node in awaited:
        assert any(kw.arg == "timeout" for kw in node.keywords), (
            f"reply awaited without a timeout at web_app.py:{node.lineno}"
        )


def test_a_dead_render_thread_settles_pending_replies(session):
    """Otherwise a caller blocked in `run_on_render_thread` waits out its whole
    timeout against a thread that is never coming back."""
    future: Future = session._commands.post_with_reply(lambda r: None)
    session._notify_render_failed(RuntimeError("gpu lost"))
    assert future.done()
    with pytest.raises(RuntimeError, match="render thread stopped"):
        future.result(timeout=0)
