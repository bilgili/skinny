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
        self.width = 1280
        self.height = 720
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
    avcc_description = b"AVCC"

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
    """`SkinnySession.set_param` posts.

    Note it is NOT the browser's slider path — nothing in the UI routes to it.
    A slider goes through the sidebar's `MarshalledRenderer.set_path`, which
    the next test covers. This one keeps the session method honest for whenever
    a parameter WebSocket message is added.
    """
    session.set_param("mtlx.base", 0.25)
    assert session.renderer.mtlx_overrides == {}
    _drain(session)
    assert session.renderer.mtlx_overrides["base"] == 0.25


def test_the_browsers_actual_slider_path_is_posted(session):
    """The sidebar binds the shared tree to `MarshalledRenderer`, and
    `set_param_value` resolves through its `set_path`. This is the path a
    browser slider drag really takes."""
    from skinny.params import set_param_value

    sidebar = MarshalledRenderer(session._commands, lambda: session.renderer)
    set_param_value(sidebar, "mtlx.base", 0.5)
    assert session.renderer.mtlx_overrides == {}, "slider wrote the live renderer"
    _drain(session)
    assert session.renderer.mtlx_overrides["base"] == 0.5


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


#: Renderer members the shared control tree may call on a proxy without a
#: marshalled verb: reads, and the two verbs every front-end overrides with a
#: host callback. Anything else the tree reaches has to be classified — as a
#: read (add it here, with a reason) or as a mutation (give it a marshalled
#: verb). The point is that a NEW name fails until someone decides which it is.
_TREE_MAY_REACH_UNMARSHALLED = {
    # Reads.
    "iter_graph_uniforms", "proposal_preset_from_token",
    # Host-callback verbs: `AppCallbacks.load_model` / `capture_screenshot`
    # replace these on every front-end that mounts the tree, so the direct call
    # is a fallback branch, not the live path. See `_add_scene_loader` /
    # `_add_capture`, which refuse to build the control without the callback.
    "load_model_from_path", "save_screenshot",
}


def test_shared_tree_reaches_only_classified_renderer_members():
    """Whatever `build_main_ui` binds to is a marshalling proxy on the web, so
    every renderer mutation the shared tree calls must have a marshalled verb.

    This deliberately does NOT filter the scan through
    `MarshalledRenderer._is_mutation` first. Doing that made the gate circular:
    an unmarshalled verb is by definition not yet in `_MUTATION_VERBS`, so the
    filter dropped exactly the names the gate exists to catch, and the
    assertion could only ever fire for a verb someone had already remembered to
    register. Scan every member, then require each one to be classified.
    """
    import re

    from skinny.ui import build_app_ui

    src = inspect.getsource(build_app_ui)
    # `[a-z_]+` missed names with digits or capitals; `\w+` does not.
    reached = set(re.findall(r"renderer\.(\w+)\s*\(", src))
    assert reached, "expected the tree to reach at least one renderer member"
    unclassified = (
        reached
        - set(MarshalledRenderer._MARSHALLED_VERBS)
        - _TREE_MAY_REACH_UNMARSHALLED
    )
    assert not unclassified, (
        f"the shared control tree reaches renderer members that are neither "
        f"marshalled nor declared reads: {sorted(unclassified)}. Give each a "
        f"marshalled verb on MarshalledRenderer (and QtRendererProxy), or add "
        f"it to _TREE_MAY_REACH_UNMARSHALLED with the reason it is safe."
    )


def test_the_verb_gate_can_actually_fail():
    """Negative control for the gate above.

    The version this replaced passed unchanged when a new unmarshalled verb was
    added to the scanned source, because it filtered by `_is_mutation` first.
    """
    import re

    source = "renderer.reset_zoom_rect()\nrenderer.apply_material_override(1)\n"
    reached = set(re.findall(r"renderer\.(\w+)\s*\(", source))
    unclassified = (
        reached
        - set(MarshalledRenderer._MARSHALLED_VERBS)
        - _TREE_MAY_REACH_UNMARSHALLED
    )
    assert unclassified == {"reset_zoom_rect"}, (
        "the gate no longer notices an unregistered verb"
    )


# ── A stopped session refuses, it does not accumulate ────────────────────


def test_a_stopped_session_drops_posts_instead_of_queueing_them(session):
    """After the render thread gives up, nothing drains. A browser that keeps
    dragging would otherwise grow the queue for the life of the process."""
    session._notify_render_failed(RuntimeError("gpu lost"))
    for _ in range(50):
        session.handle_camera("orbit", {"dx": 1, "dy": 0})
    assert len(session._commands) == 0


def test_a_stopped_session_fails_an_awaited_command_at_once(session):
    """Rather than making the caller wait out a 30 s timeout against a thread
    that is never coming back."""
    session._notify_render_failed(RuntimeError("gpu lost"))
    with pytest.raises(RuntimeError, match="render thread stopped"):
        session.run_on_render_thread(lambda r: None)


def test_a_timed_out_command_is_cancelled_not_left_to_land(session):
    """`run_pending` skips cancelled commands, but only if someone cancels.
    An uncancelled resize lands minutes after the caller was told it failed."""
    with pytest.raises(TimeoutError):
        session.run_on_render_thread(lambda r: r.log.append(("late",)),
                                     timeout=0.01)
    _drain(session)
    assert session.renderer.log == [], "the timed-out command ran anyway"


def test_prime_stream_returns_an_unsettled_future(session):
    """The only caller is a Tornado coroutine. Blocking there stalls every
    other session's frame writes for the whole timeout."""
    future = session.prime_stream()
    assert isinstance(future, Future)
    assert not future.done(), "prime_stream blocked the caller"
    _drain(session)
    assert future.result(timeout=0) == (1280, 720, b"AVCC")


# ── The USD control setter posts its stage write ─────────────────────────


def test_usd_control_setter_posts_the_stage_write_and_the_dirty_flag():
    """`attr.Set` mutates the live stage the render thread reads, and pxr gives
    no guarantee for a concurrent read/write. The flag rides the same command,
    because splitting them let the Qt proxy swallow the flag half."""
    from skinny.usd_controls import accessors_for

    applied: list = []

    class _Attr:
        def Get(self):
            return 1.0

        def Set(self, v):
            applied.append(("set", v))

        def GetPath(self):
            return "/World/light.intensity"

    live = _StubRenderer()
    queue = RenderCommandQueue()
    proxy = MarshalledRenderer(queue, lambda: live)
    binding = type("B", (), {"kind": "usd", "attribute": _Attr()})()

    _get, _set = accessors_for(proxy, binding)
    _set(2.5)
    assert applied == [], "the stage was written on the caller's thread"
    queue.run_pending(live)
    assert applied == [("set", 2.5)]
    assert live._usd_live_dirty is True, "the dirty flag was swallowed"


def test_usd_control_setter_applies_inline_without_a_proxy():
    """The single-threaded front-end binds the live renderer and already runs
    on the owning thread."""
    from skinny.usd_controls import accessors_for

    applied: list = []
    attr = type("A", (), {
        "Get": lambda self: 1.0,
        "Set": lambda self, v: applied.append(v),
        "GetPath": lambda self: "/W/l.i",
    })()
    live = _StubRenderer()
    binding = type("B", (), {"kind": "usd", "attribute": attr})()

    accessors_for(live, binding)[1](3.0)
    assert applied == [3.0]
    assert live._usd_live_dirty is True


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


def _off_thread_modules() -> list[Path]:
    """Every module whose code runs on a thread that does not own the renderer.

    **Derived, not hand-listed.** A hand-list silently exempts the next pane
    someone adds under `ui/panel/`, which is exactly the module most likely to
    reach for the lock by copying its neighbour.
    """
    return sorted(SRC.glob("ui/panel/*.py")) + [SRC / "web_app.py"]


def test_the_off_thread_module_list_is_not_empty():
    """A derived list that silently derives nothing is a gate that always
    passes."""
    mods = _off_thread_modules()
    assert len(mods) >= 3, f"expected the panel front-end modules, got {mods}"
    assert (SRC / "ui/panel/windows.py") in mods


@pytest.mark.parametrize(
    "path", _off_thread_modules(), ids=lambda p: p.name,
)
def test_no_front_end_takes_the_session_lock(path):
    """`session._lock` is the render+encode lock. Taking it around a renderer
    call serialises that call against the render but does not move it onto the
    owning thread — the distinction the whole change rests on (design D1). The
    render loop's own drain is the one legitimate holder, and it lives on the
    session itself, not in these modules.

    AST, not a substring: `sess = session` then `with sess._lock:` defeated the
    string check, and so did `getattr(session, "_lock")`. This walks every
    attribute access named `_lock` instead.
    """
    tree = ast.parse(path.read_text())
    hits = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_lock"
        # `web_app.py` owns the lock; `self._lock` in the render loop is the
        # legitimate holder. Any OTHER base is a front-end reaching for it.
        and not (isinstance(node.value, ast.Name) and node.value.id == "self")
    ]
    hits += [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "_lock"
    ]
    assert not hits, (
        f"{path.name} reaches for the render lock at line(s) {hits}; post the "
        f"mutation through the session instead"
    )


@pytest.mark.parametrize(
    "path", _off_thread_modules(), ids=lambda p: p.name,
)
def test_no_front_end_writes_a_renderer_attribute_off_thread(path):
    """The thing task 4.4 actually claims: no direct renderer attribute write
    from a non-owning thread.

    The lock gate above cannot see `renderer.foo = x` with no lock at all,
    which is a *worse* version of the banned pattern. This one looks for
    attribute stores whose base is a name the front-end uses for a live
    renderer. A write inside a posted command binds the renderer to the
    callback's own parameter (`r`, or a shadowing `renderer`), so it is not a
    module-level name and does not trip this.
    """
    LIVE = {"renderer"}
    hits: list[tuple[str, str, int]] = []

    class Scan(ast.NodeVisitor):
        """Skips whole function bodies whose first parameter is the renderer.

        `ast.walk` cannot express that — it flattens the tree, so a `continue`
        on a FunctionDef still visits every node inside it. A visitor that
        simply does not recurse is the only way to model the scope.
        """

        def _is_command_body(self, node) -> bool:
            args = node.args.args
            return bool(args) and args[0].arg in LIVE | {"r"}

        def visit_FunctionDef(self, node) -> None:
            if not self._is_command_body(node):
                self.generic_visit(node)

        def visit_Lambda(self, node) -> None:
            if not self._is_command_body(node):
                self.generic_visit(node)

        def _record(self, target, lineno) -> None:
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in LIVE):
                hits.append((target.value.id, target.attr, lineno))

        def visit_Assign(self, node) -> None:
            for t in node.targets:
                self._record(t, node.lineno)
            self.generic_visit(node)

        def visit_AugAssign(self, node) -> None:
            self._record(node.target, node.lineno)
            self.generic_visit(node)

    Scan().visit(ast.parse(path.read_text()))
    assert not hits, (
        f"{path.name} writes renderer state directly off-thread: {hits}. "
        f"Post the write through the session's command queue."
    )


def test_the_renderer_write_gate_can_actually_fail(tmp_path):
    """Negative control: a bare off-thread write, and a legitimate one inside a
    posted command, must be told apart."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "def bad(session):\n"
        "    renderer = session.renderer\n"
        "    renderer.exposure = 2.0\n"
        "def ok(renderer):\n"
        "    renderer.exposure = 2.0\n"
    )
    LIVE = {"renderer"}
    hits: list[int] = []

    class Scan(ast.NodeVisitor):
        def visit_FunctionDef(self, node) -> None:
            args = node.args.args
            if not (args and args[0].arg in LIVE | {"r"}):
                self.generic_visit(node)

        def visit_Assign(self, node) -> None:
            for t in node.targets:
                if (isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id in LIVE):
                    hits.append(node.lineno)
            self.generic_visit(node)

    Scan().visit(ast.parse(probe.read_text()))
    assert hits == [3], (
        "the gate must flag the off-thread write and spare the posted one"
    )


def test_the_lock_gate_can_actually_fail(tmp_path):
    """Negative control. The string version this replaced passed for
    `sess = session; with sess._lock:` — the AST version must not."""
    probe = tmp_path / "probe.py"
    probe.write_text("def f(session):\n    sess = session\n    with sess._lock:\n        pass\n")
    tree = ast.parse(probe.read_text())
    hits = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "_lock"
        and not (isinstance(n.value, ast.Name) and n.value.id == "self")
    ]
    assert hits, "the aliased lock grab slipped through"


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
