"""Hostless tests for the web session's render loop (review-surfaced-defects 1).

Two independent guarantees, both exercised against a stub renderer:

* a parameter write posted from another thread cannot make the accumulation
  state hash raise `dictionary keys changed during iteration`, because the
  write runs on the render thread between frames; and
* a command that raises anyway does not retire the render thread while the
  session stays marked running.
"""

from __future__ import annotations

import inspect
import threading
import time

import pytest

from skinny.params import ACCUM_STATE_PROVIDERS, _set_nested
from skinny.playback import PlaybackClock
from skinny.render_session import MarshalledRenderer, RenderCommandQueue
from skinny.web_app import SkinnySession

MTLX_PROVIDER = next(
    p for p in ACCUM_STATE_PROVIDERS if p.name == "mtlx_overrides"
)


class _StubRenderer:
    """Just enough renderer for the accumulation-state provider."""

    def __init__(self) -> None:
        self.mtlx_overrides = {f"mtlx.f{i}": float(i) for i in range(200)}


def _session(session_id: str) -> SkinnySession:
    """A session object with no GPU behind it -- `__init__` builds only the
    queue, lock and frame queue; `initialize()` is what needs a device."""
    session = SkinnySession(session_id)
    session.renderer = _StubRenderer()
    return session


def test_posted_mtlx_write_cannot_break_the_state_hash() -> None:
    session = _session("marshal-01")
    try:
        renderer = session.renderer
        stop = threading.Event()
        failures: list[BaseException] = []

        def ui_thread() -> None:
            n = 0
            while not stop.is_set():
                n += 1
                session.set_param(f"mtlx.slider{n % 8}", 0.5)

        writer = threading.Thread(target=ui_thread, daemon=True)
        writer.start()
        try:
            for _ in range(20000):
                try:
                    session._commands.run_pending(renderer)
                    MTLX_PROVIDER.extractor(renderer)
                except RuntimeError as exc:  # pragma: no cover - the defect
                    failures.append(exc)
                    break
        finally:
            stop.set()
            writer.join(timeout=5)

        assert not failures, failures[0]
        # The writes landed -- marshalling is not dropping them.
        session._commands.run_pending(renderer)
        assert any(k.startswith("slider") for k in renderer.mtlx_overrides)
    finally:
        session._active.pop(session.session_id, None)


def test_set_param_does_not_touch_the_renderer_until_the_loop_runs() -> None:
    session = _session("marshal-02")
    try:
        session.set_param("mtlx.layer_top_melanin", 0.42)
        assert "layer_top_melanin" not in session.renderer.mtlx_overrides
        session._commands.run_pending(session.renderer)
        assert session.renderer.mtlx_overrides["layer_top_melanin"] == 0.42
    finally:
        session._active.pop(session.session_id, None)


def test_render_loop_survives_a_raising_iteration() -> None:
    session = _session("guard-01")
    try:
        calls: list[int] = []

        def flaky(dt: float) -> None:
            calls.append(len(calls))
            if len(calls) == 1:
                raise RuntimeError("dictionary keys changed during iteration")
            session._running = False

        session._render_iteration = flaky
        session._running = True
        session._render_loop()

        assert len(calls) == 2, "the loop retired on the first exception"
        assert isinstance(session._render_error, RuntimeError)
    finally:
        session._active.pop(session.session_id, None)


def test_render_loop_gives_up_visibly_when_it_never_recovers() -> None:
    session = _session("guard-02")
    try:
        session.MAX_RENDER_FAILURES = 3
        calls: list[int] = []

        def always_raises(dt: float) -> None:
            calls.append(len(calls))
            raise RuntimeError("renderer is gone")

        session._render_iteration = always_raises
        session._running = True
        session._render_loop()

        # Stopped, and never left marked running with a dead thread.
        assert len(calls) == 3
        assert session._running is False
        assert isinstance(session._render_error, RuntimeError)
    finally:
        session._active.pop(session.session_id, None)


def test_a_raising_command_is_reported_and_does_not_propagate() -> None:
    queue = RenderCommandQueue()
    reported: list[str] = []
    renderer = _StubRenderer()

    def boom(_r):
        raise ValueError("bad edit")

    queue.post(boom)
    queue.post(lambda r: _set_nested(r, "mtlx.after", 1.0))
    queue.run_pending(renderer, on_error=reported.append)

    assert len(reported) == 1 and "bad edit" in reported[0]
    # The command after the failure still ran.
    assert renderer.mtlx_overrides["after"] == 1.0


def test_usd_declared_controls_write_through_the_marshalling_seam() -> None:
    """`resolve_control_binding` used `_set_nested` directly, which resolves
    through a proxy to the live mapping behind it — the same unmarshalled
    `mtlx.*` insert, reached through a USD-declared control instead of a slider.
    """
    from skinny.usd_controls import control_accessors

    posted: list[tuple[str, object]] = []
    proxy = type("Proxy", (), {
        "set_path": lambda self, path, value: posted.append((path, value)),
        "mtlx_overrides": {},
    })()

    for target in ("mtlx:layer_top_melanin", "renderer:env_intensity"):
        spec = type("Spec", (), {"target": target})()
        _getter, setter = control_accessors(proxy, spec)
        setter(0.5)

    assert posted == [("mtlx.layer_top_melanin", 0.5), ("env_intensity", 0.5)]
    assert proxy.mtlx_overrides == {}, "the write reached through the proxy"


# ── Round-2 fixes from the codex pre-merge review ────────────────────


def test_terminal_failure_is_visible_and_settles_waiters() -> None:
    """Giving up must reach the client and unblock anyone awaiting a reply --
    `_running = False` alone reads as a merely slow frame (finding 6)."""
    session = _session("guard-03")
    try:
        session.MAX_RENDER_FAILURES = 2
        notices: list[str] = []
        session._handlers.append(
            type("H", (), {"send_render_failed": lambda self, r: notices.append(r)})()
        )
        waiter = session._commands.post_with_reply(lambda r: "never runs")

        session._render_iteration = lambda dt: (_ for _ in ()).throw(
            RuntimeError("device lost"))
        session._running = True
        session._render_loop()

        assert len(notices) == 1 and "device lost" in notices[0]
        assert session._running is False
        with pytest.raises(RuntimeError, match="render thread stopped"):
            waiter.result(timeout=1)
    finally:
        session._active.pop(session.session_id, None)


def test_resize_is_applied_by_the_render_thread_not_the_caller() -> None:
    """`session.resize` posts the whole compound; holding the lock around a
    direct `renderer.resize` only serialises it, it does not move it (finding 3).
    """
    session = _session("resize-01")
    try:
        applied: list[tuple] = []
        session._apply_resize = lambda r, w, h: (applied.append((threading.get_ident(), w, h)), (w, h))[1]
        session.RESIZE_TIMEOUT_S = 5.0
        caller = threading.get_ident()
        result: list = []

        def render_thread() -> None:
            for _ in range(200):
                session._commands.run_pending(session.renderer)
                time.sleep(0.005)

        t = threading.Thread(target=render_thread, daemon=True)
        t.start()
        result.append(session.resize(320, 240))
        t.join(timeout=3)

        assert result == [(320, 240)]
        assert len(applied) == 1
        assert applied[0][0] != caller, "resize ran on the calling thread"
        assert applied[0][1:] == (320, 240)
    finally:
        session._active.pop(session.session_id, None)


def test_marshalled_renderer_refuses_an_unmarshalled_apply_verb() -> None:
    """Passing `apply_*` through to the live renderer would run a mutation on
    the caller's thread while reading as "every write is a command" (finding 4).
    """
    calls: list[tuple] = []
    live = type("Live", (), {
        "apply_material_override": lambda self, *a: calls.append(a),
        "apply_light_override": lambda self, *a: calls.append(("light",) + a),
        "width": 640,
    })()
    queue = RenderCommandQueue()
    proxy = MarshalledRenderer(queue, lambda: live)

    assert proxy.width == 640                       # reads still pass through

    proxy.apply_material_override(2, "base_color", 0.5)
    assert calls == [], "the marshalled verb ran inline"
    queue.run_pending(live)
    assert calls == [(2, "base_color", 0.5)]

    with pytest.raises(AttributeError, match="no marshalled verb"):
        proxy.apply_light_override


def test_marshalled_renderer_has_no_mirror_to_absorb_a_clock_write() -> None:
    live = type("Live", (), {})()
    live.clock = PlaybackClock(start_time_code=0.0, end_time_code=100.0)
    queue = RenderCommandQueue()
    MarshalledRenderer(queue, lambda: live).set_clock_state("normalized", 0.5)
    queue.run_pending(live)
    assert live.clock.current_time_code == 50.0


def test_preset_application_goes_through_the_marshalling_seam() -> None:
    """`apply_preset` looped raw `_set_nested`, which resolves through a proxy
    into the live mapping -- the same unmarshalled `mtlx.*` insert (finding 2).
    """
    from skinny.presets import Preset, apply_preset

    live = _StubRenderer()
    queue = RenderCommandQueue()
    proxy = MarshalledRenderer(queue, lambda: live)
    before = dict(live.mtlx_overrides)

    apply_preset(proxy, Preset(name="p", values={"mtlx.layer_top_melanin": 0.7}))
    assert live.mtlx_overrides == before, "preset wrote the live renderer directly"
    queue.run_pending(live)
    assert live.mtlx_overrides["layer_top_melanin"] == 0.7


def test_shared_tree_reaches_only_marshalled_renderer_verbs() -> None:
    """The boundary check behind `_MARSHALLED_VERBS`.

    Whatever `build_main_ui` binds to is a marshalling proxy on the web, so any
    `apply_*` the shared tree calls must have a marshalled verb. The tree turns
    out to reach exactly one — `apply_material_override`, from the material
    block and graph-uniform controls — so refusing the other eight breaks
    nothing. If someone wires a second verb into the tree, this fails instead of
    silently running it on the caller's thread.
    """
    import re

    from skinny.render_session import MarshalledRenderer
    from skinny.ui import build_app_ui

    src = inspect.getsource(build_app_ui)
    reached = set(re.findall(r"renderer\.(apply_[a-z_]+)\(", src))
    assert reached, "expected the tree to reach at least one apply_* verb"
    assert reached <= set(MarshalledRenderer._MARSHALLED_VERBS), (
        f"shared tree reaches unmarshalled verbs: "
        f"{sorted(reached - set(MarshalledRenderer._MARSHALLED_VERBS))}"
    )
