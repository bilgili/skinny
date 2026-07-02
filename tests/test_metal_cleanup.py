"""Metal context teardown + dispatch-hygiene kill harness.

Change ``nanovdb-volume-rendering`` (design D5.2/D5.3, spec
``metal-dispatch-hygiene``). Two halves:

- **Hostless** (default ``pytest``, no GPU anywhere): ``destroy()``
  idempotency, context-manager protocol, atexit register/unregister with a
  weak-only registry, and chained SIGINT/SIGTERM handlers — all against a stub
  device injected via ``MetalContext.__new__`` (construction is monolithic; the
  stub route exercises the real ``destroy()``/registry/hook code without a
  Metal device).
- **gpu-marked kill harness**: subprocess probes proving clean exit, SIGKILL
  mid-render, and the atexit path leave the GPU usable. Runs only under the
  guarded Metal runner (one Metal process at a time); the parent process never
  constructs a device — every device lives in a child (``sys.executable``,
  inherited env so ``VULKAN_SDK``/``DYLD_LIBRARY_PATH`` survive; never nohup —
  SIP strips DYLD).
"""

from __future__ import annotations

import gc
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import pytest

from skinny import metal_context as mc
from skinny.metal_context import MetalContext

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHILD = Path(__file__).resolve().parent / "metal_cleanup_child.py"
_ON_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
# Budget for a probe subprocess (device construct + trivial dispatch + exit).
_PROBE_BUDGET_S = 60.0


class FakeDevice:
    """Stub for the slang-rhi device: counts the teardown-relevant calls."""

    def __init__(self) -> None:
        self.wait_calls = 0
        self.close_calls = 0

    def wait_for_idle(self) -> None:
        self.wait_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _stub_context(device: FakeDevice | None = None) -> MetalContext:
    """Build a MetalContext around a stub device without running ``__init__``
    (which requires Apple-Silicon + slangpy). Mirrors the attribute state
    ``__init__`` establishes before/after device creation, then runs the real
    registration seam so atexit/signal hooks behave exactly as in production."""
    ctx = MetalContext.__new__(MetalContext)
    ctx.device = device if device is not None else FakeDevice()
    ctx.surface = None
    ctx.swapchain_info = None
    ctx._destroyed = False
    mc._register_context(ctx)
    return ctx


@pytest.fixture(autouse=True)
def _pristine_lifecycle_state():
    """Snapshot + restore the module-level hook state around every test so a
    failing assertion can never leak installed handlers into the suite."""
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    yield
    mc._drain_live_contexts()
    mc._uninstall_teardown_hooks()
    mc._PREV_SIGNAL_HANDLERS.clear()
    signal.signal(signal.SIGINT, prev_int)
    signal.signal(signal.SIGTERM, prev_term)


# ── destroy() idempotency + context manager (hostless) ──────────────


def test_destroy_is_idempotent():
    fake = FakeDevice()
    ctx = _stub_context(fake)
    ctx.destroy()
    ctx.destroy()
    ctx.destroy()
    assert fake.wait_calls == 1, "wait_for_idle must run exactly once"
    assert fake.close_calls == 1, "device.close must run exactly once"


def test_destroy_safe_when_construction_half_failed():
    # Before device creation: the attribute state __init__ seeds first.
    ctx = MetalContext.__new__(MetalContext)
    ctx.device = None
    ctx.surface = None
    ctx.swapchain_info = None
    ctx._destroyed = False
    ctx.destroy()  # must not raise
    # Even a completely bare instance (nothing set) must not raise.
    MetalContext.__new__(MetalContext).destroy()


def test_context_manager_destroys_on_clean_exit():
    fake = FakeDevice()
    with _stub_context(fake) as ctx:
        assert ctx.device is fake
    assert fake.wait_calls == 1 and fake.close_calls == 1
    ctx.destroy()  # still idempotent afterwards
    assert fake.close_calls == 1


def test_exception_between_dispatches_tears_down_and_propagates():
    """Spec scenario: a raise after dispatch N, before N+1 → drain + close run
    exactly once, and the exception is NOT swallowed by ``__exit__``."""
    fake = FakeDevice()
    with pytest.raises(RuntimeError, match="dispatch N\\+1"):
        with _stub_context(fake):
            raise RuntimeError("dispatch N+1 failed")
    assert fake.wait_calls == 1, "queue must be drained exactly once"
    assert fake.close_calls == 1, "device must be closed exactly once"


# ── atexit hook (hostless) ───────────────────────────────────────────


def test_atexit_registered_on_construction_unregistered_on_clean_destroy():
    assert not mc._ATEXIT_REGISTERED
    ctx = _stub_context()
    assert mc._ATEXIT_REGISTERED, "atexit hook must register with the first live context"
    ctx.destroy()
    assert not mc._ATEXIT_REGISTERED, "clean destroy of the last context must unregister"


def test_atexit_hook_destroys_live_contexts():
    fake = FakeDevice()
    ctx = _stub_context(fake)  # bound: the weak registry must not be the only ref
    mc._atexit_teardown()
    assert fake.wait_calls == 1 and fake.close_calls == 1
    assert ctx._destroyed


def test_atexit_marker_printed_when_env_set(capsys, monkeypatch):
    monkeypatch.setenv(mc._TEARDOWN_MARKER_ENV, "1")
    ctx = _stub_context()
    mc._atexit_teardown()
    assert mc._TEARDOWN_MARKER in capsys.readouterr().out
    assert ctx._destroyed


def test_registry_holds_only_weakrefs():
    """A hard ref in the atexit path would keep the device alive and reorder
    interpreter teardown — the registry must never extend context lifetime."""
    ctx = _stub_context()
    ref = weakref.ref(ctx)
    del ctx
    gc.collect()
    assert ref() is None, "registry kept the context alive (hard reference)"


# ── SIGINT / SIGTERM handlers (hostless) ─────────────────────────────


def test_signal_handler_installed_chains_and_restores():
    seen: list[tuple] = []

    def app_handler(signum, frame):
        seen.append((signum, frame))

    signal.signal(signal.SIGTERM, app_handler)
    fake = FakeDevice()
    ctx = _stub_context(fake)
    assert signal.getsignal(signal.SIGTERM) is mc._handle_teardown_signal

    # Fire the handler directly: teardown runs, then the previous handler is
    # delegated to (the app's semantics are preserved, never replaced).
    mc._handle_teardown_signal(signal.SIGTERM, None)
    assert fake.wait_calls == 1 and fake.close_calls == 1
    assert seen == [(signal.SIGTERM, None)]

    ctx.destroy()  # idempotent no-op for teardown, but unregisters the hooks
    assert signal.getsignal(signal.SIGTERM) is app_handler, "previous handler not restored"


def test_sigint_preserves_keyboardinterrupt_semantics():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    fake = FakeDevice()
    ctx = _stub_context(fake)  # bound: the weak registry must not be the only ref
    with pytest.raises(KeyboardInterrupt):
        mc._handle_teardown_signal(signal.SIGINT, None)
    assert fake.wait_calls == 1 and fake.close_calls == 1
    assert ctx._destroyed


def test_handler_not_restored_if_app_installed_its_own_later():
    ctx = _stub_context()

    def late_handler(signum, frame):  # app overrides us mid-session
        pass

    signal.signal(signal.SIGTERM, late_handler)
    ctx.destroy()
    assert signal.getsignal(signal.SIGTERM) is late_handler, "must never clobber a later handler"


def test_no_signal_install_from_non_main_thread():
    """signal.signal raises off the main thread — registration must not try."""
    before = signal.getsignal(signal.SIGTERM)
    err: list[BaseException] = []

    def register():
        try:
            _stub_context()
        except BaseException as exc:  # noqa: BLE001 — assert no raise below
            err.append(exc)

    t = threading.Thread(target=register)
    t.start()
    t.join()
    assert not err, f"registration raised off the main thread: {err}"
    assert signal.getsignal(signal.SIGTERM) is before, "handler installed off main thread"
    assert mc._ATEXIT_REGISTERED, "atexit (thread-safe) should still register"


def test_sigterm_subprocess_chains_to_default_death():
    """End-to-end hostless: a child registers a stub context, raises SIGTERM at
    itself → the chained handler tears down, then the restored default action
    kills the process (rc == -SIGTERM). No GPU anywhere."""
    child = (
        "import signal\n"
        "from skinny import metal_context as mc\n"
        "class FakeDevice:\n"
        "    def wait_for_idle(self): print('DRAINED', flush=True)\n"
        "    def close(self): print('CLOSED', flush=True)\n"
        "ctx = mc.MetalContext.__new__(mc.MetalContext)\n"
        "ctx.device = FakeDevice()\n"
        "ctx.surface = None\n"
        "ctx.swapchain_info = None\n"
        "ctx._destroyed = False\n"
        "mc._register_context(ctx)\n"
        "signal.raise_signal(signal.SIGTERM)\n"
        "print('UNREACHABLE', flush=True)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True, text=True, timeout=60.0, env=env, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == -signal.SIGTERM, (
        f"expected death by SIGTERM, got rc={proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "DRAINED" in proc.stdout and "CLOSED" in proc.stdout, proc.stdout
    assert "UNREACHABLE" not in proc.stdout, "SIGTERM default action was swallowed"


# ── gpu kill harness (design D5.3) ───────────────────────────────────
#
# Never run in default `pytest` (ZERO-SWAP / one-guarded-Metal-process rules):
#   PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
#       tests/test_metal_cleanup.py -m gpu -x -q
# The parent never constructs a Metal device; children own it one at a time.


def _run_child(*args: str, env_extra: dict | None = None,
               timeout: float = _PROBE_BUDGET_S) -> subprocess.CompletedProcess:
    env = os.environ.copy()  # inherits VULKAN_SDK / DYLD_LIBRARY_PATH
    env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(_CHILD), *args],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=str(_REPO_ROOT),
    )


def _assert_probe_ok(proc: subprocess.CompletedProcess, what: str) -> None:
    assert proc.returncode == 0 and "PROBE_OK" in proc.stdout, (
        f"{what}: GPU probe failed (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not _ON_APPLE_SILICON, reason="Metal kill harness needs Apple-Silicon macOS")
class TestMetalKillHarness:
    def test_clean_exit_probe(self):
        """A subprocess that dispatches and exits cleanly leaves the GPU usable
        for a second probe subprocess within the budget."""
        _assert_probe_ok(_run_child("probe"), "first clean-exit probe")
        start = time.monotonic()
        second = _run_child("probe")
        elapsed = time.monotonic() - start
        _assert_probe_ok(second, "second probe after clean exit")
        assert elapsed < _PROBE_BUDGET_S, f"second probe took {elapsed:.1f}s"

    def test_sigkill_mid_render_leaves_gpu_usable(self, tmp_path):
        """Spec scenario 'SIGKILL mid-render leaves GPU usable': the child gets
        no chance to run teardown; bounded command buffers must die with it."""
        sentinel = tmp_path / "frame1.flag"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        child = subprocess.Popen(
            [sys.executable, str(_CHILD), "render", str(sentinel)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, cwd=str(_REPO_ROOT),
        )
        try:
            deadline = time.monotonic() + _PROBE_BUDGET_S
            while not sentinel.exists():
                assert child.poll() is None, (
                    f"render child died before frame 1 (rc={child.returncode}): "
                    f"{child.stderr.read().decode(errors='replace') if child.stderr else ''}"
                )
                assert time.monotonic() < deadline, "render child never wrote the frame-1 sentinel"
                time.sleep(0.1)
            # GPU work is in flight — kill without any teardown opportunity.
            child.kill()  # SIGKILL
            child.wait(timeout=30.0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30.0)
        assert child.returncode == -signal.SIGKILL

        start = time.monotonic()
        probe = _run_child("probe")
        elapsed = time.monotonic() - start
        _assert_probe_ok(probe, "probe after SIGKILL mid-render")
        assert elapsed < _PROBE_BUDGET_S, (
            f"post-SIGKILL probe took {elapsed:.1f}s (> {_PROBE_BUDGET_S:.0f}s budget) — "
            "abandoned work may be wedging the device"
        )

    def test_atexit_teardown_runs_on_normal_exit(self):
        """The child never calls destroy(); the atexit hook must drain + close
        (observable via the env-gated marker print)."""
        proc = _run_child("atexit", env_extra={mc._TEARDOWN_MARKER_ENV: "1"})
        assert proc.returncode == 0, (
            f"atexit child failed (rc={proc.returncode})\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        assert "CHILD_EXITING" in proc.stdout
        assert mc._TEARDOWN_MARKER in proc.stdout, (
            "atexit teardown marker missing — the hook did not run on normal exit\n"
            f"stdout: {proc.stdout}"
        )
