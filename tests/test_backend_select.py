"""Backend-selection unit tests (no GPU — the device probe is mocked).

Covers the shared ``--backend`` flag surface and the ``select_backend`` /
``make_context`` resolver in :mod:`skinny.backend_select`. ``auto`` resolves to
Metal on a Metal-capable host (mocked) and to Vulkan elsewhere.
"""

from __future__ import annotations

import argparse

import pytest

from skinny import backend_select as bs
from skinny.backend_select import select_backend
from skinny.cli_common import add_render_flags


def _parser(**kw) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_render_flags(p, **kw)
    return p


# ── the --backend flag on the shared surface ─────────────────────────

def test_backend_flag_default_is_sentinel_none(monkeypatch):
    # default None = "use env / persisted / auto", resolved by select_backend.
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    ns = _parser().parse_args([])
    assert ns.backend is None


@pytest.mark.parametrize("value", ["auto", "metal", "vulkan"])
def test_backend_flag_accepts_choices(value):
    ns = _parser().parse_args(["--backend", value])
    assert ns.backend == value


def test_backend_flag_rejects_bad():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--backend", "cuda"])


def test_backend_flag_can_be_suppressed():
    ns = _parser(backend=False).parse_args([])
    assert not hasattr(ns, "backend")


def test_backend_flag_choices_and_default_identical_across_frontends():
    # Every front-end builds --backend from the one add_render_flags definition,
    # so the choices + default are identical by construction. Assert the shape.
    action = next(
        a for a in _parser()._actions if getattr(a, "dest", None) == "backend"
    )
    assert tuple(action.choices) == ("auto", "metal", "vulkan")
    assert action.default is None


# ── select_backend: precedence + auto→Metal/Vulkan resolution ────────

def test_auto_resolves_metal_when_available(monkeypatch):
    # auto → Metal on a Metal-capable host.
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    monkeypatch.setattr(bs, "metal_available", lambda: (True, ""))
    assert select_backend(None) == "metal"
    assert select_backend("auto") == "metal"


def test_auto_resolves_vulkan_when_metal_unavailable(monkeypatch):
    # auto → Vulkan when no Metal device constructs (non-Apple-Silicon, etc.).
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    monkeypatch.setattr(bs, "metal_available", lambda: (False, "not arm64 macOS"))
    assert select_backend(None) == "vulkan"
    assert select_backend("auto") == "vulkan"


def test_explicit_vulkan_resolves_vulkan(monkeypatch):
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    assert select_backend("vulkan") == "vulkan"


def test_explicit_metal_resolves_metal_when_available(monkeypatch):
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    monkeypatch.setattr(bs, "metal_available", lambda: (True, ""))
    assert select_backend("metal") == "metal"


def test_explicit_metal_raises_clearly_when_unavailable(monkeypatch):
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    monkeypatch.setattr(
        bs, "metal_available", lambda: (False, "native Metal requires Apple Silicon (arm64)")
    )
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        select_backend("metal")


def test_precedence_explicit_beats_env_persisted(monkeypatch):
    monkeypatch.setenv("SKINNY_BACKEND", "metal")
    # explicit flag wins over both env and persisted.
    assert select_backend("vulkan", persisted="metal") == "vulkan"


def test_precedence_env_beats_persisted(monkeypatch):
    monkeypatch.setenv("SKINNY_BACKEND", "vulkan")
    assert select_backend(None, persisted="metal") == "vulkan"


def test_precedence_persisted_beats_auto(monkeypatch):
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    assert select_backend(None, persisted="vulkan") == "vulkan"


def test_unknown_backend_raises_value_error(monkeypatch):
    monkeypatch.delenv("SKINNY_BACKEND", raising=False)
    with pytest.raises(ValueError, match="unknown backend"):
        select_backend("opengl")


# ── make_context dispatch (no GPU for the error path) ────────────────

def test_make_context_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        bs.make_context("opengl")
