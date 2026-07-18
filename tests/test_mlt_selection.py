"""Selection-seam invariants for the MLT integrator (change mlt-integrator).

Source-level / no-GPU guards: MLT must be registered end-to-end (renderer list,
CLI index map, auto→wavefront derivation, headless map) and refused outside its
envelope. Nothing here constructs a Renderer or touches a GPU.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from skinny import cli_common, mlt_capability
from skinny.cli_common import (
    DEFAULT_EXECUTION_FOR_INTEGRATOR,
    INTEGRATOR_INDEX,
    reject_mlt_unsupported,
    resolve_execution_mode,
    startup_integrator_name,
    validate_render_flags,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text()


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


# ── Registration ─────────────────────────────────────────────────────────────

def test_renderer_registers_mlt_mode():
    src = _read("renderer.py")
    m = re.search(r"integrator_modes:\s*list\[str\]\s*=\s*\[([^\]]*)\]", src)
    assert m, "integrator_modes list not found in renderer.py"
    modes = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    assert modes[:4] == ["Path", "BDPT", "SPPM", "MLT"], modes


def test_integrator_index_registers_mlt():
    assert INTEGRATOR_INDEX["mlt"] == 3
    # The reverse map used by startup_integrator_name must round-trip.
    assert startup_integrator_name(None, 3) == "mlt"


def test_headless_integrator_map_includes_mlt():
    src = _read("headless.py")
    m = re.search(r"_INTEGRATORS\s*=\s*\{([^}]*)\}", src)
    assert m and '"mlt": 3' in m.group(1), "headless _INTEGRATORS must map mlt to 3"


def test_cli_choices_include_mlt():
    parser = argparse.ArgumentParser()
    cli_common.add_render_flags(parser)
    ns = parser.parse_args(["--integrator", "mlt"])
    assert ns.integrator == "mlt"


# ── auto execution-mode derivation ───────────────────────────────────────────

def test_mlt_auto_derives_wavefront():
    assert DEFAULT_EXECUTION_FOR_INTEGRATOR["mlt"] == "wavefront"
    assert resolve_execution_mode("auto", "mlt") == "wavefront"
    assert resolve_execution_mode(None, "mlt") == "wavefront"
    # An explicit mode still wins (and is then refused by the guard, below).
    assert resolve_execution_mode("megakernel", "mlt") == "megakernel"


# ── Capability gate (shipped ON since the Vulkan transport landed) ──────────

def test_mlt_gate_is_on_and_accepted():
    # The transport is wired (change mlt-integrator): an in-envelope
    # `--integrator mlt` session is accepted, no monkeypatch needed.
    assert mlt_capability.MLT_IMPLEMENTED is True
    reject_mlt_unsupported("mlt", "wavefront")
    validate_render_flags(_ns(integrator="mlt", execution_mode="wavefront"))


def test_mlt_refused_when_gate_monkeypatched_off(monkeypatch):
    # The gate is referenced live, so flipping it back off (rollback path)
    # restores the clean refusal instead of silently rendering another
    # integrator.
    monkeypatch.setattr(mlt_capability, "MLT_IMPLEMENTED", False)
    with pytest.raises(SystemExit, match="not yet implemented"):
        reject_mlt_unsupported("mlt", "wavefront")
    with pytest.raises(SystemExit, match="not yet implemented"):
        validate_render_flags(_ns(integrator="mlt", execution_mode="wavefront"))


def test_non_mlt_integrators_unaffected():
    reject_mlt_unsupported("path", "megakernel")
    reject_mlt_unsupported("sppm", "wavefront")
    reject_mlt_unsupported(None, "megakernel")
    validate_render_flags(_ns(integrator="path", execution_mode="megakernel"))


# ── Refusals (capability gate on — the post-wiring envelope) ────────────────

@pytest.fixture()
def _implemented(monkeypatch):
    monkeypatch.setattr(mlt_capability, "MLT_IMPLEMENTED", True)


def test_mlt_alone_clean_when_implemented(_implemented):
    reject_mlt_unsupported("mlt", "wavefront")
    validate_render_flags(_ns(integrator="mlt", execution_mode="wavefront"))


def test_mlt_explicit_megakernel_refused(_implemented):
    with pytest.raises(SystemExit, match="no megakernel path"):
        reject_mlt_unsupported("mlt", "megakernel")


def test_persisted_mlt_explicit_megakernel_refused(_implemented):
    # The interactive front-ends re-check the persisted case through the same
    # function, keyed on the effective startup integrator.
    integ = startup_integrator_name(None, 3)
    with pytest.raises(SystemExit, match="no megakernel path"):
        reject_mlt_unsupported(integ, resolve_execution_mode("megakernel", integ))


def test_mlt_spectral_accepted(_implemented):
    # change spectral-mlt: --spectral --integrator mlt is in-envelope (the target
    # function becomes the spectral BDPT estimator). No raise on wavefront.
    reject_mlt_unsupported("mlt", "wavefront", spectral=True)
    # still wavefront-only, even with spectral.
    with pytest.raises(SystemExit, match="no megakernel path"):
        reject_mlt_unsupported("mlt", "megakernel", spectral=True)


def test_mlt_neural_proposal_refused(_implemented):
    with pytest.raises(SystemExit, match="BSDF proposal"):
        reject_mlt_unsupported("mlt", "wavefront", proposals="bsdf,neural")


def test_mlt_restir_reuse_refused(_implemented):
    with pytest.raises(SystemExit, match="reuse"):
        reject_mlt_unsupported("mlt", "wavefront", reuse="restir_di")


def test_mlt_online_training_refused(_implemented):
    with pytest.raises(SystemExit, match="online-training"):
        reject_mlt_unsupported("mlt", "wavefront", online_training=True)


def test_validate_render_flags_routes_mlt_envelope(_implemented):
    # A still-refused axis (neural proposal) routes through validate_render_flags;
    # spectral is now accepted (change spectral-mlt) so it can't be the probe.
    ns = _ns(integrator="mlt", execution_mode="wavefront",
             spectral=False, proposals="bsdf,neural", reuse=None, online_training=False)
    with pytest.raises(SystemExit, match="BSDF proposal"):
        validate_render_flags(ns)
    # spectral mlt passes the envelope router (wavefront, flat, bsdf-only).
    ok = _ns(integrator="mlt", execution_mode="wavefront",
             spectral=True, proposals=None, reuse=None, online_training=False)
    validate_render_flags(ok)


def test_bsdf_only_proposals_accepted(_implemented):
    reject_mlt_unsupported("mlt", "wavefront", proposals="bsdf", reuse="none")


# ── Front-end wiring is present (source-level) ──────────────────────────────

def test_interactive_front_ends_recheck_persisted_mlt():
    for rel in ("app.py", "ui/qt/app.py"):
        src = _read(rel)
        assert "reject_mlt_unsupported(" in src, \
            f"{rel} must re-check the persisted-mlt case via reject_mlt_unsupported"


def test_state_hash_still_includes_integrator_index():
    src = _read("renderer.py")
    start = src.index("def _current_state_hash")
    assert "self.integrator_index" in src[start:start + 2000]
