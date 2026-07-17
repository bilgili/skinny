"""Hostless tests for the MLT host wiring (change mlt-integrator).

Two halves, no GPU:
- the pure bootstrap resample (``skinny.mlt_bootstrap.resample_chain_seeds``:
  b math, weight-proportional seeding, zero-weight refusal, determinism);
- source-level guards that the Vulkan renderer wiring exists (the ensure/
  destroy/dispatch seam, the MLT uniform tail, the Metal refusal) — the
  ``test_mlt_selection`` grep-style pattern, since constructing a Renderer
  needs a GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from skinny.mlt_bootstrap import resample_chain_seeds

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text()


# ── resample_chain_seeds ─────────────────────────────────────────────────────

def test_b_is_mean_of_weights():
    w = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float32)
    b, _ = resample_chain_seeds(w, 8, seed=1)
    assert b == pytest.approx(float(w.mean()))


def test_seeds_proportional_to_weights():
    # Index 2 carries 3× index 1's weight; index 0 must never be drawn.
    w = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    _, seeds = resample_chain_seeds(w, 40_000, seed=7)
    assert seeds.dtype == np.uint32 and seeds.shape == (40_000,)
    counts = np.bincount(seeds, minlength=3)
    assert counts[0] == 0
    assert counts[2] / counts[1] == pytest.approx(3.0, rel=0.05)


def test_deterministic_per_seed():
    w = np.array([1.0, 2.0, 3.0])
    _, a = resample_chain_seeds(w, 128, seed=42)
    _, b = resample_chain_seeds(w, 128, seed=42)
    _, c = resample_chain_seeds(w, 128, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_all_zero_weights_raise():
    with pytest.raises(RuntimeError, match="no light-carrying paths"):
        resample_chain_seeds(np.zeros(16), 4, seed=0)


def test_nonfinite_only_weights_raise():
    w = np.array([np.nan, np.inf, -1.0, 0.0])
    with pytest.raises(RuntimeError, match="no light-carrying paths"):
        resample_chain_seeds(w, 4, seed=0)


def test_nonfinite_entries_zeroed_not_drawn():
    w = np.array([np.nan, 2.0, np.inf])
    b, seeds = resample_chain_seeds(w, 1000, seed=3)
    assert b == pytest.approx(2.0 / 3.0)  # non-finite zeroed before the mean
    assert set(np.unique(seeds)) == {1}


# ── renderer wiring (source-level, no GPU) ───────────────────────────────────

def test_renderer_has_mlt_pass_lifecycle():
    src = _read("renderer.py")
    for sym in ("_ensure_wavefront_mlt_pass", "_destroy_wavefront_mlt_pass",
                "_run_wavefront_mlt_bootstrap", "resample_chain_seeds",
                "WavefrontMltPass"):
        assert sym in src, f"renderer.py must reference {sym}"


def test_renderer_dispatch_branch_selects_mlt():
    src = _read("renderer.py")
    start = src.index("def _record_wavefront_dispatch")
    body = src[start:start + 4000]
    assert "integrator_index == 3" in body, \
        "_record_wavefront_dispatch must branch on the MLT integrator index"
    assert "_ensure_wavefront_mlt_pass" in body
    assert "record_frame" in body


def test_renderer_packs_mlt_uniform_tail_in_shader_order():
    # The SKINNY_MLT FrameConstants tail: 4 floats then 4 uints. It must be
    # packed BEFORE the trailing tileOriginY filler u32 — `tileOriginY` is
    # SKINNY_METAL-gated, so in the Vulkan MLT SPIR-V `mltSigma` sits at
    # offset 564 right after sppmGroupPmfEnv (spirv-cross-verified); a tail
    # packed after the filler would shift every MLT field by +4 bytes.
    src = _read("renderer.py")
    start = src.index("def _pack_uniforms(")
    body = src[start:]
    for field in ("mlt_sigma", "mlt_large_step_prob", "mlt_max_depth",
                  "_mlt_seed"):
        assert field in body, f"_pack_uniforms must pack {field}"
    # 4f then 4I — the exact common.slang field order.
    m_f = re.search(r'"ffff",\s*float\(self\.mlt_sigma\)', body)
    m_i = re.search(r'"IIII",\s*chains', body)
    assert m_f and m_i and m_f.start() < m_i.start()
    filler = body.index("tileOriginY (Metal band loop patches)")
    assert m_i.end() < filler, \
        "MLT uniform tail must precede the tileOriginY filler u32"


def test_metal_dispatch_refuses_mlt_loudly():
    src = _read("renderer.py")
    start = src.index("def _render_scene_metal")
    body = src[start:start + 3000]
    assert "NotImplementedError" in body and "mlt-integrator" in body, \
        "Metal dispatch must refuse MLT explicitly (adapter pending)"


def test_vk_wavefront_has_mlt_pass_and_recorder():
    src = _read("vk_wavefront.py")
    assert "class WavefrontMltPass" in src
    assert "class _VkMltRecorder" in src
    assert '"SKINNY_MLT=1"' in src, "MLT kernels must compile under -DSKINNY_MLT=1"
    assert 'tag="_mlt"' in src, "MLT .spv names must never alias the RGB kernels"
    for entry in ("wfMltBootstrap", "wfMltInit", "wfMltMutate", "wfMltResolve"):
        assert entry in src


def test_scene_layout_declares_mlt_bindings_for_wavefront():
    src = _read("vk_compute.py")
    assert "mlt_bindings" in src
    assert "(52, 53, 54, 55, 56)" in src, \
        "the wavefront scene set-0 layout must declare MLT bindings 52-56"
