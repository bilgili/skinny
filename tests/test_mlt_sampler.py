"""Hostless tests for the PSSMLT sampler numpy mirror (change mlt-integrator).

Asserts the pbrt MLTSampler semantics the GPU shader must reproduce; the
shader side is the SKINNY_MLT RNG override in shaders/common.slang, kept in
lockstep with this mirror (source-level constant checks below; bit-agreement
is a GPU-harness follow-up in the transport tasks).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from skinny.sampling.mlt_sampler import (
    MLT_MAX_DIMS,
    MLT_NUM_STREAMS,
    MltSampler,
    erf_inv,
    sample_normal,
)

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _sampler(seed_index=7, sigma=0.01, lsp=0.3) -> MltSampler:
    return MltSampler(seed_index, seed=1, sigma=sigma, large_step_probability=lsp)


# ── initial-state bookkeeping (design D3) ────────────────────────────────────

def test_initial_state_matches_pbrt():
    s = _sampler()
    assert s.current_iteration == 0
    assert s.large_step is True
    assert s.last_large_step_iteration == 0
    # Initial evaluation (no start_iteration): fresh uniform fills, lastMod = 0.
    v = s.get_1d()
    assert 0.0 <= v < 1.0
    assert s.last_mod[0] == 0


def test_stream_interleaving():
    s = _sampler()
    s.start_stream(0)
    i0 = [s._next_index() for _ in range(3)]
    s.start_stream(2)
    i2 = [s._next_index() for _ in range(3)]
    assert i0 == [0, MLT_NUM_STREAMS, 2 * MLT_NUM_STREAMS]
    assert i2 == [2, 2 + MLT_NUM_STREAMS, 2 + 2 * MLT_NUM_STREAMS]


# ── accept/reject restore exactness ─────────────────────────────────────────

def test_reject_restores_exactly():
    s = _sampler(lsp=0.0)  # force small steps so values perturb
    s.start_stream(0)
    before = [float(s.get_1d()) for _ in range(8)]
    state_before = (s.value.copy(), s.last_mod.copy())
    s.start_iteration()
    s.start_stream(0)
    after = [float(s.get_1d()) for _ in range(8)]
    assert after != before
    s.reject()
    np.testing.assert_array_equal(s.value, state_before[0])
    np.testing.assert_array_equal(s.last_mod, state_before[1])
    assert s.current_iteration == 0


def test_accept_keeps_proposal():
    s = _sampler(lsp=0.0)
    s.start_stream(0)
    _ = [s.get_1d() for _ in range(4)]
    s.start_iteration()
    s.start_stream(0)
    proposed = [float(s.get_1d()) for _ in range(4)]
    s.accept()
    s.start_iteration()
    s.start_stream(0)
    # nSmall == 1 for each dim: next values perturb FROM the accepted proposal.
    nxt = [float(s.get_1d()) for _ in range(4)]
    assert all(a != b for a, b in zip(proposed, nxt))


# ── large-step lazy reset ────────────────────────────────────────────────────

def test_large_step_lazily_refreshes_untouched_dims():
    s = _sampler(lsp=1.0)  # every iteration is a large step
    s.start_stream(0)
    v0 = float(s.get_1d())
    s.start_iteration()  # large step
    s.accept()           # lastLargeStepIteration = 1
    assert s.last_large_step_iteration == 1
    # Dim 0 untouched during iteration 1. Next read at iteration 2 must first
    # lazily refresh it (lastMod 0 < lastLargeStep 1) before mutating.
    s.start_iteration()
    s.start_stream(0)
    v1 = float(s.get_1d())
    assert v1 != v0
    assert s.last_mod[0] == s.current_iteration


def test_aggregated_small_step_sigma():
    # A dim untouched for n iterations gets ONE perturbation with sigma*sqrt(n).
    sigma, n = 0.05, 16
    deltas = []
    for seed in range(600):
        s = MltSampler(seed, seed=2, sigma=sigma, large_step_probability=0.0)
        s.start_stream(0)
        v0 = float(s.get_1d())
        for _ in range(n):
            s.start_iteration()
            s.accept()  # small-step accepts don't move lastLargeStep
        s.start_stream(0)
        v1 = float(s.get_1d())
        d = v1 - v0
        d -= round(d)  # unwrap [0,1) torus
        deltas.append(d)
    measured = np.std(deltas)
    expected = sigma * math.sqrt(n)
    assert measured == pytest.approx(expected, rel=0.15)


# ── erfInv / sampleNormal ────────────────────────────────────────────────────

def test_erf_inv_inverts_erf():
    for x in (-0.9, -0.5, -0.1, 0.0, 0.2, 0.7, 0.95):
        assert math.erf(float(erf_inv(x))) == pytest.approx(x, abs=1e-5)


def test_sample_normal_distribution():
    rng = np.random.default_rng(0)
    us = rng.random(20000)
    xs = np.array([float(sample_normal(u, 0.0, 1.0)) for u in us])
    assert np.mean(xs) == pytest.approx(0.0, abs=0.03)
    assert np.std(xs) == pytest.approx(1.0, abs=0.03)


# ── budget invariant ─────────────────────────────────────────────────────────

def test_budget_overflow_asserts():
    s = _sampler()
    with pytest.raises(AssertionError):
        s._ensure_ready(MLT_MAX_DIMS)


# ── shader lockstep (source-level) ───────────────────────────────────────────

def test_shader_constants_match_mirror():
    # The PSS machinery lives in common.slang as the SKINNY_MLT RNG override
    # (design D2): under -DSKINNY_MLT rng.next() serves lazily-mutated primary
    # samples, so the whole bdpt transport tree becomes PSS-driven unmodified;
    # the #else branch is byte-identical to the shipped RNG (RGB .spv unchanged).
    src = (_SRC / "shaders/common.slang").read_text()
    assert "#if defined(SKINNY_MLT)" in src
    m = re.search(r"MLT_MAX_DIMS\s*=\s*(\d+)u", src)
    assert m and int(m.group(1)) == MLT_MAX_DIMS
    m = re.search(r"MLT_NUM_STREAMS\s*=\s*(\d+)u", src)
    assert m and int(m.group(1)) == MLT_NUM_STREAMS
    # Reject-restore / lazy-reset / stream machinery markers exist.
    for token in ("void reject()", "void accept()", "void startIteration(",
                  "void startStream(", "lastLargeStepIteration", "mltErfInv",
                  "valueBackup", "mltCreateSampler"):
        assert token in src, f"{token} missing from common.slang SKINNY_MLT block"
