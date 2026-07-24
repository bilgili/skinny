"""Hostless tests for `skinny.mlt_chain` (change renderer-module-carveout,
Stage A) — the MLT host chain-state module carved out of `renderer.py`.

No GPU, no `Renderer`: the seed derivation, mutation budget, and uniform-tail
predicate are pure, and `run_bootstrap` is driven with a stub pass.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from skinny import mlt_chain


# ── next_seed ────────────────────────────────────────────────────────────────

# Pinned against the pre-carve-out `Renderer._next_mlt_seed` formula. These
# integers are load-bearing: the parity gate re-renders MLT in a FRESH
# interpreter and must get the same chains, so a drift here is a silent
# reproducibility regression, not a cosmetic one.
_PINNED = {
    0: 558161692,
    1: 2583214201,
    2: 2337085335,
    7: 3163809701,
    42: 4006318150,
    1000: 818481298,
}


@pytest.mark.parametrize("frame_index,expected", sorted(_PINNED.items()))
def test_next_seed_pins_exact_pre_carveout_values(frame_index, expected):
    assert mlt_chain.next_seed(frame_index) == expected


def test_next_seed_matches_the_original_formula():
    # The formula itself, spelled out — so a refactor of the module cannot
    # quietly change the mapping while the pinned table is edited to match.
    for i in (0, 3, 99, 65535, 2 ** 31 - 1):
        assert mlt_chain.next_seed(i) == \
            zlib.crc32(struct.pack("<I", i)) & 0xFFFFFFFF


def test_next_seed_masks_frame_index_to_u32():
    # A signed "<i" pack raises struct.error past 2**31; mltSeed is a u32
    # shader field (codex pre-merge review).
    assert mlt_chain.next_seed(2 ** 31) == 3439090748
    assert mlt_chain.next_seed(2 ** 32 - 1) == 4294967295
    assert mlt_chain.next_seed(2 ** 32 + 5) == mlt_chain.next_seed(5)
    assert 0 <= mlt_chain.next_seed(2 ** 40 + 7) <= 0xFFFFFFFF


def test_next_seed_decorrelates_consecutive_resets():
    seeds = [mlt_chain.next_seed(i) for i in range(64)]
    assert len(set(seeds)) == 64


def test_next_seed_is_hash_free():
    # `hash()` is PYTHONHASHSEED-randomized; deriving the seed from it made the
    # same scene score relMSE 0.17 / 0.25 / 1.10 across three runs.
    import inspect
    src = inspect.getsource(mlt_chain.next_seed)
    body = src.split('"""')[-1]
    assert "zlib.crc32" in body
    assert "hash(" not in body
    assert "_current_state_hash" not in body


# ── iterations_per_frame ─────────────────────────────────────────────────────

def test_iterations_is_about_one_mutation_per_pixel():
    assert mlt_chain.iterations_per_frame(256, 256, 16384) == 4
    assert mlt_chain.iterations_per_frame(128, 128, 16384) == 1
    assert mlt_chain.iterations_per_frame(1920, 1080, 16384) == 127


def test_iterations_is_at_least_one():
    # More chains than pixels still runs a mutation (round() would give 0).
    assert mlt_chain.iterations_per_frame(4, 4, 16384) == 1
    assert mlt_chain.iterations_per_frame(0, 0, 16384) == 1


def test_iterations_guards_zero_chains():
    assert mlt_chain.iterations_per_frame(64, 64, 0) == 4096


# ── uniform_tail_active ──────────────────────────────────────────────────────

_MEGA, _WAVE = 0, mlt_chain.EXECUTION_WAVEFRONT


@pytest.mark.parametrize("integrator", [0, 1, 2, 4])
def test_tail_off_for_every_non_mlt_integrator(integrator):
    for is_metal in (False, True):
        assert not mlt_chain.uniform_tail_active(integrator, is_metal, _WAVE, True)


def test_tail_always_on_for_mlt_on_vulkan():
    # One oversized shared UBO: appending the tail is harmless, only the MLT
    # .spv reads the offsets.
    for mode in (_MEGA, _WAVE):
        for built in (False, True):
            assert mlt_chain.uniform_tail_active(3, False, mode, built)


def test_tail_on_metal_needs_wavefront_and_a_built_pass():
    # The Metal blob length must equal the dispatched pipeline's reflected fc.
    assert mlt_chain.uniform_tail_active(3, True, _WAVE, True)
    assert not mlt_chain.uniform_tail_active(3, True, _WAVE, False)
    assert not mlt_chain.uniform_tail_active(3, True, _MEGA, True)
    assert not mlt_chain.uniform_tail_active(3, True, _MEGA, False)


# ── run_bootstrap ────────────────────────────────────────────────────────────

class _StubMlt:
    """Records the host round-trip's call order without a device."""

    def __init__(self, weights, num_chains=8):
        self.num_chains = num_chains
        self.b = -1.0
        self.seeded = True  # must be cleared then re-set by run_bootstrap
        self._weights = np.asarray(weights, dtype=np.float32)
        self.uploaded = None
        self.calls = []

    def read_bootstrap_weights(self):
        self.calls.append(("read", self.b, self.seeded))
        return self._weights

    def upload_chain_seeds(self, seeds):
        self.calls.append(("upload_seeds", len(seeds)))
        self.uploaded = seeds


def _run(mlt, *, seed=7, with_uniforms=True):
    calls = mlt.calls
    upload = (lambda: calls.append(("uniforms", mlt.b))) if with_uniforms else None
    mlt_chain.run_bootstrap(
        mlt, seed=seed, submit=lambda phase: calls.append(("submit", phase)),
        upload_uniforms=upload)
    return calls


def test_run_bootstrap_call_order_on_vulkan():
    mlt = _StubMlt([0.0, 1.0, 3.0])
    calls = _run(mlt)
    assert [c[0] for c in calls] == [
        "uniforms", "submit", "read", "upload_seeds", "submit", "uniforms"]
    assert [c[1] for c in calls if c[0] == "submit"] == ["bootstrap", "init"]


def test_run_bootstrap_clears_state_before_the_dispatches():
    # b/seeded must be cleared BEFORE the uniforms upload — the bootstrap
    # kernel's fc must not carry a stale b, and a mid-flight failure must not
    # leave the pass looking seeded.
    mlt = _StubMlt([1.0, 2.0])
    calls = _run(mlt)
    assert calls[0] == ("uniforms", 0.0)
    assert ("read", 0.0, False) in calls


def test_run_bootstrap_publishes_b_and_seeds():
    w = [0.0, 1.0, 3.0]
    mlt = _StubMlt(w, num_chains=4096)
    _run(mlt, seed=7)
    assert mlt.b == pytest.approx(float(np.mean(w)))
    assert mlt.seeded is True
    assert mlt.uploaded is not None and len(mlt.uploaded) == 4096
    assert np.bincount(mlt.uploaded, minlength=3)[0] == 0  # zero weight never drawn


def test_run_bootstrap_reuploads_uniforms_after_b_is_known():
    # The frame's resolve reads fc.mltB, so the final upload must see the
    # measured b, not 0.
    mlt = _StubMlt([2.0, 2.0])
    calls = _run(mlt)
    assert calls[-1] == ("uniforms", pytest.approx(2.0))


def test_run_bootstrap_without_uniform_uploads_is_the_metal_shape():
    # On Metal the blob is a per-dispatch argument packed inside `submit`.
    mlt = _StubMlt([1.0, 1.0])
    calls = _run(mlt, with_uniforms=False)
    assert [c[0] for c in calls] == ["submit", "read", "upload_seeds", "submit"]
    assert mlt.b == pytest.approx(1.0) and mlt.seeded is True


def test_run_bootstrap_is_deterministic_per_seed():
    w = [1.0, 2.0, 3.0]
    a, b = _StubMlt(w, 256), _StubMlt(w, 256)
    _run(a, seed=42)
    _run(b, seed=42)
    assert np.array_equal(a.uploaded, b.uploaded)
    c = _StubMlt(w, 256)
    _run(c, seed=43)
    assert not np.array_equal(a.uploaded, c.uploaded)


def test_run_bootstrap_propagates_the_all_zero_refusal():
    mlt = _StubMlt([0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="no light-carrying paths"):
        _run(mlt)
    assert mlt.seeded is False, "a failed reseed must not look seeded"
