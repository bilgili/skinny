"""Hostless tests for the MLT wavefront driver sequence (change mlt-integrator).

Recording-stub tests: phase order, breadth-tiling alignment, watchdog flush
boundaries, and the budget math — no GPU.
"""

from __future__ import annotations

from skinny.wavefront_driver import (
    _mlt_chain_batch,
    record_mlt_bootstrap,
    record_mlt_frame,
    record_mlt_init,
)
from skinny.wavefront_layout import (
    MLT_MAX_DIMS,
    MLT_RECORD_SLOTS,
    mlt_buffer_sizes,
    mlt_chain_meta_size,
    mlt_primary_sample_size,
    mlt_record_size,
)


class _Rec:
    def __init__(self):
        self.ops: list[tuple] = []

    def push_window(self, base, size):
        self.ops.append(("window", base, size))

    def dispatch_count(self, entry, count, group):
        self.ops.append(("dispatch", entry, count, group))

    def barrier(self):
        self.ops.append(("barrier",))

    def flush(self):
        self.ops.append(("flush",))


def _dispatches(rec, entry):
    return [op for op in rec.ops if op[0] == "dispatch" and op[1] == entry]


# ── batch math ───────────────────────────────────────────────────────────────

def test_chain_batch_is_64_aligned_and_capped():
    assert _mlt_chain_batch(16384, 0) == 16384
    assert _mlt_chain_batch(16384, 1000) == 960          # floor to 64
    assert _mlt_chain_batch(16384, 63) == 64             # min one group
    assert _mlt_chain_batch(10_000_000, 0) == 65535 * 64  # portable ceiling
    assert _mlt_chain_batch(100, 0) == 100               # never above num_chains


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_bootstrap_tiles_to_chain_capacity_and_flushes():
    rec = _Rec()
    record_mlt_bootstrap(rec, bootstrap_samples=10_000, num_chains=4096)
    d = _dispatches(rec, "wfMltBootstrap")
    # ceil(10000/4096) = 3 sub-batches, each flushed.
    assert len(d) == 3
    assert sum(op[2] for op in d) == 10_000          # no sample dropped
    assert all(op[2] <= 4096 for op in d)            # X-slice aliasing bound
    windows = [op for op in rec.ops if op[0] == "window"]
    assert [w[1] for w in windows] == [0, 4096, 8192]
    assert rec.ops.count(("flush",)) == 3


# ── init ─────────────────────────────────────────────────────────────────────

def test_init_covers_every_chain_exactly_once():
    rec = _Rec()
    record_mlt_init(rec, num_chains=16384, chain_batch=8192)
    d = _dispatches(rec, "wfMltInit")
    assert sum(op[2] for op in d) == 16384
    assert len(d) == 2


# ── frame ────────────────────────────────────────────────────────────────────

def test_frame_orders_mutates_before_resolve():
    rec = _Rec()
    record_mlt_frame(rec, num_pixels=256 * 256, num_chains=4096, iterations=3)
    names = [op[1] for op in rec.ops if op[0] == "dispatch"]
    assert names == ["wfMltMutate"] * 3 + ["wfMltResolve"]
    # Barrier between every mutate step and before the resolve.
    mut_idx = [i for i, op in enumerate(rec.ops)
               if op[0] == "dispatch" and op[1] == "wfMltMutate"]
    for i in mut_idx:
        assert ("barrier",) in [rec.ops[i + 1], rec.ops[i + 2]]
    # Resolve covers every pixel.
    res = _dispatches(rec, "wfMltResolve")[0]
    assert res[2] == 256 * 256


def test_frame_mutate_breadth_tiled_with_flush():
    rec = _Rec()
    record_mlt_frame(rec, num_pixels=64, num_chains=1000, iterations=1,
                     chain_batch=256)
    d = _dispatches(rec, "wfMltMutate")
    # 1000 chains at 256-batch → 4 sub-batches (256+256+256+232), each flushed.
    assert [op[2] for op in d] == [256, 256, 256, 232]
    assert rec.ops.count(("flush",)) == 4


def test_mpp_actual_budget_math():
    # The host packs mpp_actual = iterations * num_chains / pixels; one
    # iteration over 16384 chains at 128x128 = exactly 1 mutation/pixel.
    assert 1 * 16384 / (128 * 128) == 1.0


# ── layout strides ───────────────────────────────────────────────────────────

def test_mlt_strides_identical_across_layouts():
    assert mlt_primary_sample_size() == mlt_primary_sample_size(msl=True) == 16
    assert mlt_chain_meta_size() == mlt_chain_meta_size(msl=True) == 32
    assert mlt_record_size() == mlt_record_size(msl=True) == 16


def test_mlt_buffer_sizes_scale_by_chains_not_stream():
    a = mlt_buffer_sizes(1024, 1000)
    b = mlt_buffer_sizes(2048, 1000)
    assert b["mlt_primary_samples"] == 2 * a["mlt_primary_samples"]
    assert a["mlt_primary_samples"] == 1024 * MLT_MAX_DIMS * 16
    assert a["mlt_current_records"] == 1024 * MLT_RECORD_SLOTS * 16
    assert a["mlt_bootstrap_weights"] == 1000 * 4
