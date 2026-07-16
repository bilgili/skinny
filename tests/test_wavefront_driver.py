"""GPU-free tests for the backend-neutral wavefront stage loop.

``skinny.wavefront_driver.record_path_loop`` holds the single source of truth for
the staged path tracer's stage order, decoupled from any GPU backend. A recording
stub captures the exact primitive sequence so the loop can be pinned without a
device — and any drift between the Vulkan and Metal adapters' shared order is
caught here, not at convergence.
"""

from __future__ import annotations

import pytest

from skinny.wavefront_driver import record_bdpt_loop, record_path_loop


class _StubRecorder:
    """Records the primitive op sequence ``record_path_loop`` emits."""

    def __init__(self, *, stream_size=64, has_neural=False, has_restir=False):
        self._stream_size = stream_size
        self._has_neural = has_neural
        self._has_restir = has_restir
        self.ops: list = []

    @property
    def stream_size(self):
        return self._stream_size

    @property
    def has_neural(self):
        return self._has_neural

    @property
    def has_restir(self):
        return self._has_restir

    def barrier(self):
        self.ops.append(("barrier",))

    def clear_counts(self):
        self.ops.append(("clear_counts",))

    def push_tile(self, stream_base):
        self.ops.append(("push_tile", stream_base))

    def dispatch_full(self, entry):
        self.ops.append(("dispatch_full", entry))

    def dispatch_one(self, entry):
        self.ops.append(("dispatch_one", entry))

    def shade(self, slot, entry):
        self.ops.append(("shade", slot, entry))

    def neural_prepass(self):
        self.ops.append(("neural_prepass",))

    def restir_primary_direct(self):
        self.ops.append(("restir_primary_direct",))

    def flush_heavy_eye(self):
        self.ops.append(("flush_heavy_eye",))


def _bounce_core():
    """The intersect→build_args→scatter prefix every bounce starts with."""
    return [
        ("clear_counts",),
        ("barrier",),
        ("dispatch_full", "wfPathIntersect"),
        ("barrier",),
        ("dispatch_one", "wfBuildArgs"),
        ("barrier",),
        ("dispatch_full", "wfScatter"),
        ("barrier",),
    ]


def test_single_tile_single_bounce_flat_only():
    rec = _StubRecorder(stream_size=64)
    record_path_loop(rec, num_pixels=64, stream_size=64, max_bounces=1, build_catchall=False)
    assert rec.ops == [
        ("push_tile", 0),
        ("dispatch_full", "wfPathGenerate"),
        *_bounce_core(),
        ("shade", 0, "wfPathShadeFlat"),
        ("barrier",),
        ("dispatch_full", "wfPathResolve"),
    ]


def test_catchall_adds_second_shade_with_barrier():
    rec = _StubRecorder(stream_size=64)
    record_path_loop(rec, num_pixels=64, stream_size=64, max_bounces=1, build_catchall=True)
    # Flat shade, barrier, catch-all shade — then the resolve barrier + resolve.
    assert rec.ops[-5:] == [
        ("shade", 0, "wfPathShadeFlat"),
        ("barrier",),
        ("shade", 1, "wfPathShade"),
        ("barrier",),
        ("dispatch_full", "wfPathResolve"),
    ]


def test_two_tiles_emit_leading_barrier_on_second_tile():
    rec = _StubRecorder(stream_size=64)
    record_path_loop(rec, num_pixels=128, stream_size=64, max_bounces=1, build_catchall=False)
    # First tile starts with push_tile(0) and NO leading barrier.
    assert rec.ops[0] == ("push_tile", 0)
    # The second tile is preceded by exactly one barrier, then push_tile(64).
    i = rec.ops.index(("push_tile", 64))
    assert rec.ops[i - 1] == ("barrier",)
    assert rec.ops[i + 1] == ("dispatch_full", "wfPathGenerate")


def test_neural_prepass_inserted_every_bounce_with_retile():
    rec = _StubRecorder(stream_size=64, has_neural=True)
    record_path_loop(rec, num_pixels=64, stream_size=64, max_bounces=2, build_catchall=False)
    # Each bounce: after scatter+barrier → neural_prepass, barrier, push_tile, then shade.
    assert rec.ops.count(("neural_prepass",)) == 2
    for ni in [i for i, op in enumerate(rec.ops) if op == ("neural_prepass",)]:
        assert rec.ops[ni - 1] == ("barrier",)            # after scatter barrier
        assert rec.ops[ni + 1] == ("barrier",)
        assert rec.ops[ni + 2] == ("push_tile", 0)
        assert rec.ops[ni + 3] == ("shade", 0, "wfPathShadeFlat")


def test_restir_runs_only_at_bounce_zero():
    rec = _StubRecorder(stream_size=64, has_restir=True)
    record_path_loop(rec, num_pixels=64, stream_size=64, max_bounces=3, build_catchall=False)
    assert rec.ops.count(("restir_primary_direct",)) == 1
    ri = rec.ops.index(("restir_primary_direct",))
    assert rec.ops[ri + 1] == ("barrier",)
    assert rec.ops[ri + 2] == ("push_tile", 0)
    # ReSTIR appears in the first bounce, before the first shade.
    first_shade = rec.ops.index(("shade", 0, "wfPathShadeFlat"))
    assert ri < first_shade


# ── record_bdpt_loop (phase 4) ──────────────────────────────────────


def _bdpt_compact(classify):
    """clear → classify → build_args → scatter, each barrier-separated."""
    return [
        ("clear_counts",),
        ("dispatch_full", classify),
        ("barrier",),
        ("dispatch_one", "wfBdptBuildArgs"),
        ("barrier",),
        ("dispatch_full", "wfBdptScatter"),
        ("barrier",),
    ]


def _bdpt_tail():
    """The connect + resolve sequence every walk mode shares."""
    return [
        *_bdpt_compact("wfBdptClassify"),
        ("shade", 0, "wfBdptConnectNee"),
        ("barrier",),
        ("shade", 1, "wfBdptConnectFull"),
        ("barrier",),
        ("dispatch_full", "wfBdptResolve"),
    ]


def _staged_bounce(entry, bounces):
    out = []
    for _ in range(bounces):
        out += [*_bdpt_compact("wfBdptWalkClassify"), ("shade", 0, entry), ("barrier",)]
    return out


def test_bdpt_fused_single_tile():
    rec = _StubRecorder(stream_size=64)
    record_bdpt_loop(rec, num_pixels=64, stream_size=64, walk_mode="fused",
                     eye_bounces=5, light_bounces=6)
    assert rec.ops == [
        ("push_tile", 0),
        ("dispatch_full", "wfBdptWalk"),
        ("barrier",),
        *_bdpt_tail(),
        ("flush_heavy_eye",),
    ]


def test_bdpt_eye_staged_walk_plus_fused_light_tail():
    rec = _StubRecorder(stream_size=64)
    record_bdpt_loop(rec, num_pixels=64, stream_size=64, walk_mode="eye",
                     eye_bounces=2, light_bounces=6)
    assert rec.ops == [
        ("push_tile", 0),
        ("dispatch_full", "wfBdptGenEye"),
        ("barrier",),
        *_staged_bounce("wfBdptBounceEye", 2),
        ("dispatch_full", "wfBdptLightTail"),
        ("barrier",),
        *_bdpt_tail(),
        ("flush_heavy_eye",),
    ]


def test_bdpt_eye_light_fully_staged():
    rec = _StubRecorder(stream_size=64)
    record_bdpt_loop(rec, num_pixels=64, stream_size=64, walk_mode="eye_light",
                     eye_bounces=2, light_bounces=3)
    assert rec.ops == [
        ("push_tile", 0),
        ("dispatch_full", "wfBdptGenEye"),
        ("barrier",),
        *_staged_bounce("wfBdptBounceEye", 2),
        ("dispatch_full", "wfBdptGenLight"),
        ("barrier",),
        *_staged_bounce("wfBdptBounceLight", 3),
        ("dispatch_full", "wfBdptSplat"),
        ("barrier",),
        *_bdpt_tail(),
        ("flush_heavy_eye",),
    ]


def test_bdpt_two_tiles_leading_barrier_on_second():
    rec = _StubRecorder(stream_size=64)
    record_bdpt_loop(rec, num_pixels=128, stream_size=64, walk_mode="fused",
                     eye_bounces=5, light_bounces=6)
    assert rec.ops[0] == ("push_tile", 0)
    i = rec.ops.index(("push_tile", 64))
    assert rec.ops[i - 1] == ("barrier",)                 # leading barrier on tile 2
    assert rec.ops[i - 2] == ("flush_heavy_eye",)          # per-tile eye submit bound
    assert rec.ops[i - 3] == ("dispatch_full", "wfBdptResolve")
    assert rec.ops[i + 1] == ("dispatch_full", "wfBdptWalk")
    assert rec.ops.count(("flush_heavy_eye",)) == 2       # one per tile (2 tiles)


def test_bdpt_rejects_unknown_walk_mode():
    rec = _StubRecorder(stream_size=64)
    with pytest.raises(ValueError, match="walk_mode"):
        record_bdpt_loop(rec, num_pixels=64, stream_size=64, walk_mode="nope",
                         eye_bounces=5, light_bounces=6)


# ── record_sppm_loop (change photon-mapping-sppm) ───────────────────

from skinny.wavefront_driver import record_sppm_loop  # noqa: E402


class _SppmStub:
    """Records the primitive op sequence record_sppm_loop emits."""

    def __init__(self, *, stream_size=64):
        self._stream_size = stream_size
        self.ops: list = []

    @property
    def stream_size(self):
        return self._stream_size

    def barrier(self):
        self.ops.append(("barrier",))

    def push_tile(self, stream_base):
        self.ops.append(("push_tile", stream_base))

    def dispatch_full(self, entry):
        self.ops.append(("dispatch_full", entry))

    def dispatch_one(self, entry):
        self.ops.append(("dispatch_one", entry))

    def dispatch_count(self, entry, count, group_size):
        self.ops.append(("dispatch_count", entry, count, group_size))

    def clear_visible_points(self):
        self.ops.append(("clear_visible_points",))

    def clear_grid(self):
        self.ops.append(("clear_grid",))

    def clear_accum(self):
        self.ops.append(("clear_accum",))

    def flush_heavy_eye(self):
        self.ops.append(("flush_heavy_eye",))

    def flush(self):
        # Phase-boundary + per-photon-batch command-buffer submit (Metal watchdog
        # hygiene; no-op off Metal). Recorded so the driver's flush placement is
        # pinned hostlessly (changes spectral-wavefront + sppm-photon-dispatch-tiling).
        self.ops.append(("flush",))


def _photon_block(photons, batch=0):
    """Phase-3: clear the accumulator ONCE, then trace the photon budget as
    breadth-tiled sub-batches (change sppm-photon-dispatch-tiling). ``batch<=0``
    is the degenerate single full dispatch (Vulkan / no watchdog)."""
    b = batch if batch and batch > 0 else photons
    ops = [("clear_accum",), ("barrier",)]
    base = 0
    while base < photons:
        n = min(b, photons - base)
        ops += [
            ("push_tile", base),
            ("dispatch_count", "wfSppmPhotonTrace", n, 64),
            ("barrier",),
            ("flush",),
        ]
        base += n
    return ops


def _sppm_grid_photon(num_pixels, num_cells, photons, batch=0):
    """The single global grid-build + photon block shared by every pass. The
    grid stage ends with a phase-boundary flush; the photon stage is tiled."""
    return [
        ("clear_grid",),
        ("barrier",),
        ("dispatch_count", "wfSppmGridCount", num_pixels, 64),
        ("barrier",),
        ("dispatch_count", "wfSppmGridScanBlock", num_cells, 256),
        ("barrier",),
        ("dispatch_one", "wfSppmGridScanBlockSums"),
        ("barrier",),
        ("dispatch_count", "wfSppmGridScanAdd", num_cells, 256),
        ("barrier",),
        ("dispatch_count", "wfSppmGridScatter", num_pixels, 64),
        ("barrier",),
        ("flush",),                     # phase 2 → 3 boundary
        *_photon_block(photons, batch),
    ]


def test_sppm_single_tile_first_frame():
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=100, first_frame=True)
    assert rec.ops == [
        ("clear_visible_points",),
        ("barrier",),
        ("push_tile", 0),
        ("dispatch_full", "wfSppmEye"),
        ("flush_heavy_eye",),
        ("barrier",),
        ("flush",),                     # phase 1 → 2 boundary
        *_sppm_grid_photon(64, 256, 100),
        ("push_tile", 0),
        ("dispatch_full", "wfSppmUpdate"),
        ("barrier",),
    ]


def test_sppm_later_frame_skips_vp_clear():
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=100, first_frame=False)
    assert ("clear_visible_points",) not in rec.ops
    assert rec.ops[0] == ("push_tile", 0)
    assert rec.ops[1] == ("dispatch_full", "wfSppmEye")


def test_sppm_grid_and_photon_run_once_globally():
    # Grid build + accumulator clear must appear EXACTLY once regardless of tile
    # count (photon dispatch may tile — clear_accum must not).
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=256, stream_size=64, num_cells=1024,
                     photons=500, first_frame=False)
    assert rec.ops.count(("clear_grid",)) == 1
    assert rec.ops.count(("clear_accum",)) == 1
    # batch defaults to 0 → single full-photon dispatch.
    assert rec.ops.count(("dispatch_count", "wfSppmPhotonTrace", 500, 64)) == 1


def test_sppm_split_order_all_eye_then_grid_then_all_update():
    # 4 tiles: every eye tile precedes the grid build; every update tile follows
    # the photon pass. No per-tile interleave (the multi-tile correctness fix).
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=256, stream_size=64, num_cells=1024,
                     photons=500, first_frame=False)
    eye_idx = [i for i, op in enumerate(rec.ops) if op == ("dispatch_full", "wfSppmEye")]
    upd_idx = [i for i, op in enumerate(rec.ops) if op == ("dispatch_full", "wfSppmUpdate")]
    grid_idx = rec.ops.index(("clear_grid",))
    photon_idx = rec.ops.index(("dispatch_count", "wfSppmPhotonTrace", 500, 64))
    assert len(eye_idx) == 4 and len(upd_idx) == 4   # 256 px / 64 stream
    assert max(eye_idx) < grid_idx                   # all eye tiles before grid
    assert min(upd_idx) > photon_idx                 # all update tiles after photon
    # heavy-eye submit bound fires once per phase-1 eye tile, none in phase 4.
    assert rec.ops.count(("flush_heavy_eye",)) == 4
    assert all(fi < grid_idx for fi, op in enumerate(rec.ops)
               if op == ("flush_heavy_eye",))


def test_sppm_eye_tiles_have_distinct_bases():
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=192, stream_size=64, num_cells=512,
                     photons=10, first_frame=False)
    # push_tile now also drives the photon-batch base; isolate the eye/update
    # dispatch_full bases (photons=10 < batch → one photon push_tile(0)).
    eye_upd_bases = [
        rec.ops[i - 1][1]
        for i, op in enumerate(rec.ops)
        if op in (("dispatch_full", "wfSppmEye"), ("dispatch_full", "wfSppmUpdate"))
    ]
    # 3 eye tiles + 3 update tiles, bases 0/64/128 each phase.
    assert eye_upd_bases == [0, 64, 128, 0, 64, 128]


# ── photon-dispatch tiling (change sppm-photon-dispatch-tiling) ──────

def test_sppm_photon_dispatch_tiled_into_flushed_batches():
    # batch=256 (64-aligned), photons=500 → sub-batches [0,256] × counts [256,244],
    # each its own flushed command buffer; clear_accum stays a single pre-loop op.
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=500, first_frame=False, photon_batch=256)
    photon = [op for op in rec.ops if op[0] == "dispatch_count"
              and op[1] == "wfSppmPhotonTrace"]
    assert photon == [
        ("dispatch_count", "wfSppmPhotonTrace", 256, 64),
        ("dispatch_count", "wfSppmPhotonTrace", 244, 64),
    ]
    # bases of the photon push_tile ops (the only push_tile between clear_accum
    # and the update phase). slice ends at upd-1 (the first update tile's own
    # push_tile) so only the photon-batch bases remain.
    ca = rec.ops.index(("clear_accum",))
    upd = rec.ops.index(("dispatch_full", "wfSppmUpdate"))
    photon_bases = [op[1] for op in rec.ops[ca:upd - 1] if op[0] == "push_tile"]
    assert photon_bases == [0, 256]
    # every photon batch is flushed; clear_accum is emitted once.
    assert rec.ops.count(("clear_accum",)) == 1
    # exactly one barrier+flush follows the final photon dispatch.
    for i, op in enumerate(rec.ops):
        if op == ("dispatch_count", "wfSppmPhotonTrace", 244, 64):
            assert rec.ops[i + 1] == ("barrier",)
            assert rec.ops[i + 2] == ("flush",)
    # UNBIASED: the tiled counts sum to the full photon budget — no starvation.
    assert sum(op[2] for op in photon) == 500


def test_sppm_photon_batch_aligned_to_threadgroup_width():
    # A non-64-aligned batch MUST be floored to a 64 multiple: `dispatch_count`
    # rounds the launch up to /64, and the kernel is bounded only by the global
    # `pid >= sppmPhotonsEmitted` guard — a non-final batch that over-launched
    # would deposit photons that also belong to the next batch (double-count,
    # energy bias). With alignment every non-final batch count is exactly the
    # aligned batch (a 64 multiple), so no non-final round-up occurs.
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=500, first_frame=False, photon_batch=200)
    photon = [op[2] for op in rec.ops if op[0] == "dispatch_count"
              and op[1] == "wfSppmPhotonTrace"]
    # 200 → aligned 192; 500 = 192 + 192 + 116.
    assert photon == [192, 192, 116]
    # every NON-final batch count is a multiple of 64 (no unmasked over-launch).
    assert all(c % 64 == 0 for c in photon[:-1])
    # still unbiased: exact full budget, no overlap.
    assert sum(photon) == 500


def test_sppm_photon_batch_zero_is_single_full_dispatch():
    # batch<=0 (Vulkan / no watchdog) collapses to one full-photon dispatch, base 0.
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=500, first_frame=False, photon_batch=0)
    photon = [op for op in rec.ops if op[0] == "dispatch_count"
              and op[1] == "wfSppmPhotonTrace"]
    assert photon == [("dispatch_count", "wfSppmPhotonTrace", 500, 64)]


def test_sppm_photon_batch_ge_photons_is_single_dispatch():
    # batch >= photons → no tiling, one dispatch of the whole budget.
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=300, first_frame=False, photon_batch=4096)
    photon = [op for op in rec.ops if op[0] == "dispatch_count"
              and op[1] == "wfSppmPhotonTrace"]
    assert photon == [("dispatch_count", "wfSppmPhotonTrace", 300, 64)]


def test_sppm_photon_dispatch_never_exceeds_workgroup_limit():
    # Vulkan guarantees only maxComputeWorkGroupCount[0] >= 65535 → one dispatch
    # carries at most 65535*64 photons; larger budgets (env-aware ×8) must tile
    # even when the backend requested no watchdog batching (photon_batch <= 0),
    # or a driver may silently clamp groupCountX (dim bias). Batches stay
    # 64-aligned and sum to the full budget.
    limit = 65535 * 64
    photons = limit + 100_000
    rec = _SppmStub(stream_size=64)
    record_sppm_loop(rec, num_pixels=64, stream_size=64, num_cells=256,
                     photons=photons, first_frame=False, photon_batch=0)
    batches = [op for op in rec.ops if op[0] == "dispatch_count"
               and op[1] == "wfSppmPhotonTrace"]
    counts = [op[2] for op in batches]
    assert max(counts) <= limit
    assert sum(counts) == photons
    assert all(c % 64 == 0 for c in counts[:-1])  # non-final batches aligned
