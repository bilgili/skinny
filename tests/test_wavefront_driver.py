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
    ]


def test_bdpt_two_tiles_leading_barrier_on_second():
    rec = _StubRecorder(stream_size=64)
    record_bdpt_loop(rec, num_pixels=128, stream_size=64, walk_mode="fused",
                     eye_bounces=5, light_bounces=6)
    assert rec.ops[0] == ("push_tile", 0)
    i = rec.ops.index(("push_tile", 64))
    assert rec.ops[i - 1] == ("barrier",)
    assert rec.ops[i - 2] == ("dispatch_full", "wfBdptResolve")
    assert rec.ops[i + 1] == ("dispatch_full", "wfBdptWalk")


def test_bdpt_rejects_unknown_walk_mode():
    rec = _StubRecorder(stream_size=64)
    with pytest.raises(ValueError, match="walk_mode"):
        record_bdpt_loop(rec, num_pixels=64, stream_size=64, walk_mode="nope",
                         eye_bounces=5, light_bounces=6)
