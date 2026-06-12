"""Unit tests for the Metal wavefront record drain host logic
(change metal-record-drain, task 2.5). No GPU — the renderer methods are
exercised unbound against duck-typed fakes.

Covers the Metal merged header+records drain layout (capacity @ byte 0, count
@ byte 60, packed 64-byte records from byte 64), the counter clamp + 4-byte
count reset, the record-source resolution that arms `_wf_record_active`, and
the megakernel-source refusal on Metal.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from skinny.renderer import EXECUTION_WAVEFRONT, Renderer
from skinny.sampling.path_records import RECORD_DTYPE, RECORD_STRIDE


class FakeDrainBuffer:
    """Duck-typed Metal StorageBuffer holding the merged header+records bytes."""

    def __init__(self, capacity: int, records: np.ndarray, count: int | None = None):
        self.size = 64 + capacity * RECORD_STRIDE
        header = bytearray(64)
        header[0:4] = struct.pack("<I", capacity)
        header[60:64] = struct.pack("<I", count if count is not None else len(records))
        self._bytes = bytes(header) + records.tobytes()
        self.range_writes = []

    def download_sync(self, n: int) -> bytes:
        return self._bytes[:n]

    def upload_range(self, data: bytes, offset: int) -> None:
        self.range_writes.append((bytes(data), int(offset)))


class FakeReplay:
    def __init__(self):
        self.added = []

    def add(self, records):
        self.added.append(records)


def _records(n: int, tag: float) -> np.ndarray:
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag
    r["wi_local"] = [0.0, 0.0, 1.0]
    r["contrib"] = 1.0
    return r


class FakeMetalRenderer:
    """Just the attrs `_drain_wavefront_records` touches on the Metal branch."""

    is_metal = True

    def __init__(self, capacity: int, records: np.ndarray, count: int | None = None):
        self._drain_buffer = FakeDrainBuffer(capacity, records, count)
        self._capacity = capacity

    def _ensure_wf_record_drain(self, max_records_per_frame=None) -> int:
        return self._capacity


def test_metal_drain_reads_header_records_and_resets_count():
    recs = _records(3, tag=7.0)
    fake = FakeMetalRenderer(capacity=16, records=recs)
    replay = FakeReplay()

    count = Renderer._drain_wavefront_records(fake, replay, None)

    assert count == 3
    np.testing.assert_array_equal(replay.added[0], recs)
    # Only the 4-byte count word resets — never a full-buffer rewrite.
    assert fake._drain_buffer.range_writes == [(struct.pack("<I", 0), 60)]


def test_metal_drain_clamps_overreported_count():
    recs = _records(2, tag=1.0)
    fake = FakeMetalRenderer(capacity=2, records=recs, count=999)  # GPU over-report
    replay = FakeReplay()

    count = Renderer._drain_wavefront_records(fake, replay, None)

    assert count == 2
    assert len(replay.added[0]) == 2


def test_metal_drain_empty_frame_adds_nothing():
    fake = FakeMetalRenderer(capacity=8, records=_records(0, 0.0), count=0)
    replay = FakeReplay()

    assert Renderer._drain_wavefront_records(fake, replay, None) == 0
    assert replay.added == []
    assert fake._drain_buffer.range_writes == [(struct.pack("<I", 0), 60)]


# ── record-source resolution (arms _wf_record_active) ────────────────


class FakeSourceRenderer:
    def __init__(self, execution_mode, integrator, source="auto"):
        self.effective_execution_mode_index = execution_mode
        self.integrator_index = integrator
        self._record_source = source


def test_wavefront_path_resolves_wavefront_source():
    fake = FakeSourceRenderer(EXECUTION_WAVEFRONT, integrator=0)
    assert Renderer._resolve_record_source(fake) == "wavefront"


def test_bdpt_and_megakernel_fall_back_to_megakernel_source():
    assert Renderer._resolve_record_source(
        FakeSourceRenderer(EXECUTION_WAVEFRONT, integrator=1)) == "megakernel"
    assert Renderer._resolve_record_source(
        FakeSourceRenderer(0, integrator=0)) == "megakernel"


# ── megakernel source refused on Metal ───────────────────────────────


def test_megakernel_record_source_raises_on_metal():
    class FakeMetal:
        is_metal = True
        _scene_bindings = object()
        descriptor_sets = None

        def _resolve_record_source(self):
            return "megakernel"

    with pytest.raises(RuntimeError, match="megakernel record source is unavailable"):
        Renderer.drain_path_records_to_replay(FakeMetal(), FakeReplay())
