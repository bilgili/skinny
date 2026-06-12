"""Unit tests for the Metal UMA shared-storage weight publisher
(change metal-neural-interop, tasks 2.1–2.4, 4.3). No GPU — the renderer-side
buffers are duck-typed fakes recording `write_in_place` calls.

Covers the publisher protocol (stage on publish, write+promote at swap), byte
faithfulness against the file publisher for every `NeuralPrecision`, the
torn-write guard (a swap never lands a mixed-version weights/biases pair), the
arch-mismatch check, the `make_publisher` backend dispatch, and the
no-GPU-path availability guard.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from skinny.sampling.neural_handoff import make_publisher
from skinny.sampling.neural_handoff_file import FileWeightPublisher
from skinny.sampling.neural_handoff_interop_metal import (
    MetalSharedWeightPublisher,
    metal_interop_available,
)
from skinny.sampling.neural_weights import (
    NeuralBuildConfig,
    NeuralPrecision,
    make_dummy_weights,
)


class FakeMetalCtx:
    is_metal = True
    supports_shared_memory = True


class FakeSharedBuffer:
    """Duck-typed `metal_compute.StorageBuffer(shared=True)` — records writes."""

    def __init__(self, ctx=None):
        self.ctx = ctx or FakeMetalCtx()
        self.shared = True
        self.contents = b""
        self.writes = []

    def write_in_place(self, data: bytes, offset: int = 0) -> None:
        assert offset == 0
        self.contents = bytes(data)
        self.writes.append(bytes(data))


def _weights(value: float = 0.0):
    nw = make_dummy_weights()
    if value:
        nw.weights[:] = value
        nw.biases[:] = value
    return nw


def _publisher(precision=None, expect_arch=None):
    wbuf, bbuf = FakeSharedBuffer(), FakeSharedBuffer()
    pub = MetalSharedWeightPublisher(
        weights_buffer=wbuf, biases_buffer=bbuf,
        expect_arch=expect_arch, precision=precision)
    return pub, wbuf, bbuf


# ── protocol ─────────────────────────────────────────────────────────


def test_publish_stages_swap_writes_and_promotes():
    pub, wbuf, bbuf = _publisher()
    assert pub.current_version() == 0
    assert pub.swap() is False                      # nothing staged
    assert pub.publish(_weights(1.0)) == 1
    assert wbuf.writes == []                        # publish never touches the GPU
    assert pub.current_version() == 0
    assert pub.swap() is True
    assert pub.current_version() == 1
    assert len(wbuf.writes) == 1 and len(bbuf.writes) == 1
    assert pub.swap() is False                      # staged slot consumed
    assert pub.acquire_for_render() == (None, 1)    # weights live in the buffers


def test_two_publishes_one_swap_lands_latest_only():
    pub, wbuf, _ = _publisher()
    pub.publish(_weights(1.0))
    pub.publish(_weights(2.0))
    assert pub.swap() is True
    assert pub.current_version() == 2
    assert len(wbuf.writes) == 1                    # superseded staging never written
    assert wbuf.contents == _weights(2.0).weight_bytes_for(NeuralPrecision.FP32)


def test_close_is_idempotent_and_drops_staged():
    pub, wbuf, _ = _publisher()
    pub.publish(_weights(1.0))
    pub.close()
    pub.close()
    assert pub.swap() is False
    assert wbuf.writes == []


# ── precision faithfulness vs the file path ──────────────────────────


@pytest.mark.parametrize("precision", list(NeuralPrecision))
def test_interop_bytes_match_file_path(tmp_path, precision):
    """Spec: Metal interop publish is precision-faithful — byte-identical buffer
    contents to a file-backend publish of the same weights."""
    nw = _weights(0.625)  # exactly representable in fp16 and e4m3
    pub, wbuf, bbuf = _publisher(precision=precision)
    pub.publish(nw)
    pub.swap()

    fpub = FileWeightPublisher(weights_dir=tmp_path)
    fpub.publish(nw)
    fpub.swap()
    loaded, _ = fpub.acquire_for_render()           # what the renderer would upload

    assert wbuf.contents == loaded.weight_bytes_for(precision)
    assert bbuf.contents == loaded.bias_bytes_for(precision)


# ── torn-write guard ─────────────────────────────────────────────────


def test_concurrent_publish_never_tears_a_swap():
    """A swap must land a weights/biases pair from one version — publishes
    racing the swap from another thread never mix versions."""
    pub, wbuf, bbuf = _publisher()
    stop = threading.Event()

    def trainer():
        v = 0
        while not stop.is_set():
            v += 1
            pub.publish(_weights(float(v % 7 + 1)))

    t = threading.Thread(target=trainer, daemon=True)
    t.start()
    try:
        swaps = 0
        while swaps < 50:
            if pub.swap():
                swaps += 1
                # weights and biases written as one pair, same value tag
                w = np.frombuffer(wbuf.contents, dtype="<f4")
                b = np.frombuffer(bbuf.contents, dtype="<f4")
                assert w[0] == b[0]
                assert np.all(w == w[0]) and np.all(b == b[0])
    finally:
        stop.set()
        t.join(timeout=5)
    assert len(wbuf.writes) == len(bbuf.writes) == swaps


# ── guards ───────────────────────────────────────────────────────────


def test_arch_mismatch_raises_on_publish():
    small = NeuralBuildConfig(layers=2, bins=8, hidden=16)
    pub, _, _ = _publisher(expect_arch=small.arch)
    with pytest.raises(ValueError, match="architecture"):
        pub.publish(_weights())                     # shipped-size net ≠ small arch


def test_missing_buffers_raise():
    with pytest.raises(RuntimeError, match="bindings 33/34"):
        MetalSharedWeightPublisher()


def test_metal_interop_available_requires_shared_metal():
    class NotMetal:
        is_metal = False

    class NoShared:
        is_metal = True
        supports_shared_memory = False

    assert metal_interop_available(None) == (False, "no GPU context")
    ok, reason = metal_interop_available(NotMetal())
    assert not ok and "not a Metal context" in reason
    ok, reason = metal_interop_available(NoShared())
    assert not ok and "shared-storage" in reason
    assert metal_interop_available(FakeMetalCtx()) == (True, "")


def test_non_shared_metal_ctx_raises_with_file_fallback():
    class NoShared:
        is_metal = True
        supports_shared_memory = False

    buf = FakeSharedBuffer(ctx=NoShared())
    with pytest.raises(NotImplementedError, match="--neural-handoff file"):
        MetalSharedWeightPublisher(weights_buffer=buf, biases_buffer=buf)


# ── factory dispatch (task 2.3) + no-GPU-path guard (task 4.3) ───────


def test_make_publisher_dispatches_metal_buffer_to_metal_publisher():
    wbuf, bbuf = FakeSharedBuffer(), FakeSharedBuffer()
    pub = make_publisher(
        "interop", weights_buffer=wbuf, biases_buffer=bbuf,
        timeline_semaphore=object(),               # dropped on the Metal path
        precision=NeuralPrecision.FP32, initial=None)
    assert isinstance(pub, MetalSharedWeightPublisher)


def test_make_publisher_interop_without_any_gpu_path_names_fallback(monkeypatch):
    """Spec: guard where no GPU path exists — neither CUDA nor Metal UMA."""
    import skinny.sampling.neural_handoff_interop as cuda_mod

    monkeypatch.setattr(cuda_mod, "interop_available",
                        lambda: (False, "no CUDA device"))

    class VkBuffer:                                 # non-Metal ctx → CUDA branch
        ctx = None

    with pytest.raises(NotImplementedError) as exc:
        make_publisher("interop", weights_buffer=VkBuffer(), biases_buffer=VkBuffer())
    msg = str(exc.value)
    assert "CUDA" in msg and "Metal" in msg and "--neural-handoff file" in msg
