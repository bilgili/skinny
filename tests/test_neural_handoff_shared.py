"""Change ``shared-neural-handoff`` — the in-process CPU double-buffer handoff.

CPU-only, no GPU / CUDA / unified-memory device: the ``shared`` backend hands
weights trainer→render through RAM. Validates the publisher contract (publish→
swap→acquire + version increment), the frozen-render-buffer invariant under
post-publish trainer mutation, byte-parity with the ``file`` backend, and the
factory wiring + error message.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from skinny.sampling.neural_handoff import make_publisher
from skinny.sampling.neural_handoff_file import FileWeightPublisher
from skinny.sampling.neural_handoff_shared import SharedWeightPublisher
from skinny.sampling.neural_weights import make_dummy_weights


def _weights(seed: float = 1.0):
    nw = make_dummy_weights()
    nw.weights[:] = np.arange(nw.weights.size, dtype="<f4") * seed
    nw.biases[:] = np.linspace(-1.0, 1.0, nw.biases.size, dtype="<f4") * seed
    return nw, (nw.layers, nw.bins, nw.hidden, nw.cond)


def test_publish_swap_acquire_and_version_increment():
    nw, arch = _weights()
    p = SharedWeightPublisher(expect_arch=arch)
    assert p.acquire_for_render() == (None, 0)
    assert p.current_version() == 0

    assert p.publish(nw) == 1
    # frozen until the renderer swaps at the frame boundary
    assert p.acquire_for_render() == (None, 0)
    assert p.current_version() == 0

    assert p.swap() is True
    got, ver = p.acquire_for_render()
    assert ver == 1 and p.current_version() == 1
    assert np.array_equal(got.weights, nw.weights)
    assert np.array_equal(got.biases, nw.biases)

    # nothing pending → no swap, version unchanged
    assert p.swap() is False
    assert p.current_version() == 1


def test_render_buffer_frozen_against_trainer_mutation():
    nw, arch = _weights()
    p = SharedWeightPublisher(expect_arch=arch)
    p.publish(nw)
    p.swap()
    frozen = p.acquire_for_render()[0].weights.copy()

    # trainer keeps mutating its own working weights in place after publishing
    nw.weights[:] = 12345.0
    nw.biases[:] = -42.0
    assert np.array_equal(p.acquire_for_render()[0].weights, frozen)

    # only a new publish+swap promotes the mutated values
    p.publish(nw)
    p.swap()
    assert np.all(p.acquire_for_render()[0].weights == 12345.0)
    assert p.current_version() == 2


def test_shared_publish_byte_faithful_to_file():
    nw, arch = _weights(seed=3.0)

    fp = FileWeightPublisher(weights_dir=tempfile.mkdtemp(), expect_arch=arch)
    fp.publish(nw)
    fp.swap()
    fw, _ = fp.acquire_for_render()

    sp = SharedWeightPublisher(expect_arch=arch)
    sp.publish(nw)
    sp.swap()
    sw, _ = sp.acquire_for_render()

    assert sw.weight_bytes == fw.weight_bytes
    assert sw.bias_bytes == fw.bias_bytes
    assert sw.header_bytes == fw.header_bytes


def test_factory_resolves_shared_and_lists_all_backends():
    _, arch = _weights()
    p = make_publisher("shared", initial=None, expect_arch=arch)
    assert isinstance(p, SharedWeightPublisher)

    with pytest.raises(ValueError) as exc:
        make_publisher("bogus")
    msg = str(exc.value)
    assert "file" in msg and "shared" in msg and "interop" in msg
