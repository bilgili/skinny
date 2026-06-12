"""End-to-end Metal UMA interop weight handoff on a real device
(change metal-neural-interop, tasks 4.1–4.2).

Drives the real online-training pieces — `NeuralTrainer` cycles on a replay
buffer, `make_publisher("interop")` resolving to the Metal publisher, real
`StorageBuffer(shared=True)` weight/bias buffers on a `MetalContext` — and
asserts the spec scenarios:

* publishes land in the GPU buffers at the frame-boundary swap with no file
  written and no NFW1 round-trip, and the network version increments;
* the GPU buffer contents are byte-identical to what a file-backend publish
  of the same weights would upload (precision faithfulness on-device).

Render-level parity follows transitively: the renderer samples the network
from bindings 33/34, so identical bytes there ⇒ identical renders — the
neural render itself is already pinned Metal≡Vulkan by
`test_metal_neural_ab.py` with frozen weights.

No kernel compile here (buffer traffic only), but a Metal device is
constructed — gpu-marked so `-m 'not gpu'` sweeps skip it (thermal rule:
run via scripts/guarded_metal.sh, one Metal process at a time).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from skinny.backend_select import metal_available

pytestmark = pytest.mark.gpu


def _require_metal():
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")


def _records(n: int, tag: float) -> np.ndarray:
    from skinny.sampling.path_records import RECORD_DTYPE

    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = tag
    r["wi_local"] = [0.0, 0.0, 1.0]
    r["contrib"] = 1.0
    return r


def test_online_interop_loop_on_metal(tmp_path):
    """drain → train → publish(interop) → frame-boundary swap → GPU bytes match
    the file path, version increments, no file written."""
    from skinny.metal_compute import StorageBuffer
    from skinny.metal_context import MetalContext
    from skinny.sampling.neural_handoff import make_publisher
    from skinny.sampling.neural_handoff_file import FileWeightPublisher
    from skinny.sampling.neural_handoff_interop_metal import MetalSharedWeightPublisher
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
    from skinny.sampling.neural_weights import NeuralPrecision

    _require_metal()
    ctx = MetalContext(window=None, width=64, height=64)
    wbuf = bbuf = None
    try:
        if not ctx.supports_shared_memory:
            pytest.skip("Metal device reports no shared-storage support")
        trainer = NeuralTrainer(TrainerConfig(steps_per_cycle=2, batch=256))
        init = trainer.weights
        precision = NeuralPrecision.FP32
        wbuf = StorageBuffer(ctx, len(init.weight_bytes_for(precision)), shared=True)
        bbuf = StorageBuffer(ctx, len(init.bias_bytes_for(precision)), shared=True)
        assert wbuf.shared and bbuf.shared

        pub = make_publisher(
            "interop", weights_buffer=wbuf, biases_buffer=bbuf,
            precision=precision, initial=init, expect_arch=init.arch
            if hasattr(init, "arch") else None)
        assert isinstance(pub, MetalSharedWeightPublisher)

        fdir = tmp_path / "nfw"
        fpub = FileWeightPublisher(weights_dir=fdir, initial=init)
        replay = ReplayBuffer(capacity=100_000)
        rng = np.random.default_rng(0)

        swap_times = []
        for frame in range(3):
            replay.add(_records(2_000, tag=float(frame)))
            new_w = trainer.train_cycle(replay, rng)
            pub.publish(new_w)
            fpub.publish(new_w)
            t0 = time.perf_counter()
            assert pub.swap() is True               # frame-boundary promote
            swap_times.append(time.perf_counter() - t0)
            assert fpub.swap() is True
            assert pub.current_version() == frame + 1

            # GPU bytes == what the renderer would upload from the file path.
            loaded, fver = fpub.acquire_for_render()
            assert fver == frame + 1
            want_w = loaded.weight_bytes_for(precision)
            want_b = loaded.bias_bytes_for(precision)
            assert wbuf.download_sync(len(want_w)) == want_w
            assert bbuf.download_sync(len(want_b)) == want_b

        # interop wrote no NFW1 files anywhere (the file publisher's dir has
        # exactly its own three versions).
        assert sorted(p.name for p in fdir.iterdir()) == [
            f"weights_v{v:06d}.nfw1" for v in (1, 2, 3)]
        nw, ver = pub.acquire_for_render()
        assert nw is None and ver == 3              # renderer re-stamps, no re-upload

        # 4.2: the staged in-place copy at the frame boundary is host-memcpy
        # cheap — record the number for the change notes.
        print(f"\n[interop swap cost] weights+biases "
              f"{len(want_w) + len(want_b)} B: "
              f"{[f'{t * 1e3:.3f} ms' for t in swap_times]}")
        assert max(swap_times) < 0.05               # well under a frame
    finally:
        for buf in (wbuf, bbuf):
            if buf is not None:
                buf.destroy()
        ctx.destroy()
