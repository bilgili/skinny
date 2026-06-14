"""Training-backend seam tests (change ``neural-trainer-backends``).

Covers backend selection + capability gating (§1.3), the shared numpy dataset
contract (§2.2/2.3), the fp8 e4m3 codec + footprint (§6.5 CPU part), and the
backend parity / precision invariants (§7). The numpy-vs-torch parity and the
spline_flow dataset parity skip cleanly on a torch-free host; the numpy
reference, the dataset shape/dtype, the fp8 codec, and the fp32-bake invariant
all run with numpy alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from skinny.sampling.neural_replay import ReplayBuffer
from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig
from skinny.sampling.neural_weights import (
    NeuralBuildConfig,
    NeuralPrecision,
    e4m3_to_f32,
    f32_to_e4m3,
    make_dummy_weights,
)
from skinny.sampling.path_records import RECORD_DTYPE
from skinny.sampling.training_backends import (
    MlxTrainingBackend,
    NumpyTrainingBackend,
    TorchTrainingBackend,
    _mlx_metal,
    build_dataset_np,
    make_training_backend,
)

# The MLX backend runs only on an Apple-Silicon Metal host with the optional
# `[mlx]` extra installed; everywhere else these tests skip cleanly.
_mlx_skip = pytest.mark.skipif(not _mlx_metal(),
                               reason="mlx unavailable (Apple-Silicon Metal + [mlx] extra)")

BOUNDS = (np.zeros(3, np.float32), np.ones(3, np.float32))


def _torch_spline_flow():
    """The torch CPU backend is usable here (torch + spline_flow importable)."""
    return TorchTrainingBackend(device="cpu").is_available()


def _concentrated(n, rng, lobe=(0.3, 0.9, 0.2)):
    r = np.zeros(n, dtype=RECORD_DTYPE)
    r["pos"] = rng.uniform(0.2, 0.8, (n, 3)).astype(np.float32)
    r["normal"] = np.array([0.0, 1.0, 0.0], np.float32)
    r["wo"] = np.array([0.0, 1.0, 0.0], np.float32)
    d = np.asarray(lobe, np.float32)
    d = d / np.linalg.norm(d)
    wi = d[None, :] + 0.08 * rng.standard_normal((n, 3)).astype(np.float32)
    wi[:, 1] = np.abs(wi[:, 1]) + 0.2
    wi /= np.linalg.norm(wi, axis=1, keepdims=True)
    r["wi_local"] = wi
    r["contrib"] = rng.uniform(0.5, 1.5, (n, 3)).astype(np.float32)
    return r


# ── §1.3 selection + capability gating ───────────────────────────────────

def test_auto_precedence_cuda_then_mlx_then_numpy():
    """§1.3 (modified by mlx-neural-trainer): auto precedence is cuda > mlx > cpu —
    torch CUDA when present, else MLX on an Apple-Silicon Metal host, else numpy."""
    import skinny.sampling.training_backends as tb
    be = make_training_backend("auto")
    if tb._torch_cuda():
        assert be.name == "torch"
    elif tb._mlx_metal():
        assert be.name == "mlx"
    else:
        assert be.name == "numpy"


def test_cpu_token_is_numpy():
    assert make_training_backend("cpu").name == "numpy"


def test_explicit_cuda_without_torch_raises_clearly():
    import importlib.util
    if importlib.util.find_spec("torch") is not None and \
            __import__("torch").cuda.is_available():
        pytest.skip("CUDA present — the unavailable-token path is not exercised")
    with pytest.raises(RuntimeError) as ei:
        make_training_backend("cuda")
    msg = str(ei.value).lower()
    assert "cuda" in msg and ("pytorch" in msg or "cuda device" in msg or "spline_flow" in msg)


def test_mlx_unavailable_token_raises_clearly(monkeypatch):
    """§1.3: an explicit `mlx` token on a host without mlx/Metal fails with a
    clear message naming the missing piece (runs on every host)."""
    import skinny.sampling.training_backends as tb
    monkeypatch.setattr(tb, "_mlx_metal", lambda: False)
    with pytest.raises(RuntimeError) as ei:
        make_training_backend("mlx")
    msg = str(ei.value).lower()
    assert "mlx" in msg and ("metal" in msg or "extra" in msg)


@_mlx_skip
def test_mlx_token_is_mlx():
    be = make_training_backend("mlx")
    assert be.name == "mlx" and be.is_available()


@_mlx_skip
def test_auto_prefers_mlx_on_metal_without_cuda():
    import skinny.sampling.training_backends as tb
    if tb._torch_cuda():
        pytest.skip("CUDA present — auto resolves to cuda, not mlx")
    assert make_training_backend("auto").name == "mlx"


@_mlx_skip
def test_mlx_supports_precision():
    be = MlxTrainingBackend()
    assert be.supports_precision("fp32")
    assert be.supports_precision("fp16")               # float16 compute over fp32 masters
    assert not be.supports_precision("bf16")


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_training_backend("banana")


def test_supports_precision_reports_unsupported():
    np_be = NumpyTrainingBackend()
    assert np_be.supports_precision("fp32")
    assert not np_be.supports_precision("fp16")        # numpy oracle is fp32-only
    assert not np_be.supports_precision("bf16")
    t_be = TorchTrainingBackend(device="cpu")
    assert t_be.supports_precision("fp32")
    assert not t_be.supports_precision("fp16", "cpu")  # autocast fp16 = CUDA only
    assert t_be.supports_precision("fp16", "cuda")


# ── §2.2/2.3 shared numpy dataset contract ───────────────────────────────

def test_build_dataset_np_float32_contiguous():
    rng = np.random.default_rng(0)
    cond, z, w = build_dataset_np(_concentrated(256, rng), BOUNDS)
    for a in (cond, z, w):
        assert a.dtype == np.float32 and a.flags["C_CONTIGUOUS"]
    assert cond.shape[1] == 9 and z.shape[1] == 2
    assert cond.shape[0] == z.shape[0] == w.shape[0] > 0


def test_build_dataset_np_filters_lower_hemisphere():
    rng = np.random.default_rng(0)
    recs = _concentrated(128, rng)
    recs["wi_local"][:, 1] = -0.5            # all below the horizon → all dropped
    cond, z, w = build_dataset_np(recs, BOUNDS)
    assert cond.shape[0] == 0 and z.shape == (0, 2) and w.shape[0] == 0


@pytest.mark.skipif(not _torch_spline_flow(),
                    reason="torch + spline_flow unavailable")
def test_build_dataset_np_matches_spline_flow():
    """§2.2: numpy dataset matches spline_flow's torch build_dataset on a batch."""
    import torch
    from render_records import build_dataset            # spline_flow (on sys.path)
    rng = np.random.default_rng(7)
    recs = _concentrated(2000, rng)
    bmin, bext = BOUNDS
    cond_np, z_np, w_np = build_dataset_np(recs, BOUNDS)
    cond_t, z_t, w_t = build_dataset(recs, np.asarray(bmin), np.asarray(bext),
                                     torch.device("cpu"))
    np.testing.assert_allclose(cond_np, cond_t.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(z_np, z_t.cpu().numpy(), atol=1e-5)
    np.testing.assert_allclose(w_np, w_t.cpu().numpy(), atol=1e-4)


# ── §6.5 fp8 e4m3 codec + footprint (CPU) ────────────────────────────────

def test_e4m3_roundtrip_exact_values():
    exact = np.array([0.0, 1.0, -1.0, 2.0, 0.5, 0.25, 1.5, 1.75, 448.0, -448.0,
                      0.125, 2.0 ** -6, 2.0 ** -9], np.float32)
    np.testing.assert_array_equal(e4m3_to_f32(f32_to_e4m3(exact)), exact)


def test_e4m3_roundtrip_tolerance_and_saturation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200000).astype(np.float32)
    xn = x[np.abs(x) >= 2.0 ** -6]                  # e4m3 normalized range
    rel = np.abs(e4m3_to_f32(f32_to_e4m3(xn)) - xn) / np.abs(xn)
    assert rel.max() <= 1.0 / 16.0 + 1e-6           # 3-mantissa-bit worst case
    # saturates (no inf in e4m3), never emits the NaN slot
    assert e4m3_to_f32(f32_to_e4m3(np.array([1e9], np.float32)))[0] == 448.0
    bb = f32_to_e4m3(np.linspace(-600, 600, 50000).astype(np.float32))
    assert not np.any((bb & 0x7F) == 0x7F)


def test_fp8_quarter_footprint():
    nw = make_dummy_weights()
    rng = np.random.default_rng(1)
    nw.weights[:] = (rng.standard_normal(nw.weights.shape) * 0.3).astype(np.float32)
    f32 = nw.weight_bytes_for(NeuralPrecision.FP32)
    fp8 = nw.weight_bytes_for(NeuralPrecision.FP8_STORAGE)
    assert len(fp8) == len(f32) // 4                # exactly a quarter
    assert len(fp8) % 4 == 0                        # packs into uint words
    assert NeuralPrecision.FP8_STORAGE.storage_bytes == 1
    assert not NeuralPrecision.FP8_STORAGE.weight_half
    assert not NeuralPrecision.FP8_STORAGE.needs_device_fp16_storage
    cfg = NeuralBuildConfig(precision=NeuralPrecision.FP8_STORAGE)
    assert "NF_FP8=1" in cfg.slang_defines() and "NF_WT=uint" in cfg.slang_defines()


# ── §7 backend parity + correctness ──────────────────────────────────────

def test_numpy_backend_real_update_with_signal():
    """§7.3: a torch-free host performs a real update (weights move with signal)."""
    rng = np.random.default_rng(0)
    replay = ReplayBuffer(capacity=50_000)
    replay.add(_concentrated(8192, rng))
    tr = NeuralTrainer(TrainerConfig(backend="cpu", steps_per_cycle=120, batch=2048,
                                     lr=2e-3, bounds=BOUNDS))
    assert tr.backend_name == "numpy"
    w0 = tr.weights.weights.copy()
    new = tr.train_cycle(replay, rng)
    assert (new.layers, new.bins, new.hidden, new.cond) == (6, 24, 96, 9)
    assert np.all(np.isfinite(new.weights)) and np.all(np.isfinite(new.biases))
    assert not np.allclose(new.weights, w0), "training did not update the weights"
    assert tr.last_loss is not None and np.isfinite(tr.last_loss)


def test_numpy_backend_learns_over_cycles():
    rng = np.random.default_rng(2)
    replay = ReplayBuffer(capacity=50_000)
    tr = NeuralTrainer(TrainerConfig(backend="cpu", steps_per_cycle=120, batch=2048,
                                     lr=2e-3, bounds=BOUNDS))
    losses = []
    for _ in range(5):
        replay.add(_concentrated(4096, rng))
        tr.train_cycle(replay, rng)
        losses.append(tr.last_loss)
    assert all(np.isfinite(x) for x in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_fp16_request_still_bakes_fp32_weights():
    """§7.2: requesting fp16 training never changes the baked weight format —
    exported weights stay fp32 (here the numpy oracle falls back to fp32, and
    the bake is fp32 in every mode)."""
    rng = np.random.default_rng(0)
    replay = ReplayBuffer(capacity=50_000)
    replay.add(_concentrated(8192, rng))
    tr = NeuralTrainer(TrainerConfig(backend="cpu", train_precision="fp16",
                                     steps_per_cycle=40, batch=2048, bounds=BOUNDS))
    new = tr.train_cycle(replay, rng)
    assert new.weights.dtype == np.dtype("<f4")
    assert new.biases.dtype == np.dtype("<f4")
    # format identical to an fp32-trained run
    ref = make_dummy_weights()
    assert new.weights.shape == ref.weights.shape
    np.testing.assert_array_equal(new.headers, ref.headers)


@pytest.mark.skipif(not _torch_spline_flow(),
                    reason="torch + spline_flow unavailable (numpy↔torch parity)")
def test_numpy_matches_torch_cpu_one_cycle():
    """§7.1: numpy backend ≈ torch CPU backend on the same fixed batch/seed for
    one cycle — the drift guard between the two implementations."""
    arch = NeuralBuildConfig()
    init = make_dummy_weights(arch)
    rng = np.random.default_rng(123)
    init.weights[:] = (rng.standard_normal(init.weights.shape) * 0.05).astype(np.float32)
    init.biases[:] = (rng.standard_normal(init.biases.shape) * 0.05).astype(np.float32)

    recs = _concentrated(4096, np.random.default_rng(5))
    cond, z, w = build_dataset_np(recs, BOUNDS)
    cfg = TrainerConfig(arch=arch, steps_per_cycle=1, batch=cond.shape[0], lr=1e-3,
                        bounds=BOUNDS)

    np_be = NumpyTrainingBackend()
    np_be.warm_start(init, cfg)
    np_be.update(cond, z, w)
    nw_np = np_be.export()

    t_be = TorchTrainingBackend(device="cpu")
    t_be.warm_start(init, cfg)
    t_be.update(cond, z, w)
    nw_t = t_be.export()

    # one Adam step from identical weights/grads → agreement to a documented tol.
    diff = np.abs(nw_np.weights - nw_t.weights).max()
    print(f"\n[7.1] numpy vs torch-cpu max |Δw| after one cycle = {diff:.2e}")
    assert diff < 5e-4, f"numpy/torch drift {diff:.2e} exceeds 5e-4"


@_mlx_skip
def test_mlx_backend_survives_multithread_access():
    """Regression (change ``mlx-trainer-thread-confinement``): MLX arrays and GPU
    streams are thread-affine. The backend confines all MLX work to one owned
    worker thread, so warm-starting from one thread and then update/export from a
    *different* thread — as a direct call concurrent with the background daemon
    trainer would — must complete rather than raise
    ``There is no Stream(gpu, N) in current thread.``"""
    import threading

    arch = NeuralBuildConfig()
    init = make_dummy_weights(arch)
    cond, z, w = build_dataset_np(_concentrated(2048, np.random.default_rng(7)), BOUNDS)
    cfg = TrainerConfig(arch=arch, steps_per_cycle=4, batch=cond.shape[0], lr=1e-3,
                        bounds=BOUNDS)

    be = MlxTrainingBackend()
    be.warm_start(init, cfg)                       # caller: the main test thread

    errors: list = []
    out: dict = {}

    def worker():
        try:
            be.update(cond, z, w)                  # caller: a different thread
            out["nw"] = be.export()
        except Exception as exc:                   # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=worker, name="mlx-other-thread")
    t.start()
    t.join()

    assert not errors, f"multi-thread MLX access raised: {errors!r}"
    assert out["nw"].weights.shape == init.weights.shape
