"""Pluggable training-compute backends for online neural-proposal training.

Change ``neural-trainer-backends``. The per-cycle gradient step of
``NeuralTrainer`` (change ``neural-online-training``) runs behind a small
``TrainingBackend`` interface so the framework that does the compute can be
swapped without touching the orchestrator:

* :class:`TorchTrainingBackend` — the existing torch + ``spline_flow`` loop
  (``device=cpu|mps|cuda``); autocast-fp16 lives inside it (CUDA only). It now
  consumes the shared numpy dataset via ``torch.from_numpy`` and bakes
  in-memory (no per-cycle tempfile round-trip).
* :class:`NumpyTrainingBackend` — a torch-free reference oracle. A hand-written
  forward **and** backward of the contribution-weighted MLE on the *exact*
  shipped flow (``ConditionalSplineFlow2D(cond=9, layers=6, bins=24,
  hidden=96)``), via a tiny pure-numpy reverse-mode autodiff tape (no torch, no
  ``autograd`` dependency). It is the guaranteed-available fallback — a
  torch-free host (macOS CI) now trains for real instead of running the old
  placeholder — and the independent numeric oracle the torch / future MLX
  backends are checked against.

* :class:`MlxTrainingBackend` — an Apple MLX loop (change ``mlx-neural-trainer``)
  that runs the same contribution-weighted MLE on the Metal GPU of an
  Apple-Silicon host, via the optional ``[mlx]`` extra. It mirrors the numpy
  oracle's flow math (the parity target) on ``mlx.core`` arrays with MLX
  autodiff + a hand-rolled bias-corrected Adam, and bakes the same fp32 NFW1.

All backends share one skinny-owned dataset contract
:func:`build_dataset_np` (float32, contiguous), mirroring ``spline_flow``'s
torch ``build_dataset`` so the torch wrap stays zero-copy and the numpy build
never drifts from the offline source of truth (guarded by a parity test).

Backend selection mirrors the ``make_publisher`` / ``registry.py`` house style:
``make_training_backend(kind, ...)`` over the :data:`TRAINING_BACKENDS` token
table — ``cpu`` → numpy, ``cuda`` → torch on CUDA, ``mlx`` → MLX on Apple-Silicon
Metal. ``auto`` precedence is ``cuda > mlx > cpu``: CUDA when torch + a CUDA
device are present, else MLX when the ``[mlx]`` extra is importable on a Metal
host, else the numpy oracle. An unavailable explicit token raises clearly.
"""

from __future__ import annotations

import abc
import os
import sys
from pathlib import Path

import numpy as np

from .neural_weights import (
    NF_COND,
    NeuralBuildConfig,
    NeuralWeights,
    _layout,
    load_neural_weights,  # noqa: F401  (kept for callers/back-compat)
)

__all__ = [
    "TrainingBackend",
    "TorchTrainingBackend",
    "NumpyTrainingBackend",
    "MlxTrainingBackend",
    "TRAINING_BACKENDS",
    "make_training_backend",
    "build_dataset_np",
]

# The three Linear layers inside each SplineCoupling.net (Sequential); idx 1/3
# are SiLU. Mirrors spline_flow/export_weights.py LINEAR_IDX and neural_weights.
_LINEAR_IDX = (0, 2, 4)
_MIN_BIN = 1e-4          # spline_flow SplineCoupling._params
_DERIV_FLOOR = 1e-3      # softplus(raw_d) + 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Shared numpy dataset contract
# ─────────────────────────────────────────────────────────────────────────────

def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]


def _build_condition(recs: np.ndarray, bmin: np.ndarray, bext: np.ndarray) -> np.ndarray:
    """The canonical 9-float condition, EXACTLY as the renderer's neuralCondition
    and spline_flow's build_condition: position normalised to the scene AABB →
    [-1,1]³, then world N, then world wo."""
    ext = np.maximum(bext, 1e-6).astype(np.float32)
    p = (recs["pos"].astype(np.float32) - bmin) / ext        # [0,1]³
    p = p * 2.0 - 1.0                                         # [-1,1]³
    cond = np.concatenate([p, recs["normal"].astype(np.float32),
                           recs["wo"].astype(np.float32)], axis=1)
    return cond.astype(np.float32)


def _hemisphere_to_square_np(w: np.ndarray) -> np.ndarray:
    """Inverse of square_to_hemisphere for the upper hemisphere (y-up flow-local),
    matching spline_flow/train.py hemisphere_to_square."""
    n = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / np.maximum(n, 1e-8)
    phi = np.arctan2(w[:, 2], w[:, 0])
    phi = np.where(phi < 0.0, phi + 2.0 * np.pi, phi)
    u = phi / (2.0 * np.pi)
    v = np.clip(w[:, 1], 0.0, 1.0)
    return np.stack([u, v], axis=1).astype(np.float32)


def build_dataset_np(batch: np.ndarray,
                     bounds: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(cond[N,9], z[N,2], w[N])`` as contiguous float32 arrays from a
    ``RECORD_DTYPE`` batch and the scene ``bounds=(bmin, bext)``.

    Keeps only finite, upper-hemisphere, positive-weight samples; ``z`` is the
    square-domain target (``hemisphere_to_square`` of the flow-local ``wi``),
    ``w`` the luminance of the recorded contribution normalised to mean 1. The
    math duplicates spline_flow's torch ``build_dataset`` (a parity test guards
    the duplication); float32 + contiguous keep ``torch.from_numpy`` zero-copy.
    """
    bmin, bext = bounds
    bmin = np.asarray(bmin, np.float32).reshape(3)
    bext = np.asarray(bext, np.float32).reshape(3)

    cond = _build_condition(batch, bmin, bext)
    wi = batch["wi_local"].astype(np.float32)
    w = _luminance(batch["contrib"].astype(np.float32))

    keep = (np.isfinite(w) & (w > 0.0)
            & np.isfinite(wi).all(axis=1) & (wi[:, 1] > 1e-4))
    cond, wi, w = cond[keep], wi[keep], w[keep]

    if w.shape[0] == 0:
        return (np.ascontiguousarray(cond, np.float32),
                np.ascontiguousarray(np.zeros((0, 2), np.float32)),
                np.ascontiguousarray(w, np.float32))

    z = np.clip(_hemisphere_to_square_np(wi), 1e-4, 1.0 - 1e-4)
    m = float(w.mean())
    w = w / (m if m > 1e-12 else 1e-12)
    return (np.ascontiguousarray(cond, np.float32),
            np.ascontiguousarray(z, np.float32),
            np.ascontiguousarray(w, np.float32))


def _weights_from_linears(linears: list[tuple[np.ndarray, np.ndarray]],
                          cfg: NeuralBuildConfig) -> NeuralWeights:
    """Bake a list of ``(weight[out,in], bias[out])`` (in header order) into a
    ``NeuralWeights`` in memory — the single NFW1 layout shared with
    ``spline_flow/export_weights.export_flow`` and ``neural_weights._layout``,
    applied without any filesystem round-trip."""
    headers, n_w, n_b = _layout(cfg.layers, cfg.bins, cfg.hidden)
    weights = np.zeros(n_w, dtype="<f4")
    biases = np.zeros(n_b, dtype="<f4")
    for (w_off, b_off, in_dim, out_dim), (w, b) in zip(headers, linears):
        weights[w_off:w_off + out_dim * in_dim] = (
            np.ascontiguousarray(w, np.float32).reshape(-1))
        biases[b_off:b_off + out_dim] = np.ascontiguousarray(b, np.float32)
    return NeuralWeights(cfg.layers, cfg.bins, cfg.hidden, NF_COND,
                         np.array(headers, dtype="<u4"), weights, biases)


def _linears_from_weights(nw: NeuralWeights) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inverse of :func:`_weights_from_linears` — split a flat ``NeuralWeights``
    into per-layer ``(weight[out,in], bias[out])`` arrays in header order."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for hi in range(nw.headers.shape[0]):
        w_off, b_off, in_dim, out_dim = (int(x) for x in nw.headers[hi])
        w = nw.weights[w_off:w_off + out_dim * in_dim].reshape(out_dim, in_dim)
        b = nw.biases[b_off:b_off + out_dim]
        out.append((np.array(w, np.float64), np.array(b, np.float64)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Backend interface + selection
# ─────────────────────────────────────────────────────────────────────────────

class TrainingBackend(abc.ABC):
    """The per-cycle gradient step behind a small stateful interface.

    A backend retains its warm model + optimizer state (e.g. Adam moments)
    across cycles, exactly as the torch loop keeps ``_model/_opt/_scaler``.
    ``NeuralTrainer`` stays the orchestrator (replay sampling, dataset build,
    version/loss bookkeeping, publish); the backend owns only the framework
    work. Mirrors the ``NeuralWeightPublisher`` ABC style.
    """

    name: str = "backend"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True when this backend can actually run on the current host."""

    @abc.abstractmethod
    def supports_precision(self, precision: str, device: str | None = None) -> bool:
        """True when the backend/device can train at ``precision`` (fp32|fp16)."""

    @abc.abstractmethod
    def warm_start(self, weights: NeuralWeights, cfg) -> None:
        """Build the live model once, warm-started from ``weights``."""

    @abc.abstractmethod
    def update(self, cond: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
        """Run ``steps_per_cycle`` optimizer steps; return the last loss."""

    @abc.abstractmethod
    def export(self) -> NeuralWeights:
        """Bake the live params into a fp32 NFW1 ``NeuralWeights``, in memory."""


def _resolve_spline_flow(explicit: str | None) -> str | None:
    """Locate the spline_flow repo (has train.py / export_weights.py): explicit
    arg, then ``SKINNY_SPLINE_FLOW`` env, then a sibling of the skinny repo."""
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("SKINNY_SPLINE_FLOW")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    for up in here.parents[2:6]:
        cands.append(up.parent / "spline_flow")
    for c in cands:
        try:
            if (c / "train.py").exists():
                return str(c)
        except OSError:
            continue
    return None


def _torch_cuda() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _mlx_metal() -> bool:
    """True when Apple MLX is importable on an Apple-Silicon Metal host — the gate
    for the MLX training backend (mirrors ``_torch_cuda`` for the CUDA backend).
    The ``import mlx.core`` stays inside the function so MLX is never a hard
    dependency: a host without the optional ``[mlx]`` extra simply reports False."""
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001
        return False
    try:
        import platform
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            return False
        try:
            return bool(mx.metal.is_available())
        except AttributeError:
            # newer MLX folds the Metal check into the unified device API
            return bool(mx.default_device().type == mx.DeviceType.gpu)
    except Exception:  # noqa: BLE001
        return False


# Token → human description, mirroring registry.py's name-keyed plugin dicts.
# The CLI exposes these tokens (--neural-trainer); they map onto the framework
# backend classes inside make_training_backend.
TRAINING_BACKENDS: dict[str, str] = {
    "cpu": "numpy reference oracle (torch-free, always available)",
    "cuda": "torch on CUDA (the training box)",
    "mlx": "Apple MLX on Apple-Silicon Metal (the Mac training GPU)",
}


def make_training_backend(kind: str = "auto", *, device: str = "auto",
                          train_precision: str = "fp32",
                          spline_flow_path: str | None = None) -> TrainingBackend:
    """Build the training backend for a source token, mirroring ``make_publisher``.

    ``cpu`` → :class:`NumpyTrainingBackend`, ``cuda`` → :class:`TorchTrainingBackend`
    on CUDA, ``mlx`` → :class:`MlxTrainingBackend` on an Apple-Silicon Metal host.
    ``auto`` precedence is ``cuda > mlx > cpu``: it picks ``cuda`` when torch + a
    CUDA device are present, else ``mlx`` when the optional ``mlx`` extra is
    importable on an Apple-Silicon Metal host, else the always-available numpy
    oracle. An explicitly requested token the host cannot provide raises a clear
    error rather than silently degrading.
    """
    if kind == "auto":
        kind = "cuda" if _torch_cuda() else "mlx" if _mlx_metal() else "cpu"

    if kind == "cpu":
        return NumpyTrainingBackend(train_precision=train_precision)

    if kind == "mlx":
        be = MlxTrainingBackend(train_precision=train_precision)
        if not be.is_available():
            try:
                import mlx.core  # noqa: F401
                has_mlx = True
            except Exception:  # noqa: BLE001
                has_mlx = False
            missing = ("the mlx package (install the '[mlx]' extra)" if not has_mlx
                       else "an Apple-Silicon Metal device")
            raise RuntimeError(
                f"--neural-trainer mlx requested but {missing} is unavailable; "
                f"use 'cpu' (numpy reference) or 'auto'")
        return be

    if kind == "cuda":
        be = TorchTrainingBackend(
            device=("cuda" if device == "auto" else device),
            train_precision=train_precision, spline_flow_path=spline_flow_path)
        if not be.is_available():
            try:
                import torch  # noqa: F401
                has_torch = True
            except Exception:  # noqa: BLE001
                has_torch = False
            missing = ("PyTorch" if not has_torch
                       else "a CUDA device" if not _torch_cuda()
                       else "the spline_flow repo (set SKINNY_SPLINE_FLOW)")
            raise RuntimeError(
                f"--neural-trainer cuda requested but {missing} is unavailable; "
                f"use 'cpu' (numpy reference) or 'auto'")
        return be

    raise ValueError(
        f"unknown neural trainer backend {kind!r}; "
        f"known: {sorted(TRAINING_BACKENDS)} (+ 'auto')")


# ─────────────────────────────────────────────────────────────────────────────
# Torch backend (adapts the current CUDA loop)
# ─────────────────────────────────────────────────────────────────────────────

class TorchTrainingBackend(TrainingBackend):
    """The real torch loop. Warm-starts a ``ConditionalSplineFlow2D`` from the
    current weights, runs ``steps_per_cycle`` Adam steps of the contribution-
    weighted MLE, and bakes back to ``NeuralWeights`` in memory.

    On CUDA at ``train_precision='fp16'`` the linear-layer GEMMs run under
    ``autocast(fp16)`` + ``GradScaler`` on the tensor cores while the RQ-spline
    math stays fp32 (the renderer's NF_WT/NF_CT boundary). On cpu/mps the
    optimizer runs fp32 (autocast is CUDA-only here; MPS autocast is partial in
    torch — reported as a fall back).
    """

    def __init__(self, *, device: str = "cuda", train_precision: str = "fp32",
                 spline_flow_path: str | None = None):
        self.name = "torch"
        self._device_kind = device          # cpu|mps|cuda|auto
        self._train_precision = train_precision
        self._spline_flow_path = spline_flow_path
        self._torch = None
        self._sf = None
        self._device = None
        self._model = None
        self._opt = None
        self._scaler = None
        self._cfg = None
        self.last_loss: float | None = None

    # ── availability / capability ────────────────────────────────────────

    def _import(self):
        if self._torch is not None and self._sf is not None:
            return True
        try:
            import torch
        except Exception:  # noqa: BLE001
            return False
        sf_dir = _resolve_spline_flow(self._spline_flow_path)
        if sf_dir is None:
            return False
        if sf_dir not in sys.path:
            sys.path.insert(0, sf_dir)
        try:
            from train import ConditionalSplineFlow2D, get_device
        except Exception:  # noqa: BLE001
            return False
        self._torch = torch
        self._sf = {"Flow": ConditionalSplineFlow2D, "get_device": get_device}
        return True

    def is_available(self) -> bool:
        if not self._import():
            return False
        if self._device_kind == "cuda" and not _torch_cuda():
            return False
        return True

    def _resolve_device(self):
        torch = self._torch
        if self._device_kind == "auto":
            return self._sf["get_device"]("auto")
        return torch.device(self._device_kind)

    def supports_precision(self, precision: str, device: str | None = None) -> bool:
        if precision == "fp32":
            return True
        if precision == "fp16":
            dev = device or self._device_kind
            if dev == "auto":
                dev = "cuda" if _torch_cuda() else "cpu"
            return dev == "cuda"        # autocast+GradScaler fp16 = CUDA only here
        return False

    # ── warm model ───────────────────────────────────────────────────────

    def warm_start(self, weights: NeuralWeights, cfg) -> None:
        if not self._import():
            raise RuntimeError("TorchTrainingBackend.warm_start: torch + spline_flow "
                               "unavailable")
        torch = self._torch
        self._cfg = cfg
        self._device = self._resolve_device()
        arch = cfg.arch
        model = self._sf["Flow"](
            cond_dim=NF_COND, num_layers=arch.layers,
            num_bins=arch.bins, hidden=arch.hidden,
        ).to(self._device)
        if np.any(weights.weights):
            hi = 0
            with torch.no_grad():
                for coupling in model.layers:
                    for li in _LINEAR_IDX:
                        w_off, b_off, in_dim, out_dim = (int(x) for x in weights.headers[hi])
                        hi += 1
                        w = weights.weights[w_off:w_off + out_dim * in_dim].reshape(out_dim, in_dim)
                        b = weights.biases[b_off:b_off + out_dim]
                        lin = coupling.net[li]
                        lin.weight.copy_(torch.from_numpy(
                            np.ascontiguousarray(w, np.float32)).to(self._device))
                        lin.bias.copy_(torch.from_numpy(
                            np.ascontiguousarray(b, np.float32)).to(self._device))
        # else: all-zero dummy is a degenerate saddle (uniform spline → 0 grad);
        # keep the flow's default random init so the first update actually moves.
        self._model = model
        self._opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        use_amp = self._use_amp()
        self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _use_amp(self) -> bool:
        return (self._train_precision == "fp16"
                and self._device is not None and self._device.type == "cuda")

    # ── update ───────────────────────────────────────────────────────────

    def update(self, cond: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
        torch = self._torch
        cfg = self._cfg
        dev = self._device
        cond_t = torch.from_numpy(np.ascontiguousarray(cond, np.float32)).to(dev)
        z_t = torch.from_numpy(np.ascontiguousarray(z, np.float32)).to(dev)
        w_t = torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(dev)
        n = cond_t.shape[0]
        if n == 0:
            return None
        bs = min(int(cfg.batch), n)
        use_amp = self._use_amp()
        model, opt, scaler = self._model, self._opt, self._scaler
        model.train()
        last = None
        for _ in range(int(cfg.steps_per_cycle)):
            idx = torch.randint(0, n, (bs,), device=dev)
            c, zb, wb = cond_t[idx], z_t[idx], w_t[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                log_q = model.log_pdf_square(zb, c).squeeze(-1)
                loss = -(wb * log_q).sum() / wb.sum().clamp_min(1e-12)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            last = loss
        if last is not None and bool(torch.isfinite(last)):
            self.last_loss = float(last.detach().cpu())
            return self.last_loss
        return None

    # ── in-memory bake ───────────────────────────────────────────────────

    def export(self) -> NeuralWeights:
        linears: list[tuple[np.ndarray, np.ndarray]] = []
        for coupling in self._model.layers:
            for li in _LINEAR_IDX:
                lin = coupling.net[li]
                linears.append((lin.weight.detach().cpu().numpy().astype(np.float32),
                                lin.bias.detach().cpu().numpy().astype(np.float32)))
        return _weights_from_linears(linears, self._cfg.arch)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny pure-numpy reverse-mode autodiff (for the numpy reference backend)
# ─────────────────────────────────────────────────────────────────────────────

def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class _V:
    """A node on the autodiff tape: a float64 value, its parents, and a VJP."""

    __slots__ = ("value", "parents", "grad_fn", "grad")

    def __init__(self, value, parents=(), grad_fn=None):
        self.value = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.grad_fn = grad_fn
        self.grad = None

    @property
    def shape(self):
        return self.value.shape

    def __add__(self, o):
        o = o if isinstance(o, _V) else _V(o)
        out = _V(self.value + o.value, (self, o))
        out.grad_fn = lambda g: (_unbroadcast(g, self.shape), _unbroadcast(g, o.shape))
        return out

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, _V) else _V(o)
        out = _V(self.value - o.value, (self, o))
        out.grad_fn = lambda g: (_unbroadcast(g, self.shape), _unbroadcast(-g, o.shape))
        return out

    def __rsub__(self, o):
        return (o if isinstance(o, _V) else _V(o)).__sub__(self)

    def __mul__(self, o):
        o = o if isinstance(o, _V) else _V(o)
        a, b = self, o
        out = _V(a.value * b.value, (a, b))
        out.grad_fn = lambda g: (_unbroadcast(g * b.value, a.shape),
                                 _unbroadcast(g * a.value, b.shape))
        return out

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, _V) else _V(o)
        a, b = self, o
        out = _V(a.value / b.value, (a, b))
        out.grad_fn = lambda g: (_unbroadcast(g / b.value, a.shape),
                                 _unbroadcast(-g * a.value / (b.value * b.value), b.shape))
        return out

    def __rtruediv__(self, o):
        return (o if isinstance(o, _V) else _V(o)).__truediv__(self)

    def __neg__(self):
        out = _V(-self.value, (self,))
        out.grad_fn = lambda g: (-g,)
        return out

    def __pow__(self, p: float):
        out = _V(self.value ** p, (self,))
        out.grad_fn = lambda g: (g * p * self.value ** (p - 1),)
        return out


def _backward(loss: _V) -> None:
    topo: list[_V] = []
    seen: set[int] = set()

    def build(v: _V):
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            build(p)
        topo.append(v)

    build(loss)
    for v in topo:
        v.grad = None
    loss.grad = np.ones_like(loss.value)
    for v in reversed(topo):
        if v.grad_fn is None or v.grad is None:
            continue
        for p, g in zip(v.parents, v.grad_fn(v.grad)):
            if g is None:
                continue
            p.grad = g if p.grad is None else p.grad + g


def _linear(x: _V, W: _V, b: _V) -> _V:
    """x[B,in] @ W[out,in].T + b[out] → [B,out]."""
    out = _V(x.value @ W.value.T + b.value, (x, W, b))

    def gf(g):
        return (g @ W.value, g.T @ x.value, g.sum(axis=0))
    out.grad_fn = gf
    return out


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(np.minimum(x, 0.0)) / (1.0 + np.exp(np.minimum(x, 0.0))))


def _silu(x: _V) -> _V:
    s = _sigmoid(x.value)
    out = _V(x.value * s, (x,))
    out.grad_fn = lambda g: (g * (s * (1.0 + x.value * (1.0 - s))),)
    return out


def _softplus(x: _V) -> _V:
    v = np.logaddexp(0.0, x.value)
    s = _sigmoid(x.value)
    out = _V(v, (x,))
    out.grad_fn = lambda g: (g * s,)
    return out


def _softmax_ax1(x: _V) -> _V:
    e = np.exp(x.value - x.value.max(axis=1, keepdims=True))
    sm = e / e.sum(axis=1, keepdims=True)
    out = _V(sm, (x,))
    out.grad_fn = lambda g: ((g - (g * sm).sum(axis=1, keepdims=True)) * sm,)
    return out


def _sum_ax1(x: _V) -> _V:
    out = _V(x.value.sum(axis=1, keepdims=True), (x,))
    out.grad_fn = lambda g: (np.broadcast_to(g, x.shape).copy(),)
    return out


def _sum(x: _V) -> _V:
    out = _V(x.value.sum(), (x,))
    out.grad_fn = lambda g: (np.broadcast_to(g, x.shape).copy(),)
    return out


def _log(x: _V) -> _V:
    out = _V(np.log(x.value), (x,))
    out.grad_fn = lambda g: (g / x.value,)
    return out


def _sqrt(x: _V) -> _V:
    v = np.sqrt(x.value)
    out = _V(v, (x,))
    out.grad_fn = lambda g: (g * 0.5 / np.maximum(v, 1e-12),)
    return out


def _clamp(x: _V, lo: float, hi: float) -> _V:
    out = _V(np.clip(x.value, lo, hi), (x,))
    mask = (x.value >= lo) & (x.value <= hi)
    out.grad_fn = lambda g: (g * mask,)
    return out


def _clamp_min(x: _V, lo: float) -> _V:
    out = _V(np.maximum(x.value, lo), (x,))
    mask = x.value >= lo
    out.grad_fn = lambda g: (g * mask,)
    return out


def _clamp_max(x: _V, hi: float) -> _V:
    out = _V(np.minimum(x.value, hi), (x,))
    mask = x.value <= hi
    out.grad_fn = lambda g: (g * mask,)
    return out


def _col(x: _V) -> _V:
    """[B] → [B,1]."""
    out = _V(x.value.reshape(-1, 1), (x,))
    out.grad_fn = lambda g: (g.reshape(-1),)
    return out


def _concat1(parts: list[_V]) -> _V:
    vals = [p.value for p in parts]
    sizes = [v.shape[1] for v in vals]
    out = _V(np.concatenate(vals, axis=1), tuple(parts))

    def gf(g):
        grads = []
        o = 0
        for s in sizes:
            grads.append(g[:, o:o + s])
            o += s
        return tuple(grads)
    out.grad_fn = gf
    return out


def _slice1(x: _V, s: int, e: int) -> _V:
    out = _V(x.value[:, s:e], (x,))

    def gf(g):
        gx = np.zeros_like(x.value)
        gx[:, s:e] = g
        return (gx,)
    out.grad_fn = gf
    return out


def _cumsum0(x: _V) -> _V:
    """[B,K] → [B,K+1] with a leading zero: out[:,0]=0, out[:,j]=sum_{i<j} x_i."""
    B, K = x.value.shape
    val = np.concatenate([np.zeros((B, 1)), np.cumsum(x.value, axis=1)], axis=1)
    out = _V(val, (x,))

    def gf(g):
        # x_i affects out cols j=i+1..K → dx_i = suffix-sum of g[:,1:].
        gtail = g[:, 1:]
        dx = np.flip(np.cumsum(np.flip(gtail, axis=1), axis=1), axis=1)
        return (dx,)
    out.grad_fn = gf
    return out


def _gather(src: _V, idx: np.ndarray) -> _V:
    """src[B,M] gathered along axis 1 by idx[B] → [B]; idx is non-diff (constant),
    matching torch's non-differentiable gather indices."""
    B = idx.shape[0]
    rows = np.arange(B)
    out = _V(src.value[rows, idx], (src,))

    def gf(g):
        gs = np.zeros_like(src.value)
        np.add.at(gs, (rows, idx), g)
        return (gs,)
    out.grad_fn = gf
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Numpy reference backend (oracle)
# ─────────────────────────────────────────────────────────────────────────────

def _rqs_forward_logdet(x: _V, widths: _V, heights: _V, derivs: _V, K: int,
                        eps: float = 1e-8) -> _V:
    """log|dy/dx| of the monotone RQ spline at ``x`` — the second return of
    spline_flow.rqs_forward. (We only need the log-det here, not y.)"""
    x = _clamp(x, 0.0, 1.0)
    xk = _cumsum0(widths)
    yk = _cumsum0(heights)
    idx = (np.sum(x.value[:, None] >= xk.value, axis=1) - 1).clip(0, K - 1)
    x0 = _gather(xk, idx)
    x1 = _gather(xk, idx + 1)
    y0 = _gather(yk, idx)
    y1 = _gather(yk, idx + 1)
    d0 = _gather(derivs, idx)
    d1 = _gather(derivs, idx + 1)
    width = _clamp_min(x1 - x0, eps)
    height = _clamp_min(y1 - y0, eps)
    delta = height / width
    theta = (x - x0) / width
    one_m = 1.0 - theta
    den = delta + (d0 + d1 - delta * 2.0) * theta * one_m
    dnum = delta * delta * (d1 * theta * theta
                            + delta * 2.0 * theta * one_m
                            + d0 * one_m * one_m)
    dydx = dnum / (_clamp_min(den, eps) ** 2)
    return _log(_clamp_min(dydx, eps))


def _rqs_inverse(y: _V, widths: _V, heights: _V, derivs: _V, K: int,
                 eps: float = 1e-8) -> tuple[_V, _V]:
    """Analytic inverse of the monotone RQ spline (spline_flow.rqs_inverse).
    Returns ``(x, logdet)`` with ``logdet = -log|dy/dx|`` (= log|dx/dy|)."""
    y = _clamp(y, 0.0, 1.0)
    xk = _cumsum0(widths)
    yk = _cumsum0(heights)
    idx = (np.sum(y.value[:, None] >= yk.value, axis=1) - 1).clip(0, K - 1)
    x0 = _gather(xk, idx)
    x1 = _gather(xk, idx + 1)
    y0 = _gather(yk, idx)
    y1 = _gather(yk, idx + 1)
    d0 = _gather(derivs, idx)
    d1 = _gather(derivs, idx + 1)
    width = _clamp_min(x1 - x0, eps)
    height = _clamp_min(y1 - y0, eps)
    delta = height / width
    z = (y - y0) / height
    a = d0 + d1 - delta * 2.0
    A = z * a + delta - d0
    Bc = d0 - z * a
    C = -(delta * z)
    disc = _clamp_min(Bc * Bc - A * C * 4.0, 0.0)
    sqrt_disc = _sqrt(disc)
    theta = (C * 2.0) / _clamp_max(-Bc - sqrt_disc, -eps)
    theta = _clamp(theta, 0.0, 1.0)
    x = x0 + theta * width
    log_dydx = _rqs_forward_logdet(x, widths, heights, derivs, K, eps)
    return _clamp(x, 0.0, 1.0), -log_dydx


class NumpyTrainingBackend(TrainingBackend):
    """Torch-free reference oracle: a hand-written forward + backward of the
    contribution-weighted MLE on the shipped flow, via the tiny numpy autodiff
    tape above. Always available (numpy only); fp32 compute only.

    Warm state — the per-layer Linear weights/biases plus their Adam moments —
    persists across cycles, exactly like the torch backend's model/optimizer.
    """

    def __init__(self, *, train_precision: str = "fp32"):
        self.name = "numpy"
        self._train_precision = train_precision
        self._cfg = None
        self._leaves: list[tuple[_V, _V]] = []   # (W,b) per Linear, header order
        self._m: list[tuple[np.ndarray, np.ndarray]] = []
        self._v: list[tuple[np.ndarray, np.ndarray]] = []
        self._t = 0
        self.last_loss: float | None = None

    def is_available(self) -> bool:
        return True

    def supports_precision(self, precision: str, device: str | None = None) -> bool:
        return precision == "fp32"      # numpy oracle runs full precision only

    # ── warm model ───────────────────────────────────────────────────────

    def warm_start(self, weights: NeuralWeights, cfg) -> None:
        self._cfg = cfg
        arch = cfg.arch
        if np.any(weights.weights):
            linears = _linears_from_weights(weights)
        else:
            # all-zero dummy is a degenerate saddle (uniform spline → 0 grad);
            # seed a deterministic torch-style Linear init so updates move.
            rng = np.random.default_rng(0)
            headers, _, _ = _layout(arch.layers, arch.bins, arch.hidden)
            linears = []
            for _w_off, _b_off, in_dim, out_dim in headers:
                k = 1.0 / in_dim
                b = np.sqrt(k)
                linears.append((rng.uniform(-b, b, (out_dim, in_dim)),
                                rng.uniform(-b, b, out_dim)))
        self._leaves = [(_V(w), _V(b)) for w, b in linears]
        self._m = [(np.zeros_like(W.value), np.zeros_like(B.value))
                   for W, B in self._leaves]
        self._v = [(np.zeros_like(W.value), np.zeros_like(B.value))
                   for W, B in self._leaves]
        self._t = 0

    # ── forward log q ────────────────────────────────────────────────────

    def _log_pdf_square(self, z_np: np.ndarray, cond_np: np.ndarray) -> _V:
        """log q on the unit square for the shipped dim=2 flow — mirrors
        ConditionalSplineFlow2D.log_pdf_square (inverse over reversed layers)."""
        cfg = self._cfg.arch
        K = cfg.bins
        cond_v = _V(cond_np)
        cols = [_V(z_np[:, 0]), _V(z_np[:, 1])]   # the two square coords
        logdet: _V | None = None
        for i in reversed(range(cfg.layers)):
            mask_even = (i % 2 == 0)
            cond_col = 0 if mask_even else 1      # passthrough
            trans_col = 1 if mask_even else 0     # transformed
            W0, b0 = self._leaves[3 * i + 0]
            W1, b1 = self._leaves[3 * i + 1]
            W2, b2 = self._leaves[3 * i + 2]
            x_in = _concat1([_col(cols[cond_col]), cond_v])      # [B, 1+cond]
            h = _silu(_linear(x_in, W0, b0))
            h = _silu(_linear(h, W1, b1))
            raw = _linear(h, W2, b2)                              # [B, 3K+1]
            wsm = _softmax_ax1(_slice1(raw, 0, K)) + _MIN_BIN
            widths = wsm / _sum_ax1(wsm)
            hsm = _softmax_ax1(_slice1(raw, K, 2 * K)) + _MIN_BIN
            heights = hsm / _sum_ax1(hsm)
            derivs = _softplus(_slice1(raw, 2 * K, 3 * K + 1)) + _DERIV_FLOOR
            new_tr, ld = _rqs_inverse(cols[trans_col], widths, heights, derivs, K)
            cols[trans_col] = new_tr
            logdet = ld if logdet is None else logdet + ld
        return logdet

    # ── update ───────────────────────────────────────────────────────────

    def update(self, cond: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
        cfg = self._cfg
        cond = np.ascontiguousarray(cond, np.float64)
        z = np.ascontiguousarray(z, np.float64)
        w = np.ascontiguousarray(w, np.float64)
        n = cond.shape[0]
        if n == 0:
            return None
        bs = min(int(cfg.batch), n)
        rng = np.random.default_rng(self._t + 1)
        last = None
        for _ in range(int(cfg.steps_per_cycle)):
            idx = rng.integers(0, n, bs)
            c, zb, wb = cond[idx], z[idx], w[idx]
            log_q = self._log_pdf_square(zb, c)              # [B]
            wb_v = _V(wb)
            loss = -_sum(wb_v * log_q) / _clamp_min(_V(wb.sum()), 1e-12)
            _backward(loss)
            self._adam_step()
            last = float(loss.value)
        if last is not None and np.isfinite(last):
            self.last_loss = last
            return last
        return None

    def _adam_step(self, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0) -> None:
        cfg = self._cfg
        lr = cfg.lr
        # global grad-norm clip over all leaves (clip_grad_norm_, norm_type=2).
        total_sq = 0.0
        for W, B in self._leaves:
            if W.grad is not None:
                total_sq += float(np.sum(W.grad ** 2))
            if B.grad is not None:
                total_sq += float(np.sum(B.grad ** 2))
        total = np.sqrt(total_sq)
        scale = clip / (total + 1e-6) if total > clip else 1.0
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i, (W, B) in enumerate(self._leaves):
            for j, p in enumerate((W, B)):
                if p.grad is None:
                    continue
                g = p.grad * scale
                m = self._m[i][j]
                v = self._v[i][j]
                m[...] = beta1 * m + (1.0 - beta1) * g
                v[...] = beta2 * v + (1.0 - beta2) * (g * g)
                mhat = m / bc1
                vhat = v / bc2
                p.value -= lr * mhat / (np.sqrt(vhat) + eps)

    # ── in-memory bake ───────────────────────────────────────────────────

    def export(self) -> NeuralWeights:
        linears = [(W.value, B.value) for W, B in self._leaves]
        return _weights_from_linears(linears, self._cfg.arch)


# ─────────────────────────────────────────────────────────────────────────────
# Apple MLX backend (Metal GPU; change mlx-neural-trainer)
# ─────────────────────────────────────────────────────────────────────────────

class MlxTrainingBackend(TrainingBackend):
    """GPU training on Apple-Silicon Metal via Apple MLX. Warm-starts the shipped
    flow from the current weights, runs ``steps_per_cycle`` Adam steps of the
    contribution-weighted MLE on the Metal GPU, and bakes back to fp32 NFW1.

    The flow math mirrors :class:`NumpyTrainingBackend` — the parity target —
    op-for-op on ``mlx.core`` arrays, but lets MLX autodiff produce the gradients
    (no hand-written backward) and applies a hand-rolled, bias-corrected Adam with
    the same global grad-norm clip so a one-step update from identical weights
    matches the numpy oracle within tolerance. At ``train_precision='fp16'`` the
    flow runs in ``float16`` over fp32 master weights (D5): the loss accumulates
    in fp32, gradients land back on the fp32 leaves, and a non-finite fp16 loss
    falls the session back to fp32 with a one-time warning. ``export`` always
    bakes fp32, so the handoff format is unchanged.
    """

    def __init__(self, *, train_precision: str = "fp32"):
        self.name = "mlx"
        self._train_precision = train_precision
        self._mx = None
        self._cfg = None
        self._leaves: list = []            # flat [W0,b0,W1,b1,...] mx arrays, header order
        self._m: list = []                 # Adam 1st moments, parallel to _leaves
        self._v: list = []                 # Adam 2nd moments, parallel to _leaves
        self._t = 0
        self._fp16_disabled = False        # set once if fp16 loss goes non-finite
        self.last_loss: float | None = None
        # Single owned worker thread. MLX arrays + GPU streams are thread-affine:
        # the leaf arrays are bound to whichever thread first creates them, and
        # touching them from another thread raises "There is no Stream(gpu, N) in
        # current thread." The backend can legitimately be driven from more than
        # one thread (a direct call concurrent with the background daemon
        # trainer), so every MLX-touching op is marshaled onto this one executor
        # thread — created lazily, reused for the backend's lifetime, so all
        # warm_start / update / export work lands on the same thread.
        self._exec = None

    # ── availability / capability ────────────────────────────────────────

    def _import(self) -> bool:
        if self._mx is not None:
            return True
        if not _mlx_metal():
            return False
        import mlx.core as mx
        self._mx = mx
        return True

    def is_available(self) -> bool:
        return _mlx_metal()

    def supports_precision(self, precision: str, device: str | None = None) -> bool:
        # fp32 always; fp16 runs as float16 compute over fp32 masters (D5), with a
        # runtime fall back to fp32 if a step's loss becomes non-finite.
        return precision in ("fp32", "fp16")

    # ── single-thread marshalling ─────────────────────────────────────────

    def _run(self, fn, *args):
        """Run ``fn`` on the backend's single owned worker thread and block for
        its result, so all MLX work stays on one thread (see __init__). The
        ``max_workers=1`` executor reuses the same worker across calls, and
        exceptions raised inside ``fn`` re-raise here on the caller — preserving
        the public methods' error contract."""
        if self._exec is None:
            from concurrent.futures import ThreadPoolExecutor
            self._exec = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlx-backend")
        return self._exec.submit(fn, *args).result()

    # ── warm model ───────────────────────────────────────────────────────

    def warm_start(self, weights: NeuralWeights, cfg) -> None:
        return self._run(self._warm_start_impl, weights, cfg)

    def _warm_start_impl(self, weights: NeuralWeights, cfg) -> None:
        if not self._import():
            raise RuntimeError("MlxTrainingBackend.warm_start: mlx unavailable "
                               "(needs the '[mlx]' extra on an Apple-Silicon Metal host)")
        mx = self._mx
        self._cfg = cfg
        arch = cfg.arch
        if np.any(weights.weights):
            linears = _linears_from_weights(weights)
        else:
            # all-zero dummy is a degenerate saddle (uniform spline → 0 grad);
            # seed the same deterministic torch-style Linear init the numpy oracle
            # uses so the first update actually moves and the two stay comparable.
            rng = np.random.default_rng(0)
            headers, _, _ = _layout(arch.layers, arch.bins, arch.hidden)
            linears = []
            for _w_off, _b_off, in_dim, out_dim in headers:
                k = 1.0 / in_dim
                b = np.sqrt(k)
                linears.append((rng.uniform(-b, b, (out_dim, in_dim)),
                                rng.uniform(-b, b, out_dim)))
        # Flat leaf list in header order: [W0, b0, W1, b1, ...]; fp32 masters.
        self._leaves = []
        for w, b in linears:
            self._leaves.append(mx.array(np.ascontiguousarray(w, np.float32)))
            self._leaves.append(mx.array(np.ascontiguousarray(b, np.float32)))
        self._m = [mx.zeros(p.shape, dtype=mx.float32) for p in self._leaves]
        self._v = [mx.zeros(p.shape, dtype=mx.float32) for p in self._leaves]
        self._t = 0
        mx.eval(self._leaves, self._m, self._v)

    # ── flow forward (log q) — mirrors NumpyTrainingBackend._log_pdf_square ──

    def _gather1(self, src, idx):
        """src[B,M] gathered along axis 1 by idx[B] (int) → [B]. idx is computed
        in-graph from integer comparisons, so it carries no gradient — matching the
        numpy oracle's stop-gradient on the spline-bin selection."""
        mx = self._mx
        return mx.take_along_axis(src, idx.reshape(-1, 1), axis=1).reshape(-1)

    def _cumsum0(self, x):
        """[B,K] → [B,K+1] with a leading zero column."""
        mx = self._mx
        B = x.shape[0]
        return mx.concatenate([mx.zeros((B, 1), dtype=x.dtype), mx.cumsum(x, axis=1)],
                              axis=1)

    def _bin_index(self, q, knots, K):
        """idx = (#{knots[:,j] <= q}) - 1, clipped to [0, K-1]. Integer, non-diff."""
        mx = self._mx
        ge = (q.reshape(-1, 1) >= knots).astype(mx.int32)
        return mx.clip(mx.sum(ge, axis=1) - 1, 0, K - 1)

    def _rqs_forward_logdet(self, x, widths, heights, derivs, K, eps=1e-8):
        mx = self._mx
        x = mx.clip(x, 0.0, 1.0)
        xk = self._cumsum0(widths)
        yk = self._cumsum0(heights)
        idx = self._bin_index(x, xk, K)
        x0, x1 = self._gather1(xk, idx), self._gather1(xk, idx + 1)
        y0, y1 = self._gather1(yk, idx), self._gather1(yk, idx + 1)
        d0, d1 = self._gather1(derivs, idx), self._gather1(derivs, idx + 1)
        width = mx.maximum(x1 - x0, eps)
        height = mx.maximum(y1 - y0, eps)
        delta = height / width
        theta = (x - x0) / width
        one_m = 1.0 - theta
        den = delta + (d0 + d1 - delta * 2.0) * theta * one_m
        dnum = delta * delta * (d1 * theta * theta
                                + delta * 2.0 * theta * one_m
                                + d0 * one_m * one_m)
        dydx = dnum / (mx.maximum(den, eps) ** 2)
        return mx.log(mx.maximum(dydx, eps))

    def _rqs_inverse(self, y, widths, heights, derivs, K, eps=1e-8):
        """Analytic inverse of the monotone RQ spline — mirrors numpy
        ``_rqs_inverse``. Returns ``(x, logdet)`` with ``logdet = log|dx/dy|``."""
        mx = self._mx
        y = mx.clip(y, 0.0, 1.0)
        xk = self._cumsum0(widths)
        yk = self._cumsum0(heights)
        idx = self._bin_index(y, yk, K)
        x0 = self._gather1(xk, idx)
        y0, y1 = self._gather1(yk, idx), self._gather1(yk, idx + 1)
        d0, d1 = self._gather1(derivs, idx), self._gather1(derivs, idx + 1)
        width = mx.maximum(self._gather1(xk, idx + 1) - x0, eps)
        height = mx.maximum(y1 - y0, eps)
        delta = height / width
        z = (y - y0) / height
        a = d0 + d1 - delta * 2.0
        A = z * a + delta - d0
        Bc = d0 - z * a
        C = -(delta * z)
        disc = mx.maximum(Bc * Bc - A * C * 4.0, 0.0)
        sqrt_disc = mx.sqrt(disc)
        theta = (C * 2.0) / mx.minimum(-Bc - sqrt_disc, -eps)
        theta = mx.clip(theta, 0.0, 1.0)
        x = x0 + theta * width
        logdet = -self._rqs_forward_logdet(x, widths, heights, derivs, K, eps)
        return mx.clip(x, 0.0, 1.0), logdet

    def _log_pdf_square(self, leaves, z, cond):
        """log q on the unit square for the shipped dim=2 flow — the MLX mirror of
        :meth:`NumpyTrainingBackend._log_pdf_square` (uniform base ⇒ log q = logdet)."""
        mx = self._mx
        cfg = self._cfg.arch
        K = cfg.bins
        cols = [z[:, 0], z[:, 1]]
        logdet = None
        for i in reversed(range(cfg.layers)):
            mask_even = (i % 2 == 0)
            cond_col = 0 if mask_even else 1
            trans_col = 1 if mask_even else 0
            W0, b0 = leaves[6 * i + 0], leaves[6 * i + 1]
            W1, b1 = leaves[6 * i + 2], leaves[6 * i + 3]
            W2, b2 = leaves[6 * i + 4], leaves[6 * i + 5]
            x_in = mx.concatenate([cols[cond_col].reshape(-1, 1), cond], axis=1)
            h = self._silu(x_in @ W0.T + b0)
            h = self._silu(h @ W1.T + b1)
            raw = h @ W2.T + b2                                  # [B, 3K+1]
            wsm = mx.softmax(raw[:, 0:K], axis=1) + _MIN_BIN
            widths = wsm / mx.sum(wsm, axis=1, keepdims=True)
            hsm = mx.softmax(raw[:, K:2 * K], axis=1) + _MIN_BIN
            heights = hsm / mx.sum(hsm, axis=1, keepdims=True)
            derivs = self._softplus(raw[:, 2 * K:3 * K + 1]) + _DERIV_FLOOR
            new_tr, ld = self._rqs_inverse(cols[trans_col], widths, heights, derivs, K)
            cols[trans_col] = new_tr
            logdet = ld if logdet is None else logdet + ld
        return logdet

    def _silu(self, x):
        mx = self._mx
        return x * mx.sigmoid(x)

    def _softplus(self, x):
        mx = self._mx
        return mx.logaddexp(mx.zeros_like(x), x)

    # ── update ───────────────────────────────────────────────────────────

    def update(self, cond: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
        return self._run(self._update_impl, cond, z, w)

    def _update_impl(self, cond: np.ndarray, z: np.ndarray, w: np.ndarray) -> float | None:
        mx = self._mx
        cfg = self._cfg
        n = cond.shape[0]
        if n == 0:
            return None
        cond = np.ascontiguousarray(cond, np.float32)
        z = np.ascontiguousarray(z, np.float32)
        w = np.ascontiguousarray(w, np.float32)
        bs = min(int(cfg.batch), n)
        use_fp16 = (self._train_precision == "fp16" and not self._fp16_disabled)
        cdt = mx.float16 if use_fp16 else mx.float32

        def loss_fn(leaves, zb, cb, wb):
            lv = [p.astype(cdt) for p in leaves] if use_fp16 else leaves
            log_q = self._log_pdf_square(lv, zb.astype(cdt), cb.astype(cdt))
            log_q = log_q.astype(mx.float32)
            return -mx.sum(wb * log_q) / mx.maximum(mx.sum(wb), 1e-12)

        grad_fn = mx.value_and_grad(loss_fn)
        rng = np.random.default_rng(self._t + 1)
        last = None
        for _ in range(int(cfg.steps_per_cycle)):
            idx = rng.integers(0, n, bs)
            zb = mx.array(z[idx])
            cb = mx.array(cond[idx])
            wb = mx.array(w[idx])
            loss, grads = grad_fn(self._leaves, zb, cb, wb)
            # fp16 can overflow in the BACKWARD while the forward loss stays
            # finite, so the step is gated on the gradients' global norm too —
            # never apply a non-finite step (a 0·∞ in the grad-norm clip would
            # poison the fp32 masters and make the fall-back unrecoverable).
            gnorm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in grads))
            lf = float(loss.item())
            step_finite = np.isfinite(lf) and bool(mx.isfinite(gnorm).item())
            if not step_finite:
                if use_fp16:
                    # fp16 went non-finite — disable fp16 for the rest of the
                    # session (D5) and retry in fp32 from the next iteration.
                    self._fp16_disabled = True
                    use_fp16 = False
                    cdt = mx.float32
                    grad_fn = mx.value_and_grad(loss_fn)
                    print("[neural] mlx fp16 training step non-finite; "
                          "falling back to fp32 for the rest of the session")
                # fp32 non-finite is variance on a degenerate batch — drop the
                # step (weights untouched) rather than corrupt the masters.
                continue
            self._adam_step(grads, gnorm=gnorm)
            last = lf
        if last is not None and np.isfinite(last):
            self.last_loss = last
            return last
        return None

    def _adam_step(self, grads, *, gnorm, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        """Hand-rolled bias-corrected Adam with a global grad-norm clip — matches
        :meth:`NumpyTrainingBackend._adam_step` so MLX↔numpy stay comparable.
        ``gnorm`` is the global L2 grad-norm the caller already computed (and
        verified finite)."""
        mx = self._mx
        cfg = self._cfg
        lr = cfg.lr
        total = gnorm
        scale = mx.minimum(clip / (total + 1e-6), 1.0)   # only shrinks when total>clip
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i, g in enumerate(grads):
            g = g.astype(mx.float32) * scale
            m = beta1 * self._m[i] + (1.0 - beta1) * g
            v = beta2 * self._v[i] + (1.0 - beta2) * (g * g)
            self._m[i] = m
            self._v[i] = v
            mhat = m / bc1
            vhat = v / bc2
            self._leaves[i] = self._leaves[i] - lr * mhat / (mx.sqrt(vhat) + eps)
        mx.eval(self._leaves, self._m, self._v)   # D2: pay the step cost here

    # ── in-memory bake ───────────────────────────────────────────────────

    def export(self) -> NeuralWeights:
        return self._run(self._export_impl)

    def _export_impl(self) -> NeuralWeights:
        linears = []
        for k in range(len(self._leaves) // 2):
            W = np.array(self._leaves[2 * k]).astype(np.float32)
            b = np.array(self._leaves[2 * k + 1]).astype(np.float32)
            linears.append((W, b))
        return _weights_from_linears(linears, self._cfg.arch)
