"""Async trainer for online neural-proposal training (Stage 2).

Change ``neural-online-training``. Pulls recency-weighted batches from the
``ReplayBuffer`` and does small warm-started updates on the **exact** shipped flow
architecture (``ConditionalSplineFlow2D(cond=9, layers=6, bins=24, hidden=96)``)
using the contribution-weighted MLE loss the offline trainer uses
(``spline_flow/render_records.py``: ``loss = -Σ w·log q / Σ w``,
``w = luminance(contribution)``). It then bakes a new ``NeuralWeights`` the
publisher hands to the renderer.

Two-tier implementation (task 2.2):

* **torch + spline_flow present** (the training box — CUDA on the NVIDIA box, or
  CPU/MPS for CI): the real loop. It warm-starts a ``ConditionalSplineFlow2D`` from
  the current ``NeuralWeights``, runs ``steps_per_cycle`` Adam steps of the verified
  ``render_records`` contribution-weighted MLE on a recency-weighted batch, then
  bakes the result back through the shipped ``export_weights`` ↔ ``NFW1`` format.
  On CUDA it trains under ``autocast(fp16)`` + ``GradScaler`` — the linear-layer
  GEMMs run on the tensor cores in fp16 while the RQ-spline math stays fp32, the
  same NF_WT/NF_CT boundary the renderer uses at inference.
* **torch absent** (the renderer-only / Mac-CI venv): a placeholder update that
  returns valid, correctly-sized weights so the end-to-end online loop
  (drain → train → publish → swap → ``networkVersion++``) stays exercisable and the
  module imports clean with no PyTorch dependency.

``spline_flow`` is a sibling repo put on ``sys.path`` (``SKINNY_SPLINE_FLOW`` env or
``TrainerConfig.spline_flow_path``, else auto-discovered beside the skinny repo).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .neural_replay import ReplayBuffer  # noqa: F401  (type clarity for callers)
from .neural_weights import (
    NF_COND,
    NeuralBuildConfig,
    NeuralWeights,
    load_neural_weights,
    make_dummy_weights,
)

__all__ = ["TrainerConfig", "NeuralTrainer"]

# The three Linear layers inside each SplineCoupling.net (Sequential); 1/3 are
# SiLU. Mirrors spline_flow/export_weights.py LINEAR_IDX and neural_weights._layout.
_LINEAR_IDX = (0, 2, 4)


@dataclass
class TrainerConfig:
    arch: NeuralBuildConfig = field(default_factory=NeuralBuildConfig)
    steps_per_cycle: int = 64          # small: smooth animation → cheap warm updates
    batch: int = 4096
    lr: float = 1e-3
    device: str = "auto"               # cpu|mps|cuda|auto; cuda path = NVIDIA box
    fp16: bool = True                  # CUDA tensor-core autocast (device branch)
    # Scene AABB (bmin, bext) for the position condition — must match the
    # renderer's neuralCondition normalisation; None ⇒ unit cube (raw position).
    bounds: tuple | None = None
    # spline_flow repo location (has train.py / render_records.py / export_weights.py).
    spline_flow_path: str | None = None


def _resolve_spline_flow(explicit: str | None) -> str | None:
    """Locate the spline_flow repo: explicit arg, then ``SKINNY_SPLINE_FLOW`` env,
    then a sibling of the skinny repo. Returns the dir containing train.py, or None."""
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


class NeuralTrainer:
    """Warm-started online trainer. Holds the current weights (and, on a torch
    box, a persistent warm flow + optimiser); each cycle updates them from recent
    records and returns the new weights to publish."""

    def __init__(self, config: TrainerConfig | None = None,
                 initial: NeuralWeights | None = None):
        self.config = config or TrainerConfig()
        self._weights = initial or make_dummy_weights(self.config.arch)
        self._cycles = 0
        # torch backend, probed lazily on the first cycle (None ⇒ placeholder).
        self._probed = False
        self._torch = None
        self._sf: dict | None = None
        self._device = None
        self._model = None
        self._opt = None
        self._scaler = None
        self.last_loss: float | None = None

    @property
    def weights(self) -> NeuralWeights:
        return self._weights

    @property
    def torch_active(self) -> bool:
        """True when the real PyTorch loop is available (torch + spline_flow)."""
        self._probe()
        return self._torch is not None

    # ── backend probe ───────────────────────────────────────────────────

    def _probe(self) -> None:
        if self._probed:
            return
        self._probed = True
        try:
            import torch
        except Exception:  # noqa: BLE001 — torch-free venv → placeholder
            return
        sf_dir = _resolve_spline_flow(self.config.spline_flow_path)
        if sf_dir is None:
            return
        if sf_dir not in sys.path:
            sys.path.insert(0, sf_dir)
        try:
            from train import ConditionalSplineFlow2D, get_device
            from export_weights import export_flow
            from render_records import build_dataset
        except Exception:  # noqa: BLE001 — spline_flow import failed → placeholder
            return
        self._torch = torch
        self._sf = {
            "Flow": ConditionalSplineFlow2D,
            "build_dataset": build_dataset,
            "export_flow": export_flow,
        }
        dev = self.config.device
        self._device = get_device("auto") if dev == "auto" else torch.device(dev)

    # ── public cycle ────────────────────────────────────────────────────

    def train_cycle(self, replay: ReplayBuffer,
                    rng: np.random.Generator | None = None) -> NeuralWeights:
        """Run one warm-started training cycle on recent records; return new weights.

        With torch + spline_flow available this is the real contribution-weighted
        MLE update (CUDA + autocast-fp16 on the NVIDIA box). Without them it is the
        placeholder update that keeps the online loop honest end to end."""
        self._cycles += 1
        batch = replay.sample(self.config.batch, rng)
        if len(batch) == 0:
            return self._weights  # nothing to learn from yet
        self._probe()
        if self._torch is None:
            return self._placeholder_update()
        return self._train_cycle_torch(batch)

    # ── placeholder (torch-free) ────────────────────────────────────────

    def _placeholder_update(self) -> NeuralWeights:
        """Produce valid, correctly-sized, finite weights without training so
        publish→swap→networkVersion++ is exercised on a torch-free box."""
        nw = self._weights
        new = NeuralWeights(
            nw.layers, nw.bins, nw.hidden, nw.cond,
            nw.headers.copy(), nw.weights.copy(), nw.biases.copy(),
        )
        self._weights = new
        return new

    # ── real torch loop ─────────────────────────────────────────────────

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.bounds is not None:
            bmin, bext = self.config.bounds
            return (np.asarray(bmin, np.float32).reshape(3),
                    np.asarray(bext, np.float32).reshape(3))
        return np.zeros(3, np.float32), np.ones(3, np.float32)

    def _build_model(self):
        """Construct the flow at the configured arch and warm-start it from the
        current weights (the inverse of export_weights' flat layout)."""
        torch = self._torch
        cfg = self.config.arch
        model = self._sf["Flow"](
            cond_dim=NF_COND, num_layers=cfg.layers,
            num_bins=cfg.bins, hidden=cfg.hidden,
        ).to(self._device)
        nw = self._weights
        if not np.any(nw.weights):
            # The all-zero dummy net is a degenerate saddle (uniform spline →
            # log q ≈ 0 → zero gradient): warm-starting from it never moves.
            # Start from the flow's default random init instead; a real (non-zero)
            # warm-start net loads below.
            return model
        hi = 0
        with torch.no_grad():
            for coupling in model.layers:
                for li in _LINEAR_IDX:
                    w_off, b_off, in_dim, out_dim = (int(x) for x in nw.headers[hi])
                    hi += 1
                    w = nw.weights[w_off:w_off + out_dim * in_dim].reshape(out_dim, in_dim)
                    b = nw.biases[b_off:b_off + out_dim]
                    lin = coupling.net[li]
                    lin.weight.copy_(
                        torch.from_numpy(np.ascontiguousarray(w, np.float32)).to(self._device))
                    lin.bias.copy_(
                        torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(self._device))
        return model

    def _train_cycle_torch(self, batch: np.ndarray) -> NeuralWeights:
        torch = self._torch
        if self._model is None:
            self._model = self._build_model()
            self._opt = torch.optim.Adam(self._model.parameters(), lr=self.config.lr)
            use_amp = self.config.fp16 and self._device.type == "cuda"
            self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        bmin, bext = self._bounds()
        cond, z, w = self._sf["build_dataset"](batch, bmin, bext, self._device)
        if cond.shape[0] == 0:
            return self._weights  # no upper-hemisphere, positive-weight samples

        self._run_steps(cond, z, w)
        self._weights = self._bake()
        return self._weights

    def _run_steps(self, cond, z, w) -> None:
        torch = self._torch
        model, opt, scaler = self._model, self._opt, self._scaler
        n = cond.shape[0]
        bs = min(self.config.batch, n)
        use_amp = self.config.fp16 and self._device.type == "cuda"
        model.train()
        last = None
        for _ in range(int(self.config.steps_per_cycle)):
            idx = torch.randint(0, n, (bs,), device=self._device)
            c, zb, wb = cond[idx], z[idx], w[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # Linear GEMMs run fp16 on the tensor cores under autocast; the
                # RQ-spline solve + log-det stay fp32 (autocast leaves non-GEMM
                # ops alone) — the renderer's NF_WT/NF_CT boundary.
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

    def _bake(self) -> NeuralWeights:
        """Bake the live flow back to a ``NeuralWeights`` through the shipped
        export_weights ↔ NFW1 round-trip (one source of truth for the layout)."""
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".nfw1")
        os.close(fd)
        try:
            self._sf["export_flow"](self._model, path)
            return load_neural_weights(path, expect=self.config.arch.arch)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
