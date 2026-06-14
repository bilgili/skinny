#!/usr/bin/env python3
"""Regenerate the committed neural-flow parity goldens (OpenSpec task 1.4).

These goldens lock the Slang port of the conditional rational-quadratic neural
spline flow (``src/skinny/shaders/sampling/neural_flow.slang``) against the
PyTorch reference (``spline_flow/train.py``: ``ConditionalSplineFlow2D`` +
``square_to_hemisphere``). ``tests/test_neural_parity.py`` re-implements the
Slang inference in numpy (no torch, no GPU) and asserts it matches the PyTorch
forward (wi, solid-angle pdf) baked here, plus exercises the spline inverse.

This script is the ONLY part of the chain that needs PyTorch. It is NOT run in
CI; it is committed so the goldens are reproducible. Run it with the spline_flow
torch venv (Python 3.13 + torch), from the spline_flow repo so ``train.py`` /
``export_weights.py`` import::

    cd /Users/ahmetbilgili/projects/spline_flow
    .venv/bin/python \
        /Users/ahmetbilgili/projects/skinny/.claude/worktrees/neural-directional-proposal/\
tests/data/neural_parity/generate_goldens.py

Outputs (committed next to this script):
  * ``weights.bin``  — the baked NFW1 net (``export_weights.export_flow``);
                       the exact bytes the renderer's host loader + Slang read.
  * ``goldens.npz``  — fixed inputs + PyTorch reference outputs:
      forward_cond  [N,9] float32   condition vectors in [-1, 1]
      forward_u     [N,2] float32   base samples in [0.02, 0.98]
      forward_wi    [N,3] float32   PyTorch forward hemisphere direction (y-up)
      forward_pdf   [N]   float64   PyTorch forward solid-angle pdf q_omega
      inverse_cond  [M,9] float32   condition vectors for the round-trip test
      inverse_u     [M,2] float32   base samples driving the round-trip
      inverse_wi    [M,3] float32   forward(u) direction (in the upper hemisphere)
      inverse_pdf   [M]   float64   pdf the PyTorch INVERSE assigns to inverse_wi

Determinism: torch weights come from ``torch.manual_seed(SEED_TORCH)``; the
(cond, u) inputs come from a separate ``numpy`` Generator (``SEED_INPUTS``) so
the fixed inputs are independent of torch's RNG stream and stable across torch
versions. The architecture is pinned to the renderer's compiled-in constants
(cond_dim=9, num_layers=6, num_bins=24, hidden=96).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from train import ConditionalSplineFlow2D
from export_weights import export_flow

# Output chart for the baked net (change directional-flow-parameterization). The
# renderer's neural_flow.slang uses the Lambert azimuthal equal-area chart (V1),
# so the goldens — and the appended NFW1 chart tag — must be V1 too. V1 is
# equal-area like V0 (|J| = 2π), so only the baked direction (wi) differs from the
# retired V0 goldens; the solid-angle pdf is chart-independent and unchanged.
CHART_NAME = "V1"

# Architecture — MUST match shaders/sampling/neural_flow.slang (NF_* consts)
# and src/skinny/sampling/neural_weights.py.
COND_DIM = 9
NUM_LAYERS = 6
NUM_BINS = 24
HIDDEN = 96

SEED_TORCH = 0       # weights (matches spline_flow/parity_check.py)
SEED_INPUTS = 1      # fixed (cond, u) draws — independent of the torch stream
N_FORWARD = 64       # forward parity cases
N_INVERSE = 48       # inverse / round-trip cases

LOG2PI = math.log(2.0 * math.pi)

HERE = Path(__file__).resolve().parent
WEIGHTS_BIN = HERE / "weights.bin"
GOLDENS_NPZ = HERE / "goldens.npz"


def main() -> None:
    torch.manual_seed(SEED_TORCH)
    model = ConditionalSplineFlow2D(
        cond_dim=COND_DIM, num_layers=NUM_LAYERS, num_bins=NUM_BINS, hidden=HIDDEN,
        chart=CHART_NAME,
    ).eval()
    square_to_dir = model.chart.square_to_direction   # Lambert (V1) lift, y-up

    meta = export_flow(model, str(WEIGHTS_BIN))        # tags the NFW1 with chart=V1
    assert meta["layers"] == NUM_LAYERS and meta["bins"] == NUM_BINS, meta
    assert meta["hidden"] == HIDDEN and meta["cond"] == COND_DIM, meta
    assert meta["chart"] == CHART_NAME, meta

    rng = np.random.default_rng(SEED_INPUTS)

    # ---- forward goldens: fixed (cond, u) -> (wi, solid-angle pdf) ----
    f_cond = rng.uniform(-1.0, 1.0, (N_FORWARD, COND_DIM)).astype(np.float32)
    f_u = rng.uniform(0.02, 0.98, (N_FORWARD, 2)).astype(np.float32)
    with torch.no_grad():
        ct = torch.tensor(f_cond, dtype=torch.float32)
        ut = torch.tensor(f_u, dtype=torch.float32)
        zt, ld = model.forward(ut, ct)                       # [N,2], [N,1]
        f_wi = square_to_dir(zt).numpy().astype(np.float32)
        f_pdf = torch.exp(-ld.squeeze(-1) - LOG2PI).numpy().astype(np.float64)

    # ---- inverse / round-trip goldens ----
    # Draw fresh (cond, u), push u through the FORWARD flow to get a direction
    # guaranteed to lie in the upper hemisphere, then ask the PyTorch INVERSE
    # for the pdf it assigns to that direction. The Slang/numpy round-trip
    # (forward u -> wi, then inverse wi -> pdf) must reproduce inverse_pdf, and
    # — because forward and inverse are exact inverses — it must also match the
    # forward pdf at the same point. We store both references.
    i_cond = rng.uniform(-1.0, 1.0, (N_INVERSE, COND_DIM)).astype(np.float32)
    i_u = rng.uniform(0.02, 0.98, (N_INVERSE, 2)).astype(np.float32)
    with torch.no_grad():
        ct = torch.tensor(i_cond, dtype=torch.float32)
        ut = torch.tensor(i_u, dtype=torch.float32)
        zt, _ = model.forward(ut, ct)
        i_wi = square_to_dir(zt).numpy().astype(np.float32)
        # PyTorch inverse pdf at the same direction (q_omega = q_square / 2pi).
        log_q_square = model.log_pdf_square(zt, ct).squeeze(-1)
        i_pdf = torch.exp(log_q_square - LOG2PI).numpy().astype(np.float64)

    np.savez(
        GOLDENS_NPZ,
        forward_cond=f_cond,
        forward_u=f_u,
        forward_wi=f_wi,
        forward_pdf=f_pdf,
        inverse_cond=i_cond,
        inverse_u=i_u,
        inverse_wi=i_wi,
        inverse_pdf=i_pdf,
    )

    print(f"wrote {WEIGHTS_BIN}  ({WEIGHTS_BIN.stat().st_size} bytes)  meta={meta}")
    print(f"wrote {GOLDENS_NPZ}  ({GOLDENS_NPZ.stat().st_size} bytes)")
    print(f"  forward cases={N_FORWARD}  inverse cases={N_INVERSE}")
    print(f"  forward pdf range [{f_pdf.min():.4g}, {f_pdf.max():.4g}]")
    print(f"  inverse pdf range [{i_pdf.min():.4g}, {i_pdf.max():.4g}]")


if __name__ == "__main__":
    main()
