"""Host-side MLT bootstrap resample (change mlt-integrator, design D3).

Pure numpy — hostless-testable. After the GPU bootstrap phase writes one
scalar contribution per bootstrap sample, the host computes the
normalization constant ``b`` and seeds every chain proportionally to
bootstrap weight (pbrt ``MLTIntegrator::Render``'s ``SampleDiscrete`` over
the bootstrap weights — the textbook startup-bias elimination).
"""

from __future__ import annotations

import numpy as np


def resample_chain_seeds(weights: np.ndarray, num_chains: int,
                         seed: int) -> tuple[float, np.ndarray]:
    """Resample ``num_chains`` bootstrap indices proportional to ``weights``.

    Returns ``(b, seeds)`` where ``b = weights.mean()`` (the resolve scale,
    pbrt's ``b = sum(bootstrapWeights) / nBootstrap``) and ``seeds`` is a
    uint32 array of bootstrap indices drawn via the weight CDF with a seeded
    generator (deterministic per ``seed``). Non-finite / negative weights are
    zeroed defensively before the draw; all-zero weights raise loudly instead
    of seeding chains from a degenerate distribution.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    total = float(w.sum())
    if not (total > 0.0):
        raise RuntimeError(
            "no light-carrying paths found during MLT bootstrap — every "
            "bootstrap sample had zero (or non-finite) contribution; the "
            "scene is black from the camera (no lights / fully occluded?)")
    b = total / w.size
    cdf = np.cumsum(w)
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    u = rng.random(int(num_chains)) * cdf[-1]
    seeds = np.searchsorted(cdf, u, side="right").astype(np.uint32)
    return float(b), seeds
