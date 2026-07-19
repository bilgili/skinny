"""numpy mirror of the PSSMLT primary-sample-space sampler (change mlt-integrator).

Mirrors ``shaders/integrators/mlt_sampler.slang`` operation-for-operation —
same PCG-32 stream, same fp32 arithmetic, same lazy Kelemen mutation
bookkeeping — so the sampler semantics are testable hostlessly and a GPU
harness can later assert bit-agreement. Keep the two in lockstep.

Reference: pbrt-v4 ``MLTSampler`` (src/pbrt/samplers.{h,cpp}).
"""

from __future__ import annotations

import numpy as np

MLT_NUM_STREAMS = 3
# Must match common.slang MLT_MAX_DIMS. Worst-case usage ~72 at maxDepth 5
# (design D2); spectral MLT (change spectral-mlt) adds exactly one camera-stream
# dimension (the hero-wavelength draw) → ~73, still far under this budget.
MLT_MAX_DIMS = 192

_F32 = np.float32
_U32_MAX = _F32(np.float32(4294967295.0))


def pcg_hash(x: int) -> int:
    """common.slang ``pcgHash`` (PCG-32 XSH-RR) on uint32 wraparound."""
    state = np.uint32(np.uint64(x) * np.uint64(747796405) + np.uint64(2891336453))
    shift = np.uint32((state >> np.uint32(28)) + np.uint32(4))
    word = np.uint32(
        np.uint64((state >> shift) ^ state) * np.uint64(277803737))
    return int((word >> np.uint32(22)) ^ word)


class Rng:
    """common.slang ``RNG``: state = pcgHash(state); value = state / 2^32-1."""

    def __init__(self, state: int):
        self.state = int(state) & 0xFFFFFFFF

    def next(self) -> np.float32:
        self.state = pcg_hash(self.state)
        return _F32(_F32(self.state) / _U32_MAX)


def erf_inv(a: float) -> np.float32:
    """pbrt-v4 ``ErfInv`` polynomial (fp32, fused as plain mul-add)."""
    a = _F32(a)
    t = _F32(np.log(max(_F32(1.0) - a * a, _F32(1.175494e-38))))
    if abs(t) > 6.125:
        cs = [3.03697567e-10, 2.93243101e-8, 1.22150334e-6, 2.84108955e-5,
              3.93552968e-4, 3.02698812e-3, 4.83185798e-3, -2.64646143e-1,
              8.40016484e-1]
    else:
        cs = [5.43877832e-9, 1.43286059e-7, 1.22775396e-6, 1.12962631e-7,
              -5.61531961e-5, -1.47697705e-4, 2.31468701e-3, 1.15392562e-2,
              -2.32015476e-1, 8.86226892e-1]
    p = _F32(cs[0])
    for c in cs[1:]:
        p = _F32(p * t + _F32(c))
    return _F32(p * a)


def sample_normal(u: float, mu: float, sigma: float) -> np.float32:
    v = _F32(np.clip(_F32(2.0) * _F32(u) - _F32(1.0), -0.99999994, 0.99999994))
    return _F32(_F32(mu) + _F32(1.41421356237) * _F32(sigma) * erf_inv(v))


class MltSampler:
    """pbrt MLTSampler over a fixed-size X vector (the GPU port's semantics).

    Initial state matches a freshly-constructed pbrt sampler: iteration 0,
    largeStep True, lastLargeStepIteration 0 — the initial L evaluation runs
    with no preceding start_iteration (design D3).
    """

    def __init__(self, seed_index: int, seed: int, sigma: float,
                 large_step_probability: float):
        self.rng = Rng(pcg_hash((seed_index + seed * 16777259) & 0xFFFFFFFF))
        self.sigma = _F32(sigma)
        self.large_step_probability = _F32(large_step_probability)
        # X columns: value, valueBackup, lastMod, modBackup
        self.value = np.zeros(MLT_MAX_DIMS, dtype=np.float32)
        self.value_backup = np.zeros(MLT_MAX_DIMS, dtype=np.float32)
        self.last_mod = np.zeros(MLT_MAX_DIMS, dtype=np.uint32)
        self.mod_backup = np.zeros(MLT_MAX_DIMS, dtype=np.uint32)
        self.current_iteration = 0
        self.last_large_step_iteration = 0
        self.large_step = True
        self.stream_index = 0
        self.sample_index = 0

    def start_iteration(self) -> None:
        self.current_iteration += 1
        self.large_step = bool(self.rng.next() < self.large_step_probability)

    def start_stream(self, index: int) -> None:
        assert index < MLT_NUM_STREAMS
        self.stream_index = index
        self.sample_index = 0

    def _next_index(self) -> int:
        idx = self.stream_index + MLT_NUM_STREAMS * self.sample_index
        self.sample_index += 1
        return idx

    def _ensure_ready(self, index: int) -> np.float32:
        assert index < MLT_MAX_DIMS, "primary-sample budget exceeded (design D2 invariant)"
        if self.last_mod[index] < self.last_large_step_iteration:
            self.value[index] = self.rng.next()
            self.last_mod[index] = self.last_large_step_iteration
        self.value_backup[index] = self.value[index]
        self.mod_backup[index] = self.last_mod[index]
        if self.large_step:
            self.value[index] = self.rng.next()
        else:
            n_small = self.current_iteration - int(self.last_mod[index])
            eff_sigma = _F32(self.sigma * _F32(np.sqrt(_F32(n_small))))
            v = _F32(self.value[index] + sample_normal(self.rng.next(), 0.0, eff_sigma))
            self.value[index] = _F32(v - np.floor(v))
        self.last_mod[index] = self.current_iteration
        return self.value[index]

    def get_1d(self) -> np.float32:
        return self._ensure_ready(self._next_index())

    def get_2d(self) -> tuple[np.float32, np.float32]:
        return self.get_1d(), self.get_1d()

    def accept(self) -> None:
        if self.large_step:
            self.last_large_step_iteration = self.current_iteration

    def reject(self) -> None:
        touched = self.last_mod == np.uint32(self.current_iteration)
        self.value[touched] = self.value_backup[touched]
        self.last_mod[touched] = self.mod_backup[touched]
        self.current_iteration -= 1
