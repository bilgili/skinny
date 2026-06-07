"""Recency-weighted replay buffer for online neural-proposal training (Stage 2).

Change ``neural-online-training``. Consumes the renderer's own per-vertex path
records — the *same* ``RECORD_DTYPE`` layout the offline dump writes
(``path_records.py``) — but drained **live** each frame instead of streamed to a
``.nrec`` file. Recent paths are weighted above older ones so a trainer fed from
this buffer tracks moving geometry/lights.

Pure NumPy: runs on Mac with no CUDA. The CUDA-side drain of the GPU record
counter (bindings 36/37) lives in the renderer; this buffer is device-agnostic.
"""

from __future__ import annotations

import numpy as np

from .path_records import RECORD_DTYPE

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Fixed-capacity ring of path records with exponential recency weighting.

    Each ``add`` stamps the incoming records with a monotonically increasing
    *generation*. ``sample`` draws with probability ``exp(-decay * age)`` where
    ``age`` is generations-since-insertion, so recent records dominate — the
    mechanism that lets the net adapt after a scene change.
    """

    def __init__(self, capacity: int = 1_000_000, recency_decay: float = 0.5):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.recency_decay = float(recency_decay)
        self._records = np.empty(self.capacity, dtype=RECORD_DTYPE)
        self._gen = np.zeros(self.capacity, dtype=np.int64)
        self._write = 0          # ring write cursor
        self._size = 0           # number of valid records
        self._generation = 0     # current generation counter

    def __len__(self) -> int:
        return self._size

    def add(self, records: np.ndarray) -> None:
        """Append a batch of ``RECORD_DTYPE`` records (one frame's drain)."""
        if records.dtype != RECORD_DTYPE:
            raise TypeError(f"expected RECORD_DTYPE, got {records.dtype}")
        self._generation += 1
        n = len(records)
        if n == 0:
            return
        if n >= self.capacity:
            records = records[-self.capacity:]
            n = self.capacity
        end = self._write + n
        if end <= self.capacity:
            self._records[self._write:end] = records
            self._gen[self._write:end] = self._generation
        else:
            split = self.capacity - self._write
            self._records[self._write:] = records[:split]
            self._gen[self._write:] = self._generation
            self._records[:n - split] = records[split:]
            self._gen[:n - split] = self._generation
        self._write = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def _weights(self) -> np.ndarray:
        age = (self._generation - self._gen[:self._size]).astype(np.float64)
        return np.exp(-self.recency_decay * age)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` records, recency-weighted (with replacement)."""
        if self._size == 0:
            return np.empty(0, dtype=RECORD_DTYPE)
        rng = rng or np.random.default_rng()
        w = self._weights()
        total = w.sum()
        if total <= 0.0:
            idx = rng.integers(0, self._size, size=n)
        else:
            idx = rng.choice(self._size, size=n, p=w / total)
        return self._records[idx].copy()

    def evict_stale(self, max_age: int) -> int:
        """Forget records older than ``max_age`` generations (stale-on-motion hook).

        Stub policy for Stage 2: the NVIDIA box tunes the forgetting schedule
        against the frames-to-recover metric. Returns the count evicted.
        """
        if self._size == 0 or max_age < 0:
            return 0
        age = self._generation - self._gen[:self._size]
        keep = age <= max_age
        evicted = int((~keep).sum())
        if evicted:
            kept = self._records[:self._size][keep]
            kept_gen = self._gen[:self._size][keep]
            self._size = len(kept)
            self._records[:self._size] = kept
            self._gen[:self._size] = kept_gen
            self._write = self._size % self.capacity
        return evicted
