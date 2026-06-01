"""Per-mesh slab allocator over the shared vertex / index / BVH buffers.

GPU-free bookkeeping (the renderer does the actual uploads). Replaces the
whole-scene concatenation: each baked mesh occupies a *slab* — a coupled
reservation in all three element spaces (vertices, triangles, BVH nodes) —
whose offsets are **stable for the slab's lifetime**. Add writes one slab;
remove returns it to a free-list; neither re-lays-out resident slabs, so the
per-BLAS offsets stored in TLAS instance records stay valid (design.md §9,
geometry-suballocation spec).

Slabs are keyed by an opaque caller-supplied content fingerprint: two
instances with byte-identical geometry share a slab (dedup), and an unchanged
mesh across a resync maps to its resident slab (no re-upload). The element
spaces are coupled (one slab reserves a region in each) so the three offset
triples stay consistent; free slabs are reused whole (best-fit, no split) to
keep that coupling simple — internal fragmentation is reclaimed by the opt-in
``compact()``.

Pure Python, no Vulkan import — unit-tested directly (tasks 9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class Counts(NamedTuple):
    """Element counts for a mesh: vertices, triangles, BVH nodes."""

    v: int
    t: int
    n: int


class Offsets(NamedTuple):
    """Element offsets of a slab in the shared buffers."""

    v: int
    t: int
    n: int


class AllocResult(NamedTuple):
    offsets: Offsets
    is_new: bool  # True ⇒ the slab's bytes must be uploaded at these offsets


class Move(NamedTuple):
    """A slab relocation produced by ``compact()`` (old → new offsets)."""

    key: object
    old: Offsets
    new: Offsets
    counts: Counts


@dataclass
class _Slab:
    key: object | None
    off: Offsets
    cap: Counts   # reserved region size (≥ used; only shrinks in compaction)
    used: Counts  # current contents
    alive: bool


class SlabAllocator:
    """Append + free-list allocator over three coupled element spaces.

    Invariant: a resident (alive) slab's offsets never change except during
    ``compact()``. Growing the backing GPU buffers therefore only needs to
    re-copy existing slabs at their *same* offsets — the layout is preserved by
    construction.
    """

    def __init__(self) -> None:
        self._slabs: list[_Slab] = []          # every slab, append order (alive + dead)
        self._by_key: dict[object, _Slab] = {}  # alive slabs only
        self._hw = Counts(0, 0, 0)             # high-water marks = reserved totals

    # ── queries ─────────────────────────────────────────────────────

    def is_resident(self, key) -> bool:
        return key in self._by_key

    def offsets(self, key) -> Offsets | None:
        s = self._by_key.get(key)
        return s.off if s is not None else None

    @property
    def high_water(self) -> Counts:
        """Required capacity (in elements) for each backing buffer."""
        return self._hw

    def alive_keys(self) -> list:
        return [s.key for s in self._slabs if s.alive]

    def alive_slabs(self) -> list[tuple[object, Offsets, Counts]]:
        return [(s.key, s.off, s.used) for s in self._slabs if s.alive]

    # ── allocate / free ─────────────────────────────────────────────

    def allocate(self, key, counts: Counts) -> AllocResult:
        """Reserve a slab for ``key``. Resident key → its offsets (is_new=False).
        Otherwise reuse a best-fit free slab, or append at the high-water mark."""
        counts = Counts(*counts)
        existing = self._by_key.get(key)
        if existing is not None:
            return AllocResult(existing.off, False)

        reuse = self._best_fit(counts)
        if reuse is not None:
            reuse.key = key
            reuse.used = counts
            reuse.alive = True
            self._by_key[key] = reuse
            return AllocResult(reuse.off, True)

        off = Offsets(self._hw.v, self._hw.t, self._hw.n)
        slab = _Slab(key=key, off=off, cap=counts, used=counts, alive=True)
        self._slabs.append(slab)
        self._by_key[key] = slab
        self._hw = Counts(self._hw.v + counts.v,
                          self._hw.t + counts.t,
                          self._hw.n + counts.n)
        return AllocResult(off, True)

    def free(self, key) -> bool:
        """Return ``key``'s slab to the free-list (offsets/capacity retained for
        reuse). Returns False if the key was not resident."""
        slab = self._by_key.pop(key, None)
        if slab is None:
            return False
        slab.alive = False
        slab.key = None
        return True

    def retain_only(self, keys) -> list:
        """Free every resident key not in ``keys``. Returns the freed keys.
        Convenience for a resync: caller allocates the new key set, then calls
        this to drop meshes that left the scene."""
        keep = set(keys)
        freed = [k for k in list(self._by_key) if k not in keep]
        for k in freed:
            self.free(k)
        return freed

    # ── compaction ──────────────────────────────────────────────────

    def compact(self) -> list[Move]:
        """Pack alive slabs contiguously from 0, dropping dead slabs and freed
        gaps. Mutates offsets in place and returns the relocations whose offset
        changed (the renderer copies bytes + rewrites referencing TLAS offsets).
        After this there are no dead slabs and the high-water marks equal the
        sum of live contents."""
        alive = sorted((s for s in self._slabs if s.alive),
                       key=lambda s: (s.off.v, s.off.t, s.off.n))
        moves: list[Move] = []
        v = t = n = 0
        for s in alive:
            new = Offsets(v, t, n)
            if new != s.off:
                moves.append(Move(s.key, s.off, new, s.used))
            s.off = new
            s.cap = s.used  # shrink reservation to contents
            v += s.used.v
            t += s.used.t
            n += s.used.n
        self._slabs = alive
        self._hw = Counts(v, t, n)
        return moves

    # ── internals ───────────────────────────────────────────────────

    def _best_fit(self, counts: Counts) -> _Slab | None:
        """Smallest free slab whose reserved region fits ``counts`` in all three
        spaces; None if no free slab fits. Ranked by vertex capacity (the
        dominant space), then triangle then node capacity."""
        best: _Slab | None = None
        for s in self._slabs:
            if s.alive:
                continue
            if s.cap.v >= counts.v and s.cap.t >= counts.t and s.cap.n >= counts.n:
                if best is None or (s.cap.v, s.cap.t, s.cap.n) < (best.cap.v, best.cap.t, best.cap.n):
                    best = s
        return best
