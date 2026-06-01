"""Unit tests for the GPU-free SlabAllocator (geometry-suballocation, task 9.4).

Covers the spec invariants: offsets stable across unrelated add/remove,
free-list reuse, high-water growth preserves existing offsets, and compaction
packs alive slabs into a valid contiguous layout.
"""

from __future__ import annotations

from skinny.slab_allocator import Counts, Offsets, SlabAllocator


def _alloc(a, key, v, t, n):
    return a.allocate(key, Counts(v, t, n))


def test_append_assigns_contiguous_offsets():
    a = SlabAllocator()
    r1 = _alloc(a, "m1", 10, 4, 3)
    r2 = _alloc(a, "m2", 20, 8, 5)
    assert r1.offsets == Offsets(0, 0, 0) and r1.is_new
    assert r2.offsets == Offsets(10, 4, 3) and r2.is_new
    assert a.high_water == Counts(30, 12, 8)


def test_resident_key_returns_same_offsets_not_new():
    a = SlabAllocator()
    first = _alloc(a, "m1", 10, 4, 3)
    again = _alloc(a, "m1", 10, 4, 3)
    assert again.offsets == first.offsets
    assert again.is_new is False  # already resident ⇒ no re-upload


def test_offsets_stable_across_unrelated_add_remove():
    # The spec scenario: adding/removing a mesh must not move any other mesh.
    a = SlabAllocator()
    _alloc(a, "A", 10, 4, 3)
    off_b = _alloc(a, "B", 20, 8, 5).offsets
    off_c = _alloc(a, "C", 5, 2, 1).offsets
    # Remove A (unrelated to B/C) and add a new mesh D.
    a.free("A")
    _alloc(a, "D", 100, 40, 30)
    # B and C are untouched — their offsets and residency are unchanged.
    assert a.offsets("B") == off_b
    assert a.offsets("C") == off_c


def test_free_list_reuse_best_fit():
    a = SlabAllocator()
    _alloc(a, "A", 100, 40, 30)   # off 0
    _alloc(a, "B", 10, 4, 3)      # off 100/40/30
    hw_before = a.high_water
    a.free("A")                    # 100-vertex hole at offset 0
    # A new mesh that fits reuses the freed slab in place — no high-water growth.
    r = _alloc(a, "C", 50, 20, 15)
    assert r.is_new
    assert r.offsets == Offsets(0, 0, 0)         # reused A's region
    assert a.high_water == hw_before              # nothing appended


def test_free_list_skips_too_small_slabs():
    a = SlabAllocator()
    _alloc(a, "small", 10, 4, 3)
    a.free("small")
    # Bigger than the only free slab ⇒ must append, not reuse.
    r = _alloc(a, "big", 50, 20, 15)
    assert r.offsets == Offsets(10, 4, 3)  # appended past the (retained) small slab


def test_best_fit_picks_smallest_sufficient():
    a = SlabAllocator()
    _alloc(a, "huge", 1000, 400, 300)
    _alloc(a, "mid", 100, 40, 30)
    a.free("huge")
    a.free("mid")
    # A 50-vertex mesh fits both holes; best-fit takes the smaller (mid).
    r = _alloc(a, "x", 50, 20, 15)
    assert r.offsets == a.offsets("x")
    assert r.offsets == Offsets(1000, 400, 300)  # mid's region (appended after huge)


def test_growth_preserves_existing_offsets():
    # High-water only grows; resident slab offsets never shift (so a GPU buffer
    # grow can re-copy at the same offsets — "grow preserves layout").
    a = SlabAllocator()
    offs = {k: _alloc(a, k, 10 * i, 4 * i, 3 * i).offsets
            for i, k in enumerate(["a", "b", "c", "d"], start=1)}
    for k, off in offs.items():
        assert a.offsets(k) == off


def test_retain_only_frees_absent_keys():
    a = SlabAllocator()
    for k in ["a", "b", "c"]:
        _alloc(a, k, 10, 4, 3)
    freed = a.retain_only(["a", "c"])
    assert set(freed) == {"b"}
    assert a.is_resident("a") and a.is_resident("c")
    assert not a.is_resident("b")


def test_compaction_packs_and_reports_moves():
    a = SlabAllocator()
    _alloc(a, "A", 10, 4, 3)   # 0
    _alloc(a, "B", 20, 8, 5)   # 10/4/3
    _alloc(a, "C", 5, 2, 1)    # 30/12/8
    a.free("B")                 # hole in the middle
    moves = a.compact()
    # C slides down into B's vacated space; A stays put.
    assert a.offsets("A") == Offsets(0, 0, 0)
    assert a.offsets("C") == Offsets(10, 4, 3)
    assert a.high_water == Counts(15, 6, 4)  # A(10,4,3) + C(5,2,1), no gaps
    moved_keys = {m.key for m in moves}
    assert moved_keys == {"C"}               # only C actually relocated
    cmove = next(m for m in moves if m.key == "C")
    assert cmove.old == Offsets(30, 12, 8) and cmove.new == Offsets(10, 4, 3)


def test_compaction_is_idempotent_when_no_gaps():
    a = SlabAllocator()
    _alloc(a, "A", 10, 4, 3)
    _alloc(a, "B", 20, 8, 5)
    assert a.compact() == []   # already packed ⇒ no moves
    assert a.high_water == Counts(30, 12, 8)


def test_reuse_after_compaction():
    a = SlabAllocator()
    _alloc(a, "A", 10, 4, 3)
    _alloc(a, "B", 20, 8, 5)
    a.free("A")
    a.compact()  # drops A's dead slab, B slides to 0
    assert a.offsets("B") == Offsets(0, 0, 0)
    # After compaction there are no free slabs ⇒ a new mesh appends.
    r = _alloc(a, "C", 7, 3, 2)
    assert r.offsets == Offsets(20, 8, 5)
