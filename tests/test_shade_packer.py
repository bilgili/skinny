"""Unit tests for the wavefront shade bin-packer (pure CPU, no GPU)."""

from __future__ import annotations

import pytest

from skinny.wavefront_shade_packer import (
    MAX_SLOTS,
    SHADE_SIZE_THRESHOLD,
    pack_shade_groups,
)

SKIN = ("skin",)
DEBUG = ("debug",)


def _py(i):
    return ("python", i)


def _group_sum(group, sizes):
    return sum(sizes[m] for m in group)


def test_every_group_under_threshold():
    members = [SKIN, _py(0), _py(1), _py(2), DEBUG]
    sizes = {SKIN: 2_000_000, _py(0): 900_000, _py(1): 800_000,
             _py(2): 700_000, DEBUG: 100_000}
    groups = pack_shade_groups(members, sizes, threshold=2_000_000,
                               pin_alone={SKIN})
    for g in groups:
        # Skin alone equals the threshold; everyone else must sum <= threshold.
        assert _group_sum(g, sizes) <= 2_000_000


def test_skin_is_isolated():
    members = [SKIN, _py(0), DEBUG]
    sizes = {SKIN: 500_000, _py(0): 100_000, DEBUG: 50_000}
    groups = pack_shade_groups(members, sizes, threshold=SHADE_SIZE_THRESHOLD,
                               pin_alone={SKIN})
    skin_groups = [g for g in groups if SKIN in g]
    assert len(skin_groups) == 1
    assert skin_groups[0] == [SKIN]  # nothing packed alongside skin


def test_all_members_present_exactly_once():
    members = [SKIN, _py(0), _py(1), _py(2), DEBUG]
    sizes = {m: 300_000 for m in members}
    groups = pack_shade_groups(members, sizes, threshold=1_000_000,
                               pin_alone={SKIN})
    flat = [m for g in groups for m in g]
    assert sorted(flat, key=str) == sorted(members, key=str)
    assert len(flat) == len(set(flat))


def test_deterministic_order():
    members = [_py(2), _py(0), _py(1), SKIN]
    sizes = {SKIN: 1_000_000, _py(0): 400_000, _py(1): 400_000, _py(2): 400_000}
    a = pack_shade_groups(members, sizes, threshold=900_000, pin_alone={SKIN})
    # Same inputs in a different order → identical grouping.
    b = pack_shade_groups(list(reversed(members)), sizes, threshold=900_000,
                          pin_alone={SKIN})
    assert a == b


def test_ffd_packs_largest_first():
    members = [_py(0), _py(1), _py(2)]
    sizes = {_py(0): 600_000, _py(1): 500_000, _py(2): 400_000}
    # threshold 1.0M: 600+400 fit together, 500 alone → 2 groups.
    groups = pack_shade_groups(members, sizes, threshold=1_000_000)
    assert len(groups) == 2
    assert all(_group_sum(g, sizes) <= 1_000_000 for g in groups)


def test_empty_members_no_groups():
    assert pack_shade_groups([], {}) == []


def test_flat_only_yields_zero_nonflat_groups():
    # Flat/graph materials are slot 0 and never appear as members here.
    assert pack_shade_groups([], {}, pin_alone={SKIN}) == []


def test_max_slots_headroom_representative_scene():
    # A heavy-but-realistic scene: skin + several python materials + debug.
    members = [SKIN, DEBUG] + [_py(i) for i in range(8)]
    sizes = {SKIN: 2_000_000, DEBUG: 80_000}
    sizes.update({_py(i): 1_300_000 for i in range(8)})
    groups = pack_shade_groups(members, sizes, threshold=SHADE_SIZE_THRESHOLD,
                               pin_alone={SKIN})
    # Well under the 31 non-flat slot budget.
    assert len(groups) < MAX_SLOTS - 1


def test_overflow_raises():
    # Force more groups than MAX_SLOTS-1 by making every member exceed the
    # threshold so each lands in its own group.
    members = [_py(i) for i in range(MAX_SLOTS + 2)]
    sizes = {m: 3_000_000 for m in members}
    with pytest.raises(ValueError, match="exceeds MAX_SLOTS"):
        pack_shade_groups(members, sizes, threshold=SHADE_SIZE_THRESHOLD)


def test_missing_size_raises():
    with pytest.raises(ValueError, match="sizes missing"):
        pack_shade_groups([_py(0)], {})
