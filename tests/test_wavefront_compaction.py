"""GPU test for the wavefront stream-compaction kernel (P1 §P1-E).

Between bounces, `logic` marks terminated lanes dead and the survivors must be
gathered into a dense queue for the next intersect — otherwise dead lanes waste
dispatch occupancy and grow every bounce. `compactAlive` does the gather via an
atomic-append cursor. Verified on real GPU buffers: survivors are collected,
dead lanes skipped, and the count is exact.
"""

from __future__ import annotations

import pytest

from tests.helpers import dispatch_uint_kernel

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def compaction_module(load_shader):
    return load_shader("wavefront/compaction.slang")


def test_compacts_survivors(device, compaction_module):
    alive = [1, 0, 1, 1, 0, 0, 1]  # survivors: lanes 0, 2, 3, 6
    out = dispatch_uint_kernel(
        device, compaction_module.compactAlive, [len(alive), 1, 1],
        inputs={"aliveFlag": alive},
        outputs={"outQueue": len(alive), "outCount": 1},
        scalars={"numLanes": len(alive)},
    )
    count = int(out["outCount"][0])
    assert count == 4
    assert set(out["outQueue"].tolist()[:count]) == {0, 2, 3, 6}


def test_all_dead_yields_empty(device, compaction_module):
    alive = [0, 0, 0, 0]
    out = dispatch_uint_kernel(
        device, compaction_module.compactAlive, [4, 1, 1],
        inputs={"aliveFlag": alive},
        outputs={"outQueue": 4, "outCount": 1},
        scalars={"numLanes": 4},
    )
    assert int(out["outCount"][0]) == 0


def test_all_alive_keeps_everyone(device, compaction_module):
    alive = [1, 1, 1, 1, 1]
    out = dispatch_uint_kernel(
        device, compaction_module.compactAlive, [5, 1, 1],
        inputs={"aliveFlag": alive},
        outputs={"outQueue": 5, "outCount": 1},
        scalars={"numLanes": 5},
    )
    count = int(out["outCount"][0])
    assert count == 5
    assert sorted(out["outQueue"].tolist()) == [0, 1, 2, 3, 4]
