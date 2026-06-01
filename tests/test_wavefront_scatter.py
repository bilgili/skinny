"""GPU test for the wavefront counting-sort scatter kernel (P1 §P1-B).

After intersect counts hits per material and build_args prefix-sums those counts
into per-material base offsets, the scatter places each lane's index into its
material's contiguous slice of `materialQueue` via an atomic write cursor. This
verifies the placement on real GPU buffers: each material's slice ends up
holding exactly its lanes (order within a slice is unspecified — atomics race).
"""

from __future__ import annotations

import pytest

from tests.helpers import dispatch_uint_kernel

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def scatter_module(load_shader):
    return load_shader("wavefront/scatter.slang")


def test_scatter_groups_lanes_by_material(device, scatter_module):
    # 6 lanes, 3 materials. lane→material:
    lane_material = [0, 2, 0, 1, 2, 2]
    # counts: m0=2, m1=1, m2=3 → exclusive prefix-sum offsets:
    material_offset = [0, 2, 3]
    out = dispatch_uint_kernel(
        device, scatter_module.scatterByMaterial, [len(lane_material), 1, 1],
        inputs={"laneMaterial": lane_material, "materialOffset": material_offset},
        outputs={"writeCursor": 3, "materialQueue": 6},
        scalars={"numLanes": len(lane_material)},
    )
    q = out["materialQueue"].tolist()
    # Each material's contiguous slice holds exactly its lanes (order unspecified).
    assert set(q[0:2]) == {0, 2}   # material 0
    assert set(q[2:3]) == {3}      # material 1
    assert set(q[3:6]) == {1, 4, 5}  # material 2
    # Write cursor ends at each material's count.
    assert out["writeCursor"].tolist() == [2, 1, 3]


def test_scatter_single_material(device, scatter_module):
    lane_material = [0, 0, 0, 0]
    out = dispatch_uint_kernel(
        device, scatter_module.scatterByMaterial, [4, 1, 1],
        inputs={"laneMaterial": lane_material, "materialOffset": [0]},
        outputs={"writeCursor": 1, "materialQueue": 4},
        scalars={"numLanes": 4},
    )
    assert sorted(out["materialQueue"].tolist()) == [0, 1, 2, 3]
    assert out["writeCursor"].tolist() == [4]


def test_scatter_empty_material_leaves_gap_free_slices(device, scatter_module):
    # material 1 has no lanes; its slice is empty and must not be written.
    lane_material = [0, 2, 2]
    material_offset = [0, 1, 1]  # m0=1, m1=0, m2=2
    out = dispatch_uint_kernel(
        device, scatter_module.scatterByMaterial, [3, 1, 1],
        inputs={"laneMaterial": lane_material, "materialOffset": material_offset},
        outputs={"writeCursor": 3, "materialQueue": 3},
        scalars={"numLanes": 3},
    )
    q = out["materialQueue"].tolist()
    assert q[0] == 0           # material 0 slice [0,1)
    assert set(q[1:3]) == {1, 2}  # material 2 slice [1,3)
    assert out["writeCursor"].tolist() == [1, 0, 2]
