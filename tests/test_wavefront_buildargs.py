"""GPU tests for the wavefront build-args kernel (P1 §P1-C).

The bug-prone part of build-args is the ceil-division that sizes each
per-material indirect dispatch (`wfIndirectGroupCount`). Verify it on-GPU via
an inline harness importing the real kernel module, so the kernel and the test
share one definition (no formula drift).
"""

from __future__ import annotations

import pytest

from tests.helpers import dispatch_uint_kernel

pytestmark = pytest.mark.gpu

_HARNESS = """
import wavefront.build_args;

uint test_groups(uint laneCount, uint groupSize)
{
    return wfIndirectGroupCount(laneCount, groupSize);
}
"""


@pytest.fixture(scope="module")
def buildargs(load_source):
    return load_source("test_wf_buildargs", _HARNESS)


@pytest.fixture(scope="module")
def build_args_module(load_shader):
    return load_shader("wavefront/build_args.slang")


def test_zero_lanes_zero_groups(buildargs):
    assert int(buildargs.test_groups(0, 64)) == 0


def test_one_lane_one_group(buildargs):
    assert int(buildargs.test_groups(1, 64)) == 1


def test_exact_multiple_is_not_rounded_up(buildargs):
    assert int(buildargs.test_groups(64, 64)) == 1
    assert int(buildargs.test_groups(128, 64)) == 2


def test_partial_rounds_up(buildargs):
    assert int(buildargs.test_groups(65, 64)) == 2
    assert int(buildargs.test_groups(200, 64)) == 4  # ceil(200/64) = 4


# ── end-to-end kernel dispatch (full scan + indirect args) ─────────


def test_buildargs_kernel_scan_and_args(device, build_args_module):
    counts = [3, 0, 5, 1]
    out = dispatch_uint_kernel(
        device, build_args_module.buildArgs, [1, 1, 1],
        inputs={"materialCount": counts},
        outputs={"materialOffset": 4, "indirectArgs": 12},
        scalars={"numMaterials": 4, "groupSize": 64},
    )
    # Exclusive prefix sum.
    assert out["materialOffset"].tolist() == [0, 3, 3, 8]
    # One (x,y,z) per material; x = ceil(count/64), empty material → 0 groups.
    assert out["indirectArgs"].tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]


def test_buildargs_offsets_pack_tightly_for_large_counts(device, build_args_module):
    counts = [100, 200, 64]
    out = dispatch_uint_kernel(
        device, build_args_module.buildArgs, [1, 1, 1],
        inputs={"materialCount": counts},
        outputs={"materialOffset": 3, "indirectArgs": 9},
        scalars={"numMaterials": 3, "groupSize": 64},
    )
    assert out["materialOffset"].tolist() == [0, 100, 300]
    # ceil(100/64)=2, ceil(200/64)=4, ceil(64/64)=1
    assert out["indirectArgs"].tolist()[0::3] == [2, 4, 1]
