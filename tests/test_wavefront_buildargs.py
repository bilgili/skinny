"""GPU tests for the wavefront build-args kernel (P1 §P1-C).

The bug-prone part of build-args is the ceil-division that sizes each
per-material indirect dispatch (`wfIndirectGroupCount`). Verify it on-GPU via
an inline harness importing the real kernel module, so the kernel and the test
share one definition (no formula drift).
"""

from __future__ import annotations

import pytest

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
