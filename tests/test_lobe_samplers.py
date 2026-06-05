"""Host-side tests for the per-lobe sampler registry (sampling/lobe_samplers.py).

Pure-Python: exercises the registry, the index→shader-id fold that feeds
FrameConstants.flatLobeSamplers, and the --lobe-samplers token parser. No GPU.
The GPU parity / variance gate lives in test_sampling_parity.py.
"""

from __future__ import annotations

import pytest

from skinny.sampling import (
    LOBE_COAT,
    LOBE_DIFFUSE,
    LOBE_SPEC,
    fold_lobe_samplers,
    lobe_sampler_modes,
    parse_lobe_samplers,
    strategies_for_lobe,
)
from skinny.sampling.lobe_samplers import (
    SAMPLER_BASIS,
    SAMPLER_NATIVE,
    SAMPLER_UNIFORM,
)


def test_native_is_index_zero_for_every_lobe():
    for lobe in (LOBE_COAT, LOBE_SPEC, LOBE_DIFFUSE):
        opts = strategies_for_lobe(lobe)
        assert opts, f"lobe {lobe} has no strategies"
        assert opts[0].shader_id == SAMPLER_NATIVE
        assert opts[0].cli_token == "native"


def test_valid_strategies_per_lobe():
    # GGX lobes: native + basis. Diffuse: native + uniform. (Only valid pairings
    # are offered — basis is not valid for diffuse, uniform not for coat/spec.)
    coat = [s.cli_token for s in strategies_for_lobe(LOBE_COAT)]
    spec = [s.cli_token for s in strategies_for_lobe(LOBE_SPEC)]
    diff = [s.cli_token for s in strategies_for_lobe(LOBE_DIFFUSE)]
    assert coat == ["native", "basis"]
    assert spec == ["native", "basis"]
    assert diff == ["native", "uniform"]


def test_modes_match_strategy_names():
    assert lobe_sampler_modes(LOBE_COAT)[0] == "Native"
    assert "basis" in [s.cli_token for s in strategies_for_lobe(LOBE_SPEC)]
    assert lobe_sampler_modes(LOBE_DIFFUSE) == ["Native", "Uniform-hemisphere"]


def test_fold_all_native_is_zero():
    # The default selection must fold to 0 — the all-native, bit-identical path.
    assert fold_lobe_samplers(0, 0, 0) == 0


def test_fold_places_shader_id_in_each_lobe_byte():
    # Index 1 selects each lobe's alternate: basis (id 1) for coat/spec, uniform
    # (id 2) for diffuse. Bytes: coat<<0 | spec<<8 | diff<<16.
    assert fold_lobe_samplers(1, 0, 0) == SAMPLER_BASIS          # coat byte
    assert fold_lobe_samplers(0, 1, 0) == (SAMPLER_BASIS << 8)   # spec byte
    assert fold_lobe_samplers(0, 0, 1) == (SAMPLER_UNIFORM << 16)  # diff byte
    combined = SAMPLER_BASIS | (SAMPLER_BASIS << 8) | (SAMPLER_UNIFORM << 16)
    assert fold_lobe_samplers(1, 1, 1) == combined


def test_fold_clamps_out_of_range_index_to_native():
    assert fold_lobe_samplers(99, -1, 5) == 0


def test_parse_empty_is_all_native():
    assert parse_lobe_samplers("") == (0, 0, 0)
    assert parse_lobe_samplers(None) == (0, 0, 0)


def test_parse_full_selection():
    assert parse_lobe_samplers("coat=basis,spec=basis,diff=uniform") == (1, 1, 1)
    # diffuse alias + whitespace + partial selection (spec stays native).
    assert parse_lobe_samplers(" coat=basis , diffuse=uniform ") == (1, 0, 1)


def test_parse_round_trips_through_fold():
    idx = parse_lobe_samplers("coat=basis,diff=uniform")
    packed = fold_lobe_samplers(*idx)
    assert packed == (SAMPLER_BASIS | (SAMPLER_UNIFORM << 16))


@pytest.mark.parametrize("bad", [
    "coat",                 # missing =
    "wing=basis",           # unknown lobe
    "coat=uniform",         # uniform invalid for a GGX lobe
    "diff=basis",           # basis invalid for the diffuse lobe
    "spec=nope",            # unknown strategy
])
def test_parse_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_lobe_samplers(bad)
