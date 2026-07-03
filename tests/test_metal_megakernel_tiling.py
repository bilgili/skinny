"""Metal megakernel watchdog tiling (change metal-megakernel-watchdog-tiling).

Hostless logic tests for the row-band tiling that bounds each committed Metal
megakernel command buffer under the macOS GPU watchdog. No Metal device is
constructed here — these exercise the field-table offset and the band-count
policy as plain Python. (They still import ``skinny.renderer``, which imports
``vulkan`` at module load, so run under the SDK env like the other non-gpu
renderer tests; they are NOT gpu-marked and touch no GPU.)

The heavy-scene wedge repro itself (BDPT megakernel on the regenerated
graph-material ``bathroom.usda``) is a manual headless probe — that 41 MB scene
is not a committable fixture — while the parity matrix sweeps BDPT megakernel on
Metal across the corpus and the kill harness (``test_metal_cleanup.py -m gpu``)
proves dispatch changes leave the GPU usable.
"""

from __future__ import annotations

from types import SimpleNamespace

from skinny import renderer as R


def test_tile_origin_y_is_last_field_and_offset_matches():
    fields = R._FC_SCALAR_FIELDS
    assert fields[-1] == ("tileOriginY", 4), "tileOriginY must be the fc scalar tail"
    total = sum(sz for _, sz in fields)
    assert R._TILE_ORIGIN_Y_OFFSET == total - 4
    # The Vulkan UBO must still hold the (now 4 B longer) scalar blob.
    assert total <= R._VK_UNIFORM_BUFFER_BYTES


def _bands(integrator, w, h, env=None):
    stub = SimpleNamespace(integrator_index=integrator, width=w, height=h)
    if env is not None:
        import os
        prev = os.environ.get("SKINNY_METAL_MEGAKERNEL_BANDS")
        os.environ["SKINNY_METAL_MEGAKERNEL_BANDS"] = str(env)
        try:
            return R.Renderer._metal_megakernel_bands(stub)
        finally:
            if prev is None:
                del os.environ["SKINNY_METAL_MEGAKERNEL_BANDS"]
            else:
                os.environ["SKINNY_METAL_MEGAKERNEL_BANDS"] = prev
    return R.Renderer._metal_megakernel_bands(stub)


def test_bdpt_gets_more_bands_than_path_at_same_resolution():
    # BDPT (integrator 1) does the widest per-pixel work and must split into more,
    # smaller bands than the path tracer (integrator 0) at 720p — that split is
    # what keeps each command buffer under the watchdog on graph-material scenes.
    path_bands = _bands(0, 1280, 720)
    bdpt_bands = _bands(1, 1280, 720)
    assert bdpt_bands > path_bands
    assert bdpt_bands >= 2  # 1280x720 over the BDPT budget → multiple bands


def test_band_count_scales_with_resolution_for_bdpt():
    assert _bands(1, 1280, 720) < _bands(1, 2560, 1440)


def test_band_count_is_at_least_one_and_at_most_height():
    assert _bands(0, 64, 64) == 1          # cheap path frame → single dispatch
    assert _bands(1, 1, 1) == 1            # never zero bands
    assert _bands(1, 1, 4) <= 4           # never more bands than rows


def test_env_override_wins_and_is_clamped_to_one():
    assert _bands(1, 1280, 720, env=3) == 3
    assert _bands(0, 1280, 720, env=1) == 1
    assert _bands(1, 1280, 720, env=0) >= 1     # 0/garbage falls back to >= 1
    assert _bands(1, 1280, 720, env="oops") >= 1
