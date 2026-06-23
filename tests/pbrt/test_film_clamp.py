"""Film per-sample radiance clamp (pbrt `maxcomponentvalue`).

Change: film-maxcomponent-clamp. These are hostless (USD + pbrt parser only) —
the GPU effect of the clamp is gated by the bathroom parity matrix.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr")

from skinny.pbrt.api import import_pbrt  # noqa: E402
from skinny.usd_loader import _film_max_component  # noqa: E402

_CAM_WORLD = (
    'Camera "perspective" "float fov" 50\nWorldBegin\n'
    'Shape "sphere" "float radius" 1\n'
)


def _import_stage(tmp_path, film_line: str):
    from pxr import Usd

    src = tmp_path / "s.pbrt"
    src.write_text(f'{film_line}{_CAM_WORLD}')
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out))
    return Usd.Stage.Open(str(out))


def test_film_maxcomponentvalue_survives_import(tmp_path):
    """The importer already serialises the whole film verbatim, so the threshold
    is present in customLayerData without any emit change."""
    stage = _import_stage(
        tmp_path,
        'Film "rgb" "float maxcomponentvalue" 50 "float iso" 600 '
        '"integer xresolution" 64 "integer yresolution" 64\n',
    )
    params = stage.GetRootLayer().customLayerData["pbrt"]["film"]["params"]
    assert float(params["maxcomponentvalue"]) == pytest.approx(50.0)


def test_loader_reads_threshold(tmp_path):
    stage = _import_stage(
        tmp_path,
        'Film "rgb" "float maxcomponentvalue" 50 '
        '"integer xresolution" 64 "integer yresolution" 64\n',
    )
    assert _film_max_component(stage) == pytest.approx(50.0)


def test_loader_zero_when_film_omits_it(tmp_path):
    stage = _import_stage(
        tmp_path,
        'Film "rgb" "float iso" 600 "integer xresolution" 64 '
        '"integer yresolution" 64\n',
    )
    assert _film_max_component(stage) == 0.0


def test_loader_zero_for_nonpositive(tmp_path):
    stage = _import_stage(
        tmp_path,
        'Film "rgb" "float maxcomponentvalue" -3 "integer xresolution" 64 '
        '"integer yresolution" 64\n',
    )
    assert _film_max_component(stage) == 0.0


def test_loader_zero_on_missing_metadata(tmp_path):
    """A bare stage with no pbrt metadata must not raise — clamp stays off."""
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(tmp_path / "bare.usda"))
    assert _film_max_component(stage) == 0.0


def test_scene_carries_threshold_through_usd_loader(tmp_path):
    """End to end: import a pbrt film with the threshold, load it, and confirm the
    Scene dataclass exposes it for the renderer to push into FrameConstants."""
    from pathlib import Path

    from skinny.usd_loader import load_scene_from_usd

    src = tmp_path / "s.pbrt"
    src.write_text(
        'Film "rgb" "float maxcomponentvalue" 42 "integer xresolution" 64 '
        f'"integer yresolution" 64\n{_CAM_WORLD}'
    )
    out = tmp_path / "s.usda"
    import_pbrt(str(src), out=str(out))
    scene = load_scene_from_usd(Path(out))
    assert scene.film_max_component == pytest.approx(42.0)
