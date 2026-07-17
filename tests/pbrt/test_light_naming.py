"""Deterministic pbrt light prim names (change pbrt-deterministic-light-names).

Regression for the `id(light)`-address suffix that made prim names — and the
baked constant-env `.hdr` filename — differ across imports of the same scene.
"""

from __future__ import annotations

from pxr import UsdLux

from skinny.pbrt.api import import_pbrt

# One constant (no-map) infinite light -> a synthesized `_const.hdr`, plus a
# distant light so we also cover the two-distinct-lights case.
SCENE = """
WorldBegin
LightSource "infinite" "rgb L" [0.4 0.6 0.9]
LightSource "distant" "rgb L" [1 1 1] "point3 from" [0 0 0] "point3 to" [0 0 -1]
"""


def _light_names(stage):
    return [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]


def test_same_scene_gives_identical_light_names(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    names_a = _light_names(import_pbrt(str(src))[0])
    names_b = _light_names(import_pbrt(str(src))[0])
    assert names_a == names_b
    assert names_a  # two lights actually authored
    # positional, not address-based
    assert set(names_a) == {"light_infinite_0", "light_distant_1"}


def test_distinct_lights_get_distinct_names(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    names = _light_names(import_pbrt(str(src))[0])
    assert len(names) == len(set(names)) == 2


def test_synthesized_hdr_filename_is_deterministic(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)

    def _tex(out):
        import_pbrt(str(src), out=str(out))
        stage = __import__("pxr").Usd.Stage.Open(str(out))
        dome = next(p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight))
        return UsdLux.DomeLight(dome).GetTextureFileAttr().Get().path

    out_a, out_b = tmp_path / "a.usda", tmp_path / "b.usda"
    tex_a = _tex(out_a)
    tex_b = _tex(out_b)
    assert tex_a == tex_b == "light_infinite_0_const.hdr"
    # the whole exported layer, not just the light name, must be byte-identical
    assert out_a.read_bytes() == out_b.read_bytes()
