"""Tests for the pbrt directive parser (tasks 2.2, 2.3, 2.4)."""

from __future__ import annotations

import pytest

from skinny.pbrt.errors import PbrtParseError
from skinny.pbrt.parser import (
    parse_directives,
    parse_file,
    split_options_world,
)
from skinny.pbrt.tokenizer import tokenize


def parse(text):
    return parse_directives(tokenize(text))


def test_typed_params_and_scalar_value():
    (d,) = parse('Material "diffuse" "rgb reflectance" [0.2 0.3 0.4] "float roughness" 0.1')
    assert d.name == "Material"
    assert d.type_arg() == "diffuse"
    assert d.params.rgb("reflectance") == [0.2, 0.3, 0.4]
    assert d.params.float("roughness") == pytest.approx(0.1)


def test_scalar_rgb_broadcasts():
    (d,) = parse('Material "diffuse" "float reflectance" 0.5')
    assert d.params.rgb("reflectance") == [0.5, 0.5, 0.5]


def test_integer_indices_are_exact():
    (d,) = parse('Shape "trianglemesh" "integer indices" [0 1 2 2 3 0]')
    assert d.params.ints("indices") == [0, 1, 2, 2, 3, 0]


def test_transform_positional_array():
    (d,) = parse("Transform [1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1]")
    assert d.name == "Transform"
    assert isinstance(d.args[0], list)
    assert len(d.args[0]) == 16


def test_bool_and_string_params():
    (d,) = parse('Material "conductor" "bool remaproughness" "false" "string name" "gold"')
    assert d.params.bool("remaproughness") is False
    assert d.params.string("name") == "gold"


def test_lookat_positional_numbers():
    (d,) = parse("LookAt 0 0 5  0 0 0  0 1 0")
    assert d.name == "LookAt"
    assert d.args == [0, 0, 5, 0, 0, 0, 0, 1, 0]


def test_missing_value_raises():
    with pytest.raises(PbrtParseError):
        parse('Material "diffuse" "float roughness"')


def test_no_worldbegin_raises():
    with pytest.raises(PbrtParseError, match="WorldBegin"):
        split_options_world(parse("Camera \"perspective\""))


def test_split_options_world():
    ds = parse('Camera "perspective"\nWorldBegin\nShape "sphere"')
    options, world = split_options_world(ds)
    assert [d.name for d in options] == ["Camera"]
    assert [d.name for d in world] == ["Shape"]


def test_include_resolution(tmp_path):
    (tmp_path / "geom.pbrt").write_text('Shape "sphere" "float radius" 2\n')
    (tmp_path / "scene.pbrt").write_text(
        'WorldBegin\nInclude "geom.pbrt"\n'
    )
    ds = parse_file(str(tmp_path / "scene.pbrt"))
    names = [d.name for d in ds]
    assert names == ["WorldBegin", "Shape"]
    assert ds[1].params.float("radius") == 2


def test_include_relative_subdir(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "m.pbrt").write_text('Material "diffuse"\n')
    (tmp_path / "scene.pbrt").write_text('Include "sub/m.pbrt"\n')
    ds = parse_file(str(tmp_path / "scene.pbrt"))
    assert [d.name for d in ds] == ["Material"]
