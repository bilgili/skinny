"""Tests for the pbrt graphics-state machine (tasks 3.2, 3.4)."""

from __future__ import annotations

import numpy as np

from skinny.pbrt import transform as T
from skinny.pbrt.parser import parse_directives
from skinny.pbrt.state import build_scene
from skinny.pbrt.tokenizer import tokenize


def build(text):
    return build_scene(parse_directives(tokenize(text)))


def test_attribute_scope_restores_ctm():
    scene = build(
        """
        WorldBegin
        AttributeBegin
          Translate 10 0 0
          Shape "sphere"
        AttributeEnd
        Shape "sphere"
        """
    )
    assert len(scene.shapes) == 2
    inside, outside = scene.shapes
    assert np.allclose(T.transform_point(inside.ctm, [0, 0, 0]), [10, 0, 0])
    assert np.allclose(T.transform_point(outside.ctm, [0, 0, 0]), [0, 0, 0])


def test_transform_scope_only_ctm():
    scene = build(
        """
        WorldBegin
        TransformBegin
          Translate 0 5 0
          Shape "sphere"
        TransformEnd
        Shape "sphere"
        """
    )
    assert np.allclose(T.transform_point(scene.shapes[0].ctm, [0, 0, 0]), [0, 5, 0])
    assert np.allclose(T.transform_point(scene.shapes[1].ctm, [0, 0, 0]), [0, 0, 0])


def test_object_instances_share_geometry_under_distinct_transforms():
    scene = build(
        """
        WorldBegin
        ObjectBegin "ball"
          Shape "sphere" "float radius" 1
        ObjectEnd
        AttributeBegin
          Translate 3 0 0
          ObjectInstance "ball"
        AttributeEnd
        AttributeBegin
          Translate -3 0 0
          ObjectInstance "ball"
        AttributeEnd
        """
    )
    assert len(scene.shapes) == 2
    a, b = scene.shapes
    assert a.type == b.type == "sphere"
    assert np.allclose(T.transform_point(a.ctm, [0, 0, 0]), [3, 0, 0])
    assert np.allclose(T.transform_point(b.ctm, [0, 0, 0]), [-3, 0, 0])


def test_named_material_resolution():
    scene = build(
        """
        WorldBegin
        MakeNamedMaterial "red" "string type" "diffuse" "rgb reflectance" [0.8 0.1 0.1]
        NamedMaterial "red"
        Shape "sphere"
        """
    )
    mat = scene.shapes[0].material
    assert mat is not None
    assert mat.type == "diffuse"
    assert mat.params.rgb("reflectance") == [0.8, 0.1, 0.1]


def test_reverse_orientation_scoped():
    scene = build(
        """
        WorldBegin
        AttributeBegin
          ReverseOrientation
          Shape "sphere"
        AttributeEnd
        Shape "sphere"
        """
    )
    assert scene.shapes[0].reverse_orientation is True
    assert scene.shapes[1].reverse_orientation is False


def test_area_light_attaches_to_shape():
    scene = build(
        """
        WorldBegin
        AttributeBegin
          AreaLightSource "diffuse" "rgb L" [5 5 5]
          Shape "trianglemesh"
        AttributeEnd
        Shape "sphere"
        """
    )
    assert scene.shapes[0].area_light is not None
    assert scene.shapes[0].area_light.rgb("L") == [5, 5, 5]
    assert scene.shapes[1].area_light is None


def test_camera_world_to_camera_from_ctm():
    scene = build(
        """
        LookAt 0 0 5  0 0 0  0 1 0
        Camera "perspective" "float fov" 40
        WorldBegin
        """
    )
    assert scene.camera is not None
    # camera-to-world should place the camera origin back at the eye
    assert np.allclose(T.transform_point(scene.camera.camera_to_world, [0, 0, 0]), [0, 0, 5], atol=1e-9)
    assert scene.camera.params.float("fov") == 40
