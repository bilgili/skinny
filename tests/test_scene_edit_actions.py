"""Unit tests for the GUI-agnostic scene-graph editing helpers.

Pure functions — no Vulkan, no display, no USD stage required."""

from __future__ import annotations

import numpy as np

from skinny.ui.scene_edit_actions import (
    SUPPORTED_LIGHT_TYPES,
    add_parent_for_node,
    has_editable_stage,
    is_deletable,
    trs_to_matrix,
)
from skinny.scene_graph import RendererRef, SceneGraphNode


class TestHasEditableStage:
    """Finding #4: the editable gate also requires adopted scene metadata, so the
    async-loader window where the stage/edit-layer are published before the scene
    is not treated as editable."""

    class _R:
        pass

    def _r(self, *, stage, edit_layer, scene):
        r = self._R()
        r._usd_stage = stage
        r._usd_edit_layer = edit_layer
        r._usd_scene = scene
        return r

    def test_stage_and_layer_without_scene_is_not_editable(self):
        # The loader window: stage + edit layer published, scene not yet adopted.
        assert has_editable_stage(self._r(stage=object(), edit_layer=object(), scene=None)) is False

    def test_all_three_present_is_editable(self):
        assert has_editable_stage(self._r(stage=object(), edit_layer=object(), scene=object())) is True

    def test_missing_stage_or_layer_is_not_editable(self):
        assert has_editable_stage(self._r(stage=None, edit_layer=object(), scene=object())) is False
        assert has_editable_stage(self._r(stage=object(), edit_layer=None, scene=object())) is False


def _node(path: str, type_name: str, ref: RendererRef | None = None) -> SceneGraphNode:
    return SceneGraphNode(path=path, name=path.rsplit("/", 1)[-1],
                          type_name=type_name, renderer_ref=ref)


class TestAddParentForNode:
    def test_group_node_returns_own_path(self):
        assert add_parent_for_node(_node("/World/Group", "Xform")) == "/World/Group"
        assert add_parent_for_node(_node("/World/Scope", "Scope")) == "/World/Scope"

    def test_non_group_node_falls_back_to_world(self):
        mesh = _node("/World/Mesh", "Mesh", RendererRef("instance", 0))
        assert add_parent_for_node(mesh) == "/World"

    def test_no_selection_falls_back_to_world(self):
        assert add_parent_for_node(None) == "/World"


def test_supported_light_types_match_renderer_usd_schemas():
    assert SUPPORTED_LIGHT_TYPES == (
        "DistantLight",
        "SphereLight",
        "DomeLight",
        "RectLight",
        "DiskLight",
    )


class TestIsDeletable:
    def test_regular_nodes_deletable(self):
        assert is_deletable(_node("/World/Mesh", "Mesh", RendererRef("instance", 0)))
        assert is_deletable(_node("/World/Sun", "DistantLight", RendererRef("light_dir", 0)))
        assert is_deletable(_node("/World/Cam", "Camera", RendererRef("camera", 0)))

    def test_synthesized_nodes_not_deletable(self):
        assert not is_deletable(_node("/Skinny/DefaultLight", "DistantLight"))
        assert not is_deletable(_node("/Skinny/DefaultDome", "DomeLight"))
        assert not is_deletable(_node("/Skinny/MainCamera", "Camera"))
        assert not is_deletable(_node("/Skinny", "Scope"))

    def test_root_and_none_not_deletable(self):
        assert not is_deletable(_node("/", "Xform"))
        assert not is_deletable(None)


class TestTrsToMatrix:
    def test_translation_lands_in_row_three(self):
        m = trs_to_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert m.shape == (4, 4)
        np.testing.assert_allclose(m[3, :3], [1.0, 2.0, 3.0], atol=1e-6)

    def test_identity_trs_is_identity(self):
        m = trs_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        np.testing.assert_allclose(m, np.eye(4), atol=1e-6)

    def test_scale_on_diagonal(self):
        m = trs_to_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 3.0, 4.0))
        np.testing.assert_allclose([m[0, 0], m[1, 1], m[2, 2]], [2.0, 3.0, 4.0], atol=1e-6)
