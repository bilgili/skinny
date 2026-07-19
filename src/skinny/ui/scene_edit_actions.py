"""GUI-agnostic helpers shared by the Qt and Panel scene-graph editing controls.

Kept free of widget-toolkit imports so the wiring logic — which node is a valid
add-parent, which node may be deleted, and how a panel TRS triple maps to a
``set_transform`` matrix — can be unit-tested without a display. Both front-ends
call these so their behavior stays identical (GUI-consistency).
"""

from __future__ import annotations

import numpy as np

from skinny.scene_graph import SceneGraphNode, compose_trs_matrix

# Prim paths the renderer synthesizes (default light / dome / camera); not
# user-deletable and never a target for "Add model".
_SYNTH_PREFIX = "/Skinny"
# USD prim types that can hold children, so "Add model" nests a new prim under
# them. Anything else falls back to /World.
_GROUP_TYPES = frozenset({"Xform", "Scope"})

# Canonical UsdLux schema names accepted by ``Renderer.add_light``. Kept here
# so the Qt and Panel scene-graph controls expose the same ordered menu.
SUPPORTED_LIGHT_TYPES = (
    "DistantLight",
    "SphereLight",
    "DomeLight",
    "RectLight",
    "DiskLight",
)


def has_editable_stage(renderer) -> bool:
    """Whether scene-edit operations can author into an active USD edit layer."""
    return (
        getattr(renderer, "_usd_stage", None) is not None
        and getattr(renderer, "_usd_edit_layer", None) is not None
    )


def add_parent_for_node(node: "SceneGraphNode | None") -> str:
    """Parent prim path for an add action relative to ``node``.

    Returns the node's own path when it is a group-like prim (Xform/Scope) so
    the new model or light nests under the selection; otherwise ``/World``.
    """
    if node is not None and node.type_name in _GROUP_TYPES and node.path:
        return node.path
    return "/World"


def is_deletable(node: "SceneGraphNode | None") -> bool:
    """Whether ``node`` may be removed via ``remove_node``.

    False for no selection, the pseudo-root, and synthesized ``/Skinny/*`` nodes
    (the renderer-owned default light/dome/camera).
    """
    if node is None:
        return False
    path = (node.path or "").rstrip("/")
    if path in ("", "/"):
        return False
    if path == _SYNTH_PREFIX or path.startswith(_SYNTH_PREFIX + "/"):
        return False
    return True


def trs_to_matrix(translate, rotate_deg, scale) -> np.ndarray:
    """Compose a panel TRS triple into the 4x4 matrix ``set_transform`` expects."""
    return compose_trs_matrix(translate, rotate_deg, scale)
