"""GUI-agnostic helpers shared by the Qt and Panel scene-graph editing controls.

Kept free of widget-toolkit imports so the wiring logic — which node is a valid
add-parent, which node may be deleted, and how a panel TRS triple maps to a
``set_transform`` matrix — can be unit-tested without a display. Both front-ends
call these so their behavior stays identical (GUI-consistency).
"""

from __future__ import annotations

import numpy as np

from skinny.scene_graph import (
    SceneGraphNode,
    SceneGraphProperty,
    compose_trs_matrix,
    find_node_by_path,
)

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


# ── Property dispatch ─────────────────────────────────────────────────

_LIGHT_KIND_TO_TYPE = {
    "light_dir": "dir",
    "light_sphere": "sphere",
    "light_env": "env",
}

_CAMERA_VEC3_KEYS = {
    "target": ("target_x", "target_y", "target_z"),
    "position": ("position_x", "position_y", "position_z"),
}


def find_material_ref(graph, node: SceneGraphNode):
    """Walk ancestors for the material a shader prim belongs to.

    Shader prims carry no ``renderer_ref`` of their own, so a material
    parameter edit has to resolve through the enclosing Material node.
    """
    if graph is None:
        return None
    parts = node.path.rstrip("/").split("/")
    for i in range(len(parts) - 1, 0, -1):
        parent_path = "/".join(parts[:i]) or "/"
        parent = find_node_by_path(graph, parent_path)
        if parent is not None and parent.renderer_ref is not None:
            if parent.renderer_ref.kind == "material":
                return parent.renderer_ref
    return None


def apply_scene_property(
    renderer,
    node: SceneGraphNode,
    prop: SceneGraphProperty,
    value,
    *,
    graph=None,
) -> str | None:
    """Apply one scene-graph property edit, routing to the right renderer verb.

    Shared by the Qt scene-graph dock and the MCP server so an agent edit and a
    dock edit execute the same dispatch instead of two tables that drift.

    Routing depends on the resolved property and node — not on the prim path and
    property name alone, which do not determine the verb (material parameters
    live on shader prims that carry no renderer reference, and a transform
    component write has to recompose from its siblings).

    Returns ``None`` when the edit was routed, or a short reason when it was
    not. Callers that can surface an error should; a silent no-op is what sends
    an agent into a retry loop.

    Note: the dock keeps its own call for the dome-texture *file picker* flow,
    since that owns a dialog and async error reporting. The routing decision for
    a ``texture_file`` write lives here.
    """
    if graph is None:
        graph = getattr(renderer, "scene_graph", None)

    ref = node.renderer_ref
    type_name = getattr(prop, "type_name", "")

    # Compound TRS write: recompose from the node's sibling components.
    if type_name == "vec3f":
        return _apply_vec3(renderer, node, prop, value, ref)

    if type_name == "bool":
        toggle = prop.metadata.get("toggle", "node")
        if toggle == "subtree":
            renderer.apply_subtree_enabled(node.path, bool(value))
            return None
        if ref is None:
            return f"no renderer reference for {node.path!r}"
        # Camera bools (lens_enabled, ...) are camera scalars, not enable flags.
        if ref.kind == "renderer_camera":
            renderer.apply_camera_param(prop.name, bool(value))
            return None
        renderer.apply_node_enabled(node.path, bool(value))
        return None

    if type_name == "texture_file":
        if ref is None or ref.kind != "light_env":
            return f"{prop.name!r} on {node.path!r} is not a dome-light texture"
        # Returns False for a missing or unreadable HDR -- report it rather than
        # recording a texture that never loaded.
        if renderer.apply_dome_light_texture(ref.index, value) is False:
            return f"could not load environment texture {value!r}"
        return None

    if type_name == "lens_file":
        # A lens file is not a camera scalar; apply_camera_param would try
        # float(path) on it.
        if not hasattr(renderer, "apply_camera_lens_file"):
            return "this renderer cannot load lens files"
        if renderer.apply_camera_lens_file(value) is False:
            return f"could not load lens file {value!r}"
        return None

    if ref is None:
        ref = find_material_ref(graph, node)
        if ref is None:
            return f"no renderer reference for {node.path!r}"

    if ref.kind == "material":
        # A promoted logical input (synthesized MaterialX material) fans one
        # edit out to every generated uniform it drives (mcp-material-authoring,
        # design D5); `metadata['fanout']` carries those uniform names. Plain
        # material inputs write the single named override.
        fanout = prop.metadata.get("fanout") if prop.metadata else None
        if fanout:
            renderer.apply_material_overrides(
                ref.index, {uniform: value for uniform in fanout},
            )
        else:
            renderer.apply_material_override(ref.index, prop.name, value)
        return None
    if ref.kind in _LIGHT_KIND_TO_TYPE:
        renderer.apply_light_override(
            _LIGHT_KIND_TO_TYPE[ref.kind], ref.index, prop.name, value,
        )
        return None
    if ref.kind == "renderer_camera":
        renderer.apply_camera_param(prop.name, value)
        return None
    return f"no route for {prop.name!r} on a {ref.kind!r} node"


def _apply_vec3(renderer, node, prop, values, ref) -> str | None:
    if ref is not None and ref.kind == "renderer_camera":
        keys = _CAMERA_VEC3_KEYS.get(prop.metadata.get("camera_axis", ""))
        if keys is None:
            return f"{prop.name!r} is not a settable camera vector"
        for key, component in zip(keys, values):
            renderer.apply_camera_param(key, float(component))
        return None

    is_authored_light = node.type_name in SUPPORTED_LIGHT_TYPES
    if ref is None and not is_authored_light:
        return f"no renderer reference for {node.path!r}"
    if ref is not None and ref.kind != "instance" and not is_authored_light:
        return f"no transform route for a {ref.kind!r} node"

    # TRS needs all three vectors; take the written one and read its siblings
    # off the node so the untouched components survive.
    translate = scale = (0.0, 0.0, 0.0)
    rotate = (0.0, 0.0, 0.0)
    for p in node.properties:
        if p.name == "translate":
            translate = values if p is prop else p.value
        elif p.name == "rotate":
            rotate = values if p is prop else p.value
        elif p.name == "scale":
            scale = values if p is prop else p.value

    # Author to the stage (edit layer) so the move persists and is captured by
    # "Save edits"; fall back to the runtime path when no stage is loaded.
    if getattr(renderer, "_usd_stage", None) is not None:
        renderer.set_transform(node.path, trs_to_matrix(translate, rotate, scale))
    else:
        renderer.apply_instance_transform(node.path, translate, rotate, scale)
    return None
