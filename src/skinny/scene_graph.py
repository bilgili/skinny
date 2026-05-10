"""USD scene graph tree model for the property editor UI.

Preserves the USD prim hierarchy as a browsable tree with typed,
optionally-editable properties on each node. Built from an open
``Usd.Stage`` + the flat ``Scene`` the renderer consumes; maps each
node back to a renderer-side object (material, light, instance) via
``RendererRef`` so property edits can flow through the existing
``apply_material_override`` / ``apply_light_override`` /
``apply_instance_transform`` methods.

Not a replacement for ``Scene`` — this is a UI overlay that
references into the flat lists the GPU pipeline reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─── Data model ─────────────────────────────────────────────────────


@dataclass
class SceneGraphProperty:
    name: str
    display_name: str
    type_name: str  # float, color3f, int, bool, token, asset, vec3f, rel, string
    value: object
    editable: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class RendererRef:
    kind: str   # material, light_dir, light_sphere, instance, camera
    index: int


@dataclass
class SceneGraphNode:
    path: str
    name: str
    type_name: str  # Xform, Mesh, Material, Shader, DistantLight, SphereLight, DomeLight, RectLight, DiskLight, Camera, ...
    children: list[SceneGraphNode] = field(default_factory=list)
    properties: list[SceneGraphProperty] = field(default_factory=list)
    renderer_ref: Optional[RendererRef] = None


# ─── Type icons for display ─────────────────────────────────────────

_TYPE_ICONS: dict[str, str] = {
    "Xform": "⊕",          # ⊕
    "Scope": "⊕",
    "Mesh": "△",           # △
    "Material": "◆",       # ◆
    "Shader": "◇",         # ◇
    "DistantLight": "☀",   # ☀
    "SphereLight": "☀",
    "DomeLight": "☀",
    "RectLight": "☀",
    "DiskLight": "☀",
    "Camera": "▣",         # ▣
}


def type_icon(type_name: str) -> str:
    return _TYPE_ICONS.get(type_name, "•")  # bullet fallback


# ─── Known editable material properties with ranges ─────────────────

_MATERIAL_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "roughness":      (0.04, 1.0),
    "metallic":       (0.0,  1.0),
    "specular":       (0.0,  1.0),
    "opacity":        (0.0,  1.0),
    "ior":            (1.0,  3.0),
    "coat":           (0.0,  1.0),
    "coat_roughness": (0.0,  1.0),
    "clearcoat":      (0.0,  1.0),
    "clearcoatRoughness": (0.0, 1.0),
    "specular_roughness": (0.04, 1.0),
    "metalness":      (0.0,  1.0),
    "transmission":   (0.0,  1.0),
    "base":           (0.0,  1.0),
}

_MATERIAL_COLOR_NAMES: set[str] = {
    "diffuseColor", "emissiveColor", "base_color", "emission_color",
    "specularColor", "coat_color",
}

_LIGHT_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "intensity": (0.0, 50.0),
    "exposure":  (-5.0, 15.0),
    "angle":     (0.0, 180.0),
    "radius":    (0.01, 100.0),
    "width":     (0.01, 100.0),
    "height":    (0.01, 100.0),
}


# ─── Builder ────────────────────────────────────────────────────────


def build_scene_graph(stage, scene, time=None) -> SceneGraphNode:
    """Build a ``SceneGraphNode`` tree from an open USD stage.

    ``scene`` is the flat ``Scene`` the renderer uses — needed to create
    ``RendererRef`` back-pointers so property edits route to the right
    renderer method.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    if time is None:
        time = Usd.TimeCode.Default()

    # Build reverse maps: prim path → renderer index
    instance_map: dict[str, int] = {}
    for i, inst in enumerate(scene.instances):
        if inst.name:
            instance_map[inst.name] = i

    material_map: dict[str, int] = {}
    for i, mat in enumerate(scene.materials):
        if mat.name and mat.name != "default":
            material_map[mat.name] = i

    light_dir_idx = 0
    light_sphere_idx = 0

    def _build_node(prim) -> SceneGraphNode:
        nonlocal light_dir_idx, light_sphere_idx

        path = str(prim.GetPath())
        name = prim.GetName() or path
        type_name = prim.GetTypeName() or "Prim"

        node = SceneGraphNode(
            path=path,
            name=name,
            type_name=type_name,
        )

        # Renderer ref + properties
        if prim.IsA(UsdGeom.Mesh):
            _add_mesh_props(node, prim, time, instance_map)
        elif prim.IsA(UsdLux.DistantLight):
            node.renderer_ref = RendererRef("light_dir", light_dir_idx)
            _add_light_props(node, prim, time)
            light_dir_idx += 1
        elif prim.IsA(UsdLux.SphereLight):
            node.renderer_ref = RendererRef("light_sphere", light_sphere_idx)
            _add_light_props(node, prim, time)
            light_sphere_idx += 1
        elif prim.IsA(UsdLux.DomeLight):
            _add_light_props(node, prim, time)
        elif prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight):
            _add_light_props(node, prim, time)
        elif _is_shade_material(prim):
            _add_material_props(node, prim, time, material_map)
        elif _is_shade_shader(prim):
            _add_shader_props(node, prim, time, material_map)
        elif prim.IsA(UsdGeom.Camera):
            node.renderer_ref = RendererRef("camera", 0)
            _add_camera_props(node, prim, time)

        if prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdGeom.Camera):
            _add_transform_props(node, prim, time, instance_map)

        _add_enabled_prop(node, prim, scene, time)

        # Recurse children
        for child in prim.GetChildren():
            if child.IsActive() and not child.IsAbstract():
                node.children.append(_build_node(child))

        return node

    pseudo_root = stage.GetPseudoRoot()
    root = SceneGraphNode(
        path="/",
        name="(stage)",
        type_name="Stage",
    )
    _add_stage_props(root, stage)

    for child in pseudo_root.GetChildren():
        if child.IsActive() and not child.IsAbstract():
            root.children.append(_build_node(child))

    return root


def _is_shade_material(prim) -> bool:
    if prim.GetTypeName() == "Material":
        return True
    from pxr import UsdShade
    try:
        mat = UsdShade.Material(prim)
        return mat and bool(mat.GetSurfaceOutput())
    except Exception:
        return False


def _is_shade_shader(prim) -> bool:
    return prim.GetTypeName() == "Shader"


# ─── Property extractors ────────────────────────────────────────────


def _add_stage_props(node: SceneGraphNode, stage) -> None:
    from pxr import UsdGeom
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    node.properties.append(SceneGraphProperty(
        name="metersPerUnit", display_name="metersPerUnit",
        type_name="float", value=meters, editable=False, metadata={},
    ))
    up = str(UsdGeom.GetStageUpAxis(stage))
    node.properties.append(SceneGraphProperty(
        name="upAxis", display_name="upAxis",
        type_name="token", value=up, editable=False, metadata={},
    ))


def _add_mesh_props(
    node: SceneGraphNode, prim, time, instance_map: dict[str, int],
) -> None:
    from pxr import UsdGeom, UsdShade
    path = str(prim.GetPath())
    idx = instance_map.get(path)
    if idx is not None:
        node.renderer_ref = RendererRef("instance", idx)

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get(time)
    fvc = mesh.GetFaceVertexCountsAttr().Get(time)
    node.properties.append(SceneGraphProperty(
        name="points", display_name="points",
        type_name="int", value=len(points) if points else 0,
        editable=False, metadata={"label": "vertex count"},
    ))
    node.properties.append(SceneGraphProperty(
        name="faces", display_name="faces",
        type_name="int", value=len(fvc) if fvc else 0,
        editable=False, metadata={"label": "face count"},
    ))

    # Material binding
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound, _ = binding_api.ComputeBoundMaterial()
    if bound:
        node.properties.append(SceneGraphProperty(
            name="material:binding", display_name="material",
            type_name="rel", value=str(bound.GetPath()),
            editable=False, metadata={},
        ))


def _add_transform_props(
    node: SceneGraphNode, prim, time, instance_map: dict[str, int],
) -> None:
    from pxr import Gf, UsdGeom
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return

    world_mat = xformable.ComputeLocalToWorldTransform(time)
    parent_world = xformable.ComputeParentToWorldTransform(time)
    local_mat = Gf.Matrix4d(world_mat) * Gf.Matrix4d(parent_world).GetInverse()

    translate, rotate, scale = _decompose_matrix(local_mat)

    path = str(prim.GetPath())
    is_editable = path in instance_map

    node.properties.append(SceneGraphProperty(
        name="translate", display_name="translate",
        type_name="vec3f", value=translate,
        editable=is_editable, metadata={},
    ))
    node.properties.append(SceneGraphProperty(
        name="rotate", display_name="rotate",
        type_name="vec3f", value=rotate,
        editable=is_editable, metadata={"unit": "degrees"},
    ))
    node.properties.append(SceneGraphProperty(
        name="scale", display_name="scale",
        type_name="vec3f", value=scale,
        editable=is_editable, metadata={},
    ))


def _decompose_matrix(gf_matrix) -> tuple[tuple, tuple, tuple]:
    """Decompose a Gf.Matrix4d into (translate, rotate_degrees, scale) tuples."""
    from pxr import Gf

    m = Gf.Matrix4d(gf_matrix)
    # Gf.Matrix4d stores row-vector convention.
    # Translation is in the last row.
    translate = (
        round(float(m[3][0]), 6),
        round(float(m[3][1]), 6),
        round(float(m[3][2]), 6),
    )

    # Extract rotation + scale from upper-left 3x3
    sx = math.sqrt(m[0][0]**2 + m[0][1]**2 + m[0][2]**2)
    sy = math.sqrt(m[1][0]**2 + m[1][1]**2 + m[1][2]**2)
    sz = math.sqrt(m[2][0]**2 + m[2][1]**2 + m[2][2]**2)

    scale = (round(sx, 6), round(sy, 6), round(sz, 6))

    # Normalized rotation matrix
    if sx > 1e-8 and sy > 1e-8 and sz > 1e-8:
        r00 = m[0][0] / sx; r01 = m[0][1] / sx; r02 = m[0][2] / sx
        r10 = m[1][0] / sy; r11 = m[1][1] / sy; r12 = m[1][2] / sy
        r20 = m[2][0] / sz; r21 = m[2][1] / sz; r22 = m[2][2] / sz

        # Euler angles (XYZ convention)
        if abs(r02) < 1.0 - 1e-6:
            pitch = math.asin(max(-1.0, min(1.0, r02)))
            yaw = math.atan2(-r12, r22)
            roll = math.atan2(-r01, r00)
        else:
            pitch = math.copysign(math.pi / 2, r02)
            yaw = math.atan2(r10, r11)
            roll = 0.0

        rotate = (
            round(math.degrees(yaw), 4),
            round(math.degrees(pitch), 4),
            round(math.degrees(roll), 4),
        )
    else:
        rotate = (0.0, 0.0, 0.0)

    return translate, rotate, scale


def _add_material_props(
    node: SceneGraphNode, prim, time, material_map: dict[str, int],
) -> None:
    from pxr import UsdShade
    mat = UsdShade.Material(prim)
    mat_name = prim.GetName()
    idx = material_map.get(mat_name)

    # Also try matching by iterating — material names in the map
    # might differ from prim names when loaded via mtlx fallback
    if idx is None:
        for name, i in material_map.items():
            if name == mat_name or str(prim.GetPath()).endswith(f"/{name}"):
                idx = i
                break

    if idx is not None:
        node.renderer_ref = RendererRef("material", idx)

    # Custom data hints
    cd = prim.GetCustomData()
    if cd:
        hint = cd.get("skinnyMaterialX")
        if hint:
            node.properties.append(SceneGraphProperty(
                name="skinnyMaterialX", display_name="skinnyMaterialX",
                type_name="string", value=str(hint),
                editable=False, metadata={},
            ))


def _add_shader_props(
    node: SceneGraphNode, prim, time, material_map: dict[str, int],
) -> None:
    from pxr import UsdShade

    shader = UsdShade.Shader(prim)
    if not shader:
        return

    # Shader ID
    id_attr = shader.GetIdAttr()
    if id_attr:
        shader_id = id_attr.Get(time)
        if shader_id:
            node.properties.append(SceneGraphProperty(
                name="info:id", display_name="shader type",
                type_name="token", value=str(shader_id),
                editable=False, metadata={},
            ))

    # Find parent material for RendererRef
    parent_mat_idx = _find_ancestor_material_idx(prim, material_map)

    for inp in shader.GetInputs():
        base_name = inp.GetBaseName()

        if inp.HasConnectedSource():
            # Texture connection — read-only
            src_info = inp.GetConnectedSource()
            if src_info:
                src_api, src_name, _ = src_info
                node.properties.append(SceneGraphProperty(
                    name=base_name, display_name=base_name,
                    type_name="rel",
                    value=f"{src_api.GetPrim().GetPath()}.{src_name}",
                    editable=False, metadata={},
                ))
            continue

        value = inp.Get()
        if value is None:
            continue

        prop = _classify_shader_input(base_name, value, parent_mat_idx)
        if prop is not None:
            node.properties.append(prop)


def _find_ancestor_material_idx(prim, material_map: dict[str, int]) -> Optional[int]:
    parent = prim.GetParent()
    while parent and str(parent.GetPath()) != "/":
        parent_name = parent.GetName()
        idx = material_map.get(parent_name)
        if idx is not None:
            return idx
        parent = parent.GetParent()
    return None


def _classify_shader_input(
    name: str, value: object, parent_mat_idx: Optional[int],
) -> Optional[SceneGraphProperty]:
    editable = parent_mat_idx is not None

    # Color3f
    if name in _MATERIAL_COLOR_NAMES or _is_gf_color3(value):
        color = _to_float_tuple(value, 3)
        if color is not None:
            return SceneGraphProperty(
                name=name, display_name=name,
                type_name="color3f", value=color,
                editable=editable, metadata={},
            )

    # Float with known range
    if name in _MATERIAL_FLOAT_RANGES:
        lo, hi = _MATERIAL_FLOAT_RANGES[name]
        return SceneGraphProperty(
            name=name, display_name=name,
            type_name="float", value=float(value),
            editable=editable, metadata={"min": lo, "max": hi},
        )

    # Generic float
    if isinstance(value, (int, float)):
        return SceneGraphProperty(
            name=name, display_name=name,
            type_name="float", value=float(value),
            editable=editable, metadata={"min": 0.0, "max": max(1.0, float(value) * 2)},
        )

    # Asset path
    if hasattr(value, "resolvedPath") or hasattr(value, "path"):
        resolved = getattr(value, "resolvedPath", None) or getattr(value, "path", "")
        return SceneGraphProperty(
            name=name, display_name=name,
            type_name="asset", value=str(resolved),
            editable=False, metadata={},
        )

    # Token/string
    if isinstance(value, str):
        return SceneGraphProperty(
            name=name, display_name=name,
            type_name="token", value=value,
            editable=False, metadata={},
        )

    return None


def _add_light_props(node: SceneGraphNode, prim, time) -> None:
    from pxr import UsdLux

    light_api = UsdLux.LightAPI(prim)
    if not light_api:
        return

    # Color
    color_attr = light_api.GetColorAttr()
    if color_attr:
        color = color_attr.Get(time)
        if color is not None:
            node.properties.append(SceneGraphProperty(
                name="color", display_name="color",
                type_name="color3f", value=_to_float_tuple(color, 3) or (1.0, 1.0, 1.0),
                editable=node.renderer_ref is not None,
                metadata={},
            ))

    # Scalar light attributes
    for attr_name, (lo, hi) in _LIGHT_FLOAT_RANGES.items():
        attr = prim.GetAttribute(f"inputs:{attr_name}")
        if attr and attr.HasAuthoredValue():
            val = attr.Get(time)
            if val is not None:
                node.properties.append(SceneGraphProperty(
                    name=attr_name, display_name=attr_name,
                    type_name="float", value=float(val),
                    editable=node.renderer_ref is not None,
                    metadata={"min": lo, "max": hi},
                ))


def _add_enabled_prop(node: SceneGraphNode, prim, scene, time) -> None:
    """Append an editable bool ``enabled`` property to ``node``.

    Two flavours, distinguished by ``metadata['toggle']``:
      - ``"node"``: leaf with a renderer_ref (instance/light/camera). Initial
        value reads from the scene object's ``enabled`` flag.
      - ``"subtree"``: an Xformable container with no renderer_ref. Initial
        value reads from UsdGeom.Imageable visibility (``"invisible"`` ⇒ off).
    """
    from pxr import UsdGeom

    if node.renderer_ref is not None:
        ref = node.renderer_ref
        initial = True
        try:
            if ref.kind == "instance" and 0 <= ref.index < len(scene.instances):
                initial = bool(scene.instances[ref.index].enabled)
            elif ref.kind == "light_dir" and 0 <= ref.index < len(scene.lights_dir):
                initial = bool(scene.lights_dir[ref.index].enabled)
            elif ref.kind == "light_sphere" and 0 <= ref.index < len(scene.lights_sphere):
                initial = bool(scene.lights_sphere[ref.index].enabled)
            elif ref.kind == "camera" and scene.camera_override is not None:
                initial = bool(scene.camera_override.enabled)
        except Exception:
            initial = True
        node.properties.append(SceneGraphProperty(
            name="enabled", display_name="enabled",
            type_name="bool", value=initial,
            editable=True, metadata={"toggle": "node"},
        ))
        return

    if prim.IsA(UsdGeom.Xformable):
        imageable = UsdGeom.Imageable(prim)
        initial = True
        if imageable:
            try:
                vis = imageable.GetVisibilityAttr().Get(time)
                initial = (vis != UsdGeom.Tokens.invisible)
            except Exception:
                initial = True
        node.properties.append(SceneGraphProperty(
            name="enabled", display_name="enabled",
            type_name="bool", value=bool(initial),
            editable=True, metadata={"toggle": "subtree"},
        ))


def _add_camera_props(node: SceneGraphNode, prim, time) -> None:
    from pxr import UsdGeom
    cam = UsdGeom.Camera(prim)
    if not cam:
        return

    for attr_name, label in (
        ("focalLength", "focal length (mm)"),
        ("verticalAperture", "vertical aperture (mm)"),
        ("horizontalAperture", "horizontal aperture (mm)"),
        ("focusDistance", "focus distance"),
        ("fStop", "f-stop"),
    ):
        attr = cam.GetPrim().GetAttribute(attr_name)
        if attr:
            val = attr.Get(time)
            if val is not None:
                node.properties.append(SceneGraphProperty(
                    name=attr_name, display_name=label,
                    type_name="float", value=float(val),
                    editable=False, metadata={},
                ))


# ─── Value conversion helpers ───────────────────────────────────────


def _is_gf_color3(value) -> bool:
    t = type(value).__name__
    return "Color3" in t or "Vec3" in t


def _to_float_tuple(value, n: int) -> Optional[tuple]:
    if hasattr(value, "asTuple"):
        t = value.asTuple()
        if len(t) >= n:
            return tuple(round(float(x), 6) for x in t[:n])
    if hasattr(value, "__getitem__") and not isinstance(value, str):
        try:
            return tuple(round(float(value[i]), 6) for i in range(n))
        except (IndexError, TypeError, ValueError):
            pass
    return None


# ─── Serialization ──────────────────────────────────────────────────


def scene_graph_to_dict(node: SceneGraphNode) -> dict:
    """Convert a scene graph tree to a JSON-serializable dict."""
    d: dict = {
        "path": node.path,
        "name": node.name,
        "type": node.type_name,
        "icon": type_icon(node.type_name),
    }
    if node.renderer_ref is not None:
        d["ref"] = {"kind": node.renderer_ref.kind, "index": node.renderer_ref.index}
    if node.properties:
        d["props"] = [
            {
                "name": p.name,
                "display": p.display_name,
                "type": p.type_name,
                "value": _serialize_value(p.value),
                "editable": p.editable,
                "meta": p.metadata,
            }
            for p in node.properties
        ]
    if node.children:
        d["children"] = [scene_graph_to_dict(c) for c in node.children]
    return d


def _serialize_value(value: object) -> object:
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "asTuple"):
        return list(value.asTuple())
    return str(value)


# ─── Lookup ─────────────────────────────────────────────────────────


def inject_renderer_camera(
    root: SceneGraphNode, camera, camera_mode: str,
) -> None:
    """Append a synthetic ``/Skinny/MainCamera`` node exposing the renderer's
    active camera as editable properties.

    Removes any prior synthesized node so this can be called repeatedly
    (e.g. on camera-mode toggle).
    """
    SYNTH_PATH = "/Skinny/MainCamera"
    root.children = [c for c in root.children if c.path != SYNTH_PATH]

    node = SceneGraphNode(
        path=SYNTH_PATH,
        name="MainCamera",
        type_name="Camera",
        renderer_ref=RendererRef("renderer_camera", 0),
    )

    node.properties.append(SceneGraphProperty(
        name="fov", display_name="fov (deg)",
        type_name="float", value=float(camera.fov),
        editable=True, metadata={"min": 1.0, "max": 170.0},
    ))
    node.properties.append(SceneGraphProperty(
        name="near", display_name="near",
        type_name="float", value=float(camera.near),
        editable=True, metadata={"min": 1e-3, "max": 10.0},
    ))
    node.properties.append(SceneGraphProperty(
        name="far", display_name="far",
        type_name="float", value=float(camera.far),
        editable=True, metadata={"min": 1.0, "max": 10000.0},
    ))
    node.properties.append(SceneGraphProperty(
        name="fstop", display_name="f-stop",
        type_name="float", value=float(getattr(camera, "fstop", 0.0)),
        editable=True, metadata={"min": 2.8, "max": 22.0},
    ))
    node.properties.append(SceneGraphProperty(
        name="focus_distance", display_name="focus dist",
        type_name="float", value=float(getattr(camera, "focus_distance", 0.0)),
        editable=True, metadata={"min": 0.0, "max": 100.0},
    ))

    # Thick-lens controls. `lens_active` is read-only telemetry; the
    # toggle below flips LensSystem.enabled and routes through
    # apply_camera_param. When no lens is authored both are inert.
    lens = getattr(camera, "lens", None)
    n_elem = len(lens.elements) if lens is not None else 0
    node.properties.append(SceneGraphProperty(
        name="lens_active", display_name="lens active",
        type_name="bool", value=bool(lens is not None and lens.enabled and n_elem > 0),
        editable=False, metadata={},
    ))
    node.properties.append(SceneGraphProperty(
        name="lens_element_count", display_name="lens elements",
        type_name="int", value=int(n_elem),
        editable=False, metadata={},
    ))
    # Enable / disable the realistic-lens path entirely (toggling it
    # off makes the renderer fall back to its pinhole model). Always
    # exposed so the user has a single switch even before any lens is
    # loaded — tied through `apply_camera_param("lens_enabled", ...)`.
    node.properties.append(SceneGraphProperty(
        name="lens_enabled", display_name="lens enabled",
        type_name="bool",
        value=bool(lens is not None and lens.enabled),
        editable=True, metadata={"toggle": "node"},
    ))
    # File-picker for swapping the authored lens at runtime. The
    # button opens an "Open USDA" dialog and feeds the result to
    # Renderer.apply_camera_lens_file.
    node.properties.append(SceneGraphProperty(
        name="lens_file", display_name="lens file",
        type_name="lens_file", value="(load .usda)",
        editable=True, metadata={},
    ))
    node.properties.append(SceneGraphProperty(
        name="yaw", display_name="yaw (rad)",
        type_name="float", value=float(camera.yaw),
        editable=True, metadata={"min": -math.pi * 2, "max": math.pi * 2},
    ))
    node.properties.append(SceneGraphProperty(
        name="pitch", display_name="pitch (rad)",
        type_name="float", value=float(camera.pitch),
        editable=True, metadata={"min": -math.pi / 2, "max": math.pi / 2},
    ))

    if camera_mode == "orbit":
        node.properties.append(SceneGraphProperty(
            name="distance", display_name="distance",
            type_name="float", value=float(camera.distance),
            editable=True, metadata={"min": 0.5, "max": 50.0},
        ))
        target = (
            float(camera.target[0]),
            float(camera.target[1]),
            float(camera.target[2]),
        )
        node.properties.append(SceneGraphProperty(
            name="target", display_name="target",
            type_name="vec3f", value=target,
            editable=True, metadata={"camera_axis": "target"},
        ))
    else:
        position = (
            float(camera.position[0]),
            float(camera.position[1]),
            float(camera.position[2]),
        )
        node.properties.append(SceneGraphProperty(
            name="position", display_name="position",
            type_name="vec3f", value=position,
            editable=True, metadata={"camera_axis": "position"},
        ))

    root.children.append(node)


def populate_instance_refs(root: SceneGraphNode, scene) -> int:
    """Walk ``root`` and assign ``RendererRef('instance', i)`` to every Mesh
    node whose USD path matches a ``scene.instances[i].name``.

    Returns the number of refs that were newly attached. Idempotent — nodes
    that already have a renderer_ref are left alone.
    """
    name_to_idx: dict[str, int] = {}
    for i, inst in enumerate(scene.instances):
        if inst.name:
            name_to_idx[inst.name] = i

    updated = 0

    def walk(node: SceneGraphNode) -> None:
        nonlocal updated
        if node.renderer_ref is None and node.type_name == "Mesh":
            idx = name_to_idx.get(node.path)
            if idx is not None:
                node.renderer_ref = RendererRef("instance", idx)
                # The 'enabled' prop was created in the Xformable branch
                # (toggle=subtree) before we knew the ref. Re-tag it as a
                # node toggle now that this Mesh has a renderer leaf.
                for p in node.properties:
                    if p.name == "enabled":
                        p.editable = True
                        p.metadata = dict(p.metadata)
                        p.metadata["toggle"] = "node"
                        try:
                            p.value = bool(scene.instances[idx].enabled)
                        except Exception:
                            pass
                updated += 1
        for child in node.children:
            walk(child)

    walk(root)
    return updated


def find_node_by_path(root: SceneGraphNode, path: str) -> Optional[SceneGraphNode]:
    if root.path == path:
        return root
    for child in root.children:
        found = find_node_by_path(child, path)
        if found is not None:
            return found
    return None


# ─── TRS → 4x4 composition (for apply_instance_transform) ──────────


def decompose_trs_matrix(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Decompose a 4x4 row-vector-convention matrix into (translate,
    rotate_deg, scale). Inverse of ``compose_trs_matrix``. Pitch is
    clamped to (-90°, 90°) to avoid gimbal-lock NaN; for matrices that
    sit on the singular pole, roll is folded into yaw.
    """
    m = np.asarray(matrix, dtype=np.float64)
    tx, ty, tz = float(m[3, 0]), float(m[3, 1]), float(m[3, 2])

    # Each row of the upper 3x3 carries the corresponding axis scaled by
    # its scale factor (since we packed S then R into rows). Magnitudes
    # recover the per-axis scales; row directions, after dividing out
    # those magnitudes, give the rotation matrix.
    row0 = m[0, :3]
    row1 = m[1, :3]
    row2 = m[2, :3]
    sx = float(np.linalg.norm(row0))
    sy = float(np.linalg.norm(row1))
    sz = float(np.linalg.norm(row2))
    if sx < 1e-9 or sy < 1e-9 or sz < 1e-9:
        return (tx, ty, tz), (0.0, 0.0, 0.0), (sx, sy, sz)
    r0 = row0 / sx
    r1 = row1 / sy
    r2 = row2 / sz

    # compose_trs_matrix layout (row-vector convention):
    #   row0 = (cy·cz,             -cy·sz,            sy)
    #   row1 = (cx·sz + sx·sy·cz,   cx·cz - sx·sy·sz, -sx·cy)
    #   row2 = (sx·sz - cx·sy·cz,   sx·cz + cx·sy·sz,  cx·cy)
    sin_y = float(np.clip(r0[2], -1.0, 1.0))
    cos_y = math.sqrt(max(0.0, 1.0 - sin_y * sin_y))
    if cos_y > 1e-6:
        rx = math.atan2(-r1[2], r2[2])
        ry = math.asin(sin_y)
        rz = math.atan2(-r0[1], r0[0])
    else:
        # Gimbal lock: pitch ≈ ±90°. Fold roll into yaw.
        rx = 0.0
        ry = math.asin(sin_y)
        rz = math.atan2(r1[0], r1[1])

    rotate_deg = (math.degrees(rx), math.degrees(ry), math.degrees(rz))
    return (tx, ty, tz), rotate_deg, (sx, sy, sz)


def compose_trs_matrix(
    translate: tuple[float, float, float],
    rotate_deg: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> np.ndarray:
    """Compose translate, rotate (degrees, XYZ Euler), scale into a 4x4
    row-vector-convention matrix matching USD/skinny's storage layout."""
    tx, ty, tz = translate
    rx, ry, rz = (math.radians(a) for a in rotate_deg)
    sx, sy, sz = scale

    # Rotation matrices (intrinsic XYZ)
    cx, sx_ = math.cos(rx), math.sin(rx)
    cy, sy_ = math.cos(ry), math.sin(ry)
    cz, sz_ = math.cos(rz), math.sin(rz)

    # Combined rotation R = Rz * Ry * Rx (row-vector convention: v * R)
    r00 = cy * cz;            r01 = -cy * sz_;           r02 = sy_
    r10 = cx * sz_ + sx_ * sy_ * cz; r11 = cx * cz - sx_ * sy_ * sz_; r12 = -sx_ * cy
    r20 = sx_ * sz_ - cx * sy_ * cz; r21 = sx_ * cz + cx * sy_ * sz_; r22 = cx * cy

    m = np.array([
        [r00 * sx, r01 * sx, r02 * sx, 0.0],
        [r10 * sy, r11 * sy, r12 * sy, 0.0],
        [r20 * sz, r21 * sz, r22 * sz, 0.0],
        [tx,       ty,       tz,       1.0],
    ], dtype=np.float32)
    return m
