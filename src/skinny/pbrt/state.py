"""pbrt graphics-state machine.

Walks a directive list and builds a USD-agnostic intermediate representation
(:class:`PbrtScene`) that the translator turns into USD. Reproduces pbrt's
graphics-state model: CTM stack, named materials/textures, area-light state,
reverse-orientation, named coordinate systems, named media, and
``ObjectBegin``/``ObjectInstance`` instancing.

All transforms are kept in *pbrt space* (left-handed); the change-of-basis to
skinny's right-handed world is applied later, at bake time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import transform as T
from .errors import PbrtParseError
from .parser import Directive, ParamSet


@dataclass
class PbrtMaterial:
    type: str
    params: ParamSet
    name: str = ""  # set for named materials


@dataclass
class PbrtTexture:
    name: str
    datatype: str  # 'float' | 'spectrum'
    klass: str  # 'imagemap' | 'scale' | 'checkerboard' | 'constant' | 'mix'
    params: ParamSet


@dataclass
class PbrtMedium:
    name: str
    type: str
    params: ParamSet
    ctm: np.ndarray = field(default_factory=T.identity)  # CTM at MakeNamedMedium, pbrt space


@dataclass
class PbrtShape:
    type: str
    params: ParamSet
    ctm: np.ndarray  # object-to-world in pbrt space
    material: PbrtMaterial | None
    area_light: ParamSet | None
    reverse_orientation: bool
    inside_medium: str = ""
    outside_medium: str = ""


@dataclass
class PbrtLight:
    type: str
    params: ParamSet
    ctm: np.ndarray  # light-to-world in pbrt space


@dataclass
class PbrtCamera:
    type: str
    params: ParamSet
    camera_to_world: np.ndarray  # pbrt space (inverse of the CTM at `Camera`)


@dataclass
class PbrtScene:
    camera: PbrtCamera | None = None
    film: ParamSet | None = None
    sampler: tuple[str, ParamSet] | None = None
    integrator: tuple[str, ParamSet] | None = None
    color_space: str = "srgb"
    shapes: list[PbrtShape] = field(default_factory=list)
    lights: list[PbrtLight] = field(default_factory=list)
    named_materials: dict[str, PbrtMaterial] = field(default_factory=dict)
    textures: dict[str, PbrtTexture] = field(default_factory=dict)
    media: dict[str, PbrtMedium] = field(default_factory=dict)


@dataclass
class _State:
    ctm: np.ndarray
    material: PbrtMaterial | None
    area_light: ParamSet | None
    reverse_orientation: bool
    inside_medium: str
    outside_medium: str

    def clone(self) -> "_State":
        return _State(
            self.ctm.copy(),
            self.material,
            self.area_light,
            self.reverse_orientation,
            self.inside_medium,
            self.outside_medium,
        )


def build_scene(directives: list[Directive]) -> PbrtScene:
    """Run the graphics-state machine over *directives* and return the IR."""
    scene = PbrtScene()
    state = _State(T.identity(), None, None, False, "", "")
    attr_stack: list[_State] = []
    xform_stack: list[np.ndarray] = []
    coord_systems: dict[str, np.ndarray] = {}

    # instancing
    objects: dict[str, list[PbrtShape]] = {}
    object_begin_ctm: dict[str, np.ndarray] = {}
    current_object: str | None = None

    def emit_shape(d: Directive) -> None:
        shp = PbrtShape(
            type=d.type_arg() or "",
            params=d.params,
            ctm=state.ctm.copy(),
            material=state.material,
            area_light=state.area_light,
            reverse_orientation=state.reverse_orientation,
            inside_medium=state.inside_medium,
            outside_medium=state.outside_medium,
        )
        if current_object is not None:
            objects[current_object].append(shp)
        else:
            scene.shapes.append(shp)

    for d in directives:
        name = d.name
        if name == "WorldBegin":
            state.ctm = T.identity()
            coord_systems.clear()
        elif name == "WorldEnd":
            pass
        elif name == "Identity":
            state.ctm = T.identity()
        elif name == "Translate":
            state.ctm = state.ctm @ T.translate(*_n(d, 3))
        elif name == "Scale":
            state.ctm = state.ctm @ T.scale(*_n(d, 3))
        elif name == "Rotate":
            a, x, y, z = _n(d, 4)
            state.ctm = state.ctm @ T.rotate(a, x, y, z)
        elif name == "Transform":
            state.ctm = T.from_pbrt_array(_array_arg(d))
        elif name == "ConcatTransform":
            state.ctm = state.ctm @ T.from_pbrt_array(_array_arg(d))
        elif name == "LookAt":
            vals = _n(d, 9)
            c2w = T.look_at(vals[0:3], vals[3:6], vals[6:9])
            state.ctm = state.ctm @ T.invert(c2w)
        elif name == "CoordinateSystem":
            coord_systems[d.type_arg()] = state.ctm.copy()
        elif name == "CoordSysTransform":
            key = d.type_arg()
            if key in coord_systems:
                state.ctm = coord_systems[key].copy()
        elif name == "AttributeBegin":
            attr_stack.append(state.clone())
        elif name == "AttributeEnd":
            if not attr_stack:
                raise PbrtParseError("AttributeEnd without AttributeBegin", line=d.line)
            state = attr_stack.pop()
        elif name in ("TransformBegin",):
            xform_stack.append(state.ctm.copy())
        elif name in ("TransformEnd",):
            if not xform_stack:
                raise PbrtParseError("TransformEnd without TransformBegin", line=d.line)
            state.ctm = xform_stack.pop()
        elif name == "ObjectBegin":
            attr_stack.append(state.clone())
            current_object = d.type_arg()
            objects.setdefault(current_object, [])
            object_begin_ctm[current_object] = state.ctm.copy()
        elif name == "ObjectEnd":
            current_object = None
            if attr_stack:
                state = attr_stack.pop()
        elif name == "ObjectInstance":
            _instantiate(d, scene, state, objects, object_begin_ctm)
        elif name == "ReverseOrientation":
            state.reverse_orientation = not state.reverse_orientation
        elif name == "Material":
            state.material = PbrtMaterial(d.type_arg() or "", d.params)
        elif name == "MakeNamedMaterial":
            mname = d.type_arg() or ""
            mtype = d.params.string("type", "")
            scene.named_materials[mname] = PbrtMaterial(mtype, d.params, name=mname)
        elif name == "NamedMaterial":
            mname = d.type_arg() or ""
            state.material = scene.named_materials.get(mname)
        elif name == "Texture":
            _define_texture(d, scene)
        elif name == "MakeNamedMedium":
            mname = d.type_arg() or ""
            mtype = d.params.string("type", "homogeneous")
            scene.media[mname] = PbrtMedium(mname, mtype, d.params, ctm=state.ctm.copy())
        elif name == "MediumInterface":
            strs = [a for a in d.args if isinstance(a, str)]
            state.inside_medium = strs[0] if len(strs) > 0 else ""
            state.outside_medium = strs[1] if len(strs) > 1 else ""
        elif name == "AreaLightSource":
            state.area_light = d.params
        elif name == "LightSource":
            scene.lights.append(PbrtLight(d.type_arg() or "", d.params, state.ctm.copy()))
        elif name == "Shape":
            emit_shape(d)
        elif name == "Camera":
            # CTM at Camera is world-to-camera; camera-to-world is its inverse.
            scene.camera = PbrtCamera(d.type_arg() or "", d.params, T.invert(state.ctm))
        elif name == "Film":
            scene.film = d.params
        elif name == "Sampler":
            scene.sampler = (d.type_arg() or "", d.params)
        elif name == "Integrator":
            scene.integrator = (d.type_arg() or "", d.params)
        elif name == "ColorSpace":
            scene.color_space = d.type_arg() or "srgb"
        else:
            # Accelerator, PixelFilter, Option, Attribute, etc. — ignored for v1
            pass
    return scene


def _instantiate(d, scene, state, objects, object_begin_ctm) -> None:
    key = d.type_arg()
    if key not in objects:
        raise PbrtParseError(f"ObjectInstance of unknown object {key!r}", line=d.line)
    base = object_begin_ctm.get(key, T.identity())
    base_inv = T.invert(base)
    for shp in objects[key]:
        local = base_inv @ shp.ctm  # object-local transform
        world = state.ctm @ local
        inst = copy.copy(shp)
        inst.ctm = world
        scene.shapes.append(inst)


def _define_texture(d: Directive, scene: PbrtScene) -> None:
    strs = [a for a in d.args if isinstance(a, str)]
    if len(strs) < 3:
        raise PbrtParseError("Texture needs name, datatype, and class", line=d.line)
    tname, datatype, klass = strs[0], strs[1], strs[2]
    scene.textures[tname] = PbrtTexture(tname, datatype, klass, d.params)


def _n(d: Directive, count: int) -> list[float]:
    nums = [float(a) for a in d.args if not isinstance(a, (str, list))]
    if len(nums) < count:
        raise PbrtParseError(
            f"{d.name} expected {count} numbers, got {len(nums)}", line=d.line
        )
    return nums[:count]


def _array_arg(d: Directive) -> list:
    for a in d.args:
        if isinstance(a, list):
            return a
    raise PbrtParseError(f"{d.name} expected a [..] array", line=d.line)
