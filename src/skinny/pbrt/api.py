"""Top-level pbrt -> USD orchestration.

``import_pbrt(path)`` parses a pbrt v4 scene, runs the graphics-state machine,
and authors a USD stage loadable by skinny's ``usd_loader``, returning the stage
and a :class:`~skinny.pbrt.report.Report`.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from . import camera as camera_mod
from . import emit
from . import materials as materials_mod
from . import media as media_mod
from . import spectra
from . import transform as T
from .lights import add_light
from .parser import parse_file
from .ply import read_ply
from .report import Report
from .state import PbrtScene, PbrtShape, build_scene


def import_pbrt(path: str, out: str | None = None):
    """Parse a pbrt scene file and emit USD. Returns (stage, report)."""
    scene = build_scene(parse_file(path))
    return translate_scene(scene, out)


def translate_scene(scene: PbrtScene, out: str | None = None):
    report = Report()
    stage = emit.new_stage(out)
    world = "/World"

    for i, shp in enumerate(scene.shapes):
        _emit_shape(stage, f"{world}/shape_{i}", shp, report, scene)

    if scene.camera is not None:
        _emit_camera(stage, f"{world}/Camera", scene, report)
    else:
        report.skipped("camera", "scene has no Camera")

    asset_dir = os.path.dirname(os.path.abspath(out)) if out else None
    for light in scene.lights:
        add_light(stage, world, light, report, asset_dir=asset_dir)

    if out is not None:
        stage.GetRootLayer().Export(out)
    return stage, report


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _shape_geometry(shp: PbrtShape, report):
    """Return (points_local (N,3), indices (M,3), normals|None, uvs|None) or None."""
    p = shp.params
    t = shp.type
    if t == "trianglemesh":
        pts = np.asarray(p.floats("P", []), dtype=np.float64).reshape(-1, 3)
        idx = np.asarray(p.ints("indices", []), dtype=np.int64).reshape(-1, 3)
        nrm = None
        if "N" in p:
            nrm = np.asarray(p.floats("N"), dtype=np.float64).reshape(-1, 3)
        uvs = None
        for key in ("uv", "st"):
            if key in p:
                uvs = np.asarray(p.floats(key), dtype=np.float64).reshape(-1, 2)
                break
        return pts, idx, nrm, uvs
    if t == "plymesh":
        fname = p.string("filename", None)
        if not fname:
            report.skipped("shape:plymesh", "no filename")
            return None
        try:
            mesh = read_ply(fname)
        except Exception as exc:  # noqa: BLE001
            report.skipped("shape:plymesh", f"failed to read {fname}: {exc}")
            return None
        return mesh.points, mesh.indices, mesh.normals, mesh.uvs
    if t == "sphere":
        radius = p.float("radius", 1.0)
        pts, idx, nrm = emit.tessellate_sphere(radius)
        return pts, idx, nrm, None
    report.skipped(f"shape:{t}", "unsupported shape type")
    return None


def _emit_shape(stage, path, shp: PbrtShape, report, scene: PbrtScene) -> None:
    geo = _shape_geometry(shp, report)
    if geo is None:
        return
    pts_local, idx_local, nrm_local, uvs = geo
    pts, idx, nrm = emit.bake_world_mesh(
        pts_local, idx_local, shp.ctm, normals_local=nrm_local,
        reverse=shp.reverse_orientation,
    )
    mesh = emit.add_mesh(stage, path, pts, idx, normals=nrm, uvs=uvs)
    report.exact(f"shape:{shp.type} {path}")

    emissive = None
    if shp.area_light is not None:
        emissive = spectra.param_to_rgb(shp.area_light.get("L"), illuminant=True) or [1, 1, 1]
        scale = shp.area_light.float("scale", 1.0)
        emissive = [c * scale for c in emissive]
        twosided = shp.area_light.bool("twosided", False)
        report.approx(
            f"arealight {path}",
            "two-sided emission" if twosided else "one-sided emission (skinny may differ)",
        )

    overrides = _resolve_medium(shp, scene, report, path)
    _author_material(stage, f"{path}_mat", shp.material, mesh, report,
                     emissive_rgb=emissive, extra_overrides=overrides)


def _resolve_medium(shp: PbrtShape, scene: PbrtScene, report, path) -> dict | None:
    if not shp.inside_medium:
        return None
    medium = scene.media.get(shp.inside_medium)
    if medium is None:
        return None
    if media_mod.is_heterogeneous(medium):
        report.skipped(f"medium:{medium.type} {path}", "heterogeneous media unsupported")
        return None
    report.approx(f"medium:homogeneous {path}", "coefficients carried via customData")
    return media_mod.homogeneous_overrides(medium)


def _author_material(stage, mat_path, pbrt_material, mesh_prim, report,
                     emissive_rgb=None, extra_overrides=None):
    inputs, status, notes = materials_mod.map_material(pbrt_material, emissive_rgb=emissive_rgb)
    overrides = dict(extra_overrides or {})
    if pbrt_material is not None and pbrt_material.type == "subsurface":
        overrides.update(media_mod.subsurface_overrides(pbrt_material.params))
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for key, val in inputs.items():
        if key in ("diffuseColor", "emissiveColor", "specularColor"):
            shader.CreateInput(key, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in val]))
        else:
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(val))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if overrides:
        mat.GetPrim().SetCustomDataByKey("skinnyOverrides", overrides)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(mat)
    mtype = pbrt_material.type if pbrt_material else "diffuse"
    detail = "; ".join(notes)
    report.add(f"material:{mtype} {mat_path}", status, detail)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def _emit_camera(stage, path, scene: PbrtScene, report) -> None:
    cam = scene.camera
    film = scene.film
    xres = film.float("xresolution", 1280.0) if film else 1280.0
    yres = film.float("yresolution", 720.0) if film else 720.0
    aspect = (xres / yres) if yres else (16.0 / 9.0)

    notes: list[str] = []
    if cam.type == "perspective":
        intr = camera_mod.perspective_to_camera(cam.params, aspect, notes)
        status = "exact"
    elif cam.type == "realistic":
        intr = camera_mod.perspective_to_camera(cam.params, aspect, notes)
        notes.append("realistic lens not yet mapped; used a perspective fallback")
        status = "approx"
    else:
        intr = camera_mod.perspective_to_camera(cam.params, aspect, notes)
        notes.append(f"camera type '{cam.type}' approximated as perspective")
        status = "approx"

    usd_cam = UsdGeom.Camera.Define(stage, path)
    usd_cam.CreateFocalLengthAttr(float(intr["focal_length_mm"]))
    usd_cam.CreateVerticalApertureAttr(float(intr["vertical_aperture_mm"]))
    usd_cam.CreateHorizontalApertureAttr(float(intr["horizontal_aperture_mm"]))
    if "fstop" in intr:
        usd_cam.CreateFStopAttr(float(intr["fstop"]))
    if "focus_distance" in intr:
        usd_cam.CreateFocusDistanceAttr(float(intr["focus_distance"]))
    UsdGeom.Xformable(usd_cam).AddTransformOp().Set(
        emit.to_gf_matrix(T.to_skinny(cam.camera_to_world))
    )
    report.add(f"camera:{cam.type} {path}", status, "; ".join(notes))
