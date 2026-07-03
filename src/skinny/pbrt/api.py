"""Top-level pbrt -> USD orchestration.

``import_pbrt(path)`` parses a pbrt v4 scene, runs the graphics-state machine,
and authors a USD stage loadable by skinny's ``usd_loader``, returning the stage
and a :class:`~skinny.pbrt.report.Report`.
"""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from . import camera as camera_mod
from . import emit
from . import loopsubdiv as loopsubdiv_mod
from . import materials as materials_mod
from . import media as media_mod
from . import metadata as meta_mod
from . import spectra
from . import transform as T
from .lights import add_light
from .parser import parse_file
from .ply import read_ply
from .report import Report
from .state import PbrtMaterial, PbrtScene, PbrtShape, build_scene

# Inputs that are carried as `skinnyOverrides` customData (merged into
# Material.parameter_overrides by the loader), NOT authored as UsdPreviewSurface
# shader inputs. The subsurface medium coefficients are list-valued (per-channel
# σ) and consumed by the renderer's inline-medium packer, so authoring them as
# scalar shader inputs both fails (`float([...])`) and is meaningless here.
_OVERRIDE_ONLY_INPUTS = frozenset({
    "subsurface_sigma_a", "subsurface_sigma_s", "subsurface_g", "subsurface_eta",
})


def import_pbrt(path: str, out: str | None = None, materialx: bool = False):
    """Parse a pbrt scene file and emit USD. Returns (stage, report).

    When *materialx* is true and *out* is set, rich ``standard_surface``
    materials are written to a sidecar ``.mtlx`` next to the ``.usda`` and the
    stage references it (instead of authoring UsdPreviewSurface shaders).
    """
    scene = build_scene(parse_file(path))
    base_dir = os.path.dirname(os.path.abspath(path))
    return translate_scene(scene, out, base_dir=base_dir, materialx=materialx)


def translate_scene(
    scene: PbrtScene,
    out: str | None = None,
    base_dir: str | None = None,
    materialx: bool = False,
):
    report = Report()
    stage = emit.new_stage(out)
    world = "/World"

    # MaterialX sidecar export is path-anchored (the .mtlx lands next to the
    # .usda and the stage references it by basename). With no output path there
    # is nowhere to write the sidecar, so fall back to the UsdPreviewSurface
    # authoring path for an in-memory stage.
    emit_mtlx = bool(materialx and out is not None)
    # surfacematerial element name (== Material prim leaf) -> standard_surface
    # description package consumed by mtlx_emit.write_mtlx_document.
    mtlx_materials: dict[str, dict] = {}
    # (material_prim_path, surfacematerial_name) pairs to wire to the sidecar.
    mtlx_refs: list[tuple[str, str]] = []

    # pbrt film exposure (imagingRatio = exposureTime * ISO / 100, film.cpp) is a
    # global linear scale on output radiance. It is NOT baked into emitters anymore
    # (change pbrt-radiometric-parity): it is authored on the camera prim as
    # skinny:film:iso / skinny:film:exposureTime and applied live by the renderer
    # as an output scale, so ISO/exposure retune on the fly and a .hdr-direct env's
    # own pbrt `scale` is never lost. For a linear path tracer scaling the output
    # is algebraically identical to scaling every emitter, so a default-film render
    # is radiometrically unchanged. Emitters/env therefore carry only their own
    # per-light pbrt `scale` (exposure_scale=1.0 below).
    exposure_scale = _film_exposure_scale(scene)
    if exposure_scale != 1.0:
        report.exact("film:exposure",
                     f"imagingRatio={exposure_scale:.4g} authored on camera (live output scale)")

    # Medium names for which a UsdVol.Volume prim has already been authored
    # (one prim per named medium, however many shapes reference it).
    emitted_volumes: set[str] = set()
    for i, shp in enumerate(scene.shapes):
        _emit_shape(
            stage, f"{world}/shape_{i}", shp, report, scene, base_dir, 1.0,
            emit_mtlx=emit_mtlx, mtlx_materials=mtlx_materials, mtlx_refs=mtlx_refs,
            emitted_volumes=emitted_volumes,
        )

    if scene.camera is not None:
        _emit_camera(stage, f"{world}/Camera", scene, report, base_dir)
    else:
        report.skipped("camera", "scene has no Camera")

    asset_dir = os.path.dirname(os.path.abspath(out)) if out else None
    for light in scene.lights:
        add_light(stage, world, light, report, asset_dir=asset_dir,
                  exposure_scale=1.0, base_dir=base_dir)

    # carry the exact pbrt scene config (integrator/sampler/film/colorspace)
    meta_mod.tag_stage(stage, meta_mod.scene_metadata(scene))

    # Integrator mapping (change photon-mapping-sppm): pbrt's `sppm` maps onto
    # skinny's SPPM integrator, so report it as mapped (not skipped) and the
    # normalized skinny selection is recorded in the stage metadata above.
    if scene.integrator is not None and scene.integrator[0] in ("sppm", "photonmap"):
        report.exact("integrator:sppm",
                     "mapped to skinny SPPM (wavefront, flat materials)")

    # MaterialX sidecar: write the .mtlx document and author its references on
    # the Material prims (downgrading the bound def-Materials to typeless overs)
    # before exporting, so the reference specs are part of the exported layer.
    if emit_mtlx and mtlx_materials:
        from . import mtlx_emit

        mtlx_out = os.path.splitext(out)[0] + ".mtlx"
        mtlx_emit.write_mtlx_document(mtlx_materials, mtlx_out)
        mtlx_asset = os.path.basename(mtlx_out)
        for mat_path, sm_name in mtlx_refs:
            mtlx_emit.author_mtlx_reference(stage, mat_path, mtlx_asset, sm_name)
        report.exact(
            "materialx:export",
            f"wrote {os.path.basename(mtlx_out)} "
            f"({len(mtlx_materials)} standard_surface material(s))",
        )

    if out is not None:
        stage.GetRootLayer().Export(out)
    return stage, report


def sppm_selection(stage):
    """Return the normalized skinny SPPM selection written into an imported
    stage's ``customLayerData["pbrt"]["skinny"]`` when the pbrt scene used
    ``Integrator "sppm"`` / ``"photonmap"`` — a dict ``{"integrator": "sppm",
    "radius"?: float, "photons"?: int}`` — or ``None`` for any other integrator.

    A loader or the parity harness uses this to select skinny's SPPM integrator
    and seed its initial radius / photons-per-pass from the pbrt parameters."""
    pbrt = dict(stage.GetRootLayer().customLayerData).get("pbrt", {})
    return pbrt.get("skinny")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _shape_geometry(shp: PbrtShape, report, base_dir=None):
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
        if base_dir and not os.path.isabs(fname):
            fname = os.path.join(base_dir, fname)
        try:
            mesh = read_ply(fname)
        except Exception as exc:  # noqa: BLE001
            report.skipped("shape:plymesh", f"failed to read {fname}: {exc}")
            return None
        return mesh.points, mesh.indices, mesh.normals, mesh.uvs
    if t == "sphere":
        radius = p.float("radius", 1.0)
        pts, idx, nrm, uvs = emit.tessellate_sphere(radius)
        return pts, idx, nrm, uvs
    if t == "disk":
        radius = p.float("radius", 1.0)
        height = p.float("height", 0.0)
        inner_radius = p.float("innerradius", 0.0)
        phi_max = p.float("phimax", 360.0)
        pts, idx, nrm, uvs = emit.tessellate_disk(
            radius, height=height, inner_radius=inner_radius, phi_max=phi_max,
        )
        return pts, idx, nrm, uvs
    if t == "loopsubdiv":
        # pbrt tessellates a Loop control cage to a triangle mesh (limit surface)
        # before rendering; do the same at import and reuse the trianglemesh path.
        cage_p = np.asarray(p.floats("P", []), dtype=np.float64)
        cage_idx = np.asarray(p.ints("indices", []), dtype=np.int64)
        if cage_p.size == 0 or cage_idx.size == 0 or cage_idx.size % 3 != 0:
            report.skipped("shape:loopsubdiv", "missing or malformed P/indices")
            return None
        levels = p.int("levels", 3)  # pbrt default
        try:
            pts, idx, nrm = loopsubdiv_mod.subdivide(cage_p, cage_idx, levels)
        except ValueError as exc:
            report.skipped("shape:loopsubdiv", str(exc))
            return None
        return pts, idx, nrm, None
    report.skipped(f"shape:{t}", "unsupported shape type")
    return None


def _film_exposure_scale(scene: PbrtScene) -> float:
    """pbrt imagingRatio = exposureTime * ISO / 100 (exposureTime from shutter)."""
    iso = scene.film.float("iso", 100.0) if scene.film else 100.0
    if scene.camera is not None:
        cp = scene.camera.params
        exposure_time = cp.float("shutterclose", 1.0) - cp.float("shutteropen", 0.0)
    else:
        exposure_time = 1.0
    return max(exposure_time, 0.0) * iso / 100.0


def _emit_shape(stage, path, shp: PbrtShape, report, scene: PbrtScene, base_dir=None,
                exposure_scale: float = 1.0, *, emit_mtlx: bool = False,
                mtlx_materials: dict | None = None,
                mtlx_refs: list | None = None,
                emitted_volumes: set | None = None) -> None:
    geo = _shape_geometry(shp, report, base_dir)
    if geo is None:
        return
    pts_local, idx_local, nrm_local, uvs = geo
    pts, idx, nrm = emit.bake_world_mesh(
        pts_local, idx_local, shp.ctm, normals_local=nrm_local,
        reverse=shp.reverse_orientation,
    )
    uv_interp = "vertex"
    if uvs is None and materials_mod.references_texture(
        shp.material, scene.textures, base_dir
    ):
        # pbrt assigns default UVs to every primitive; synthesize per-triangle
        # faceVarying (0,0),(1,0),(1,1) so the bound texture samples like pbrt.
        uvs = emit.default_triangle_uvs(idx.shape[0])
        uv_interp = "faceVarying"
        report.approx(f"uv:default {path}", "synthesized pbrt default UVs (no source UV)")
    mesh = emit.add_mesh(stage, path, pts, idx, normals=nrm, uvs=uvs,
                         uv_interpolation=uv_interp)
    report.exact(f"shape:{shp.type} {path}")

    emissive = None
    if shp.area_light is not None:
        emissive = spectra.param_to_rgb(shp.area_light.get("L"), illuminant=True) or [1, 1, 1]
        scale = shp.area_light.float("scale", 1.0) * exposure_scale
        emissive = [c * scale for c in emissive]
        twosided = shp.area_light.bool("twosided", False)
        report.approx(
            f"arealight {path}",
            "two-sided emission" if twosided else "one-sided emission (skinny may differ)",
        )
        meta_mod.tag_arealight(mesh.GetPrim(), meta_mod.arealight_metadata(shp.area_light))

    overrides = _resolve_medium(stage, shp, scene, report, path, base_dir, emitted_volumes)
    material = shp.material
    if (material is not None and material.type == ""
            and (shp.inside_medium or shp.outside_medium)):
        # pbrt `Material ""` on a shape carrying a MediumInterface is the
        # null/interface material (pbrt-volume-import spec) — route it through
        # the exact same encoding as `Material "interface"` instead of the
        # grey-diffuse fallback. Gated on the shape having a MediumInterface
        # (inside OR outside medium — a cavity boundary `MediumInterface "" "m"`
        # is still a null boundary, just with no importer-supported interior);
        # `Material ""` with no MediumInterface keeps the default material.
        material = PbrtMaterial("interface", material.params)
        report.exact(f"material:null {path}",
                     'Material "" + MediumInterface -> null boundary (interface)')
    if material is not None and material.type == "interface":
        # Null/boundary material: mark the routing key explicitly so the
        # renderer-side predicate does not have to sniff lobe values (D2/3.2).
        overrides = dict(overrides or {})
        overrides["volume_interface"] = True
    _author_material(stage, f"{path}_mat", material, mesh, report,
                     emissive_rgb=emissive, extra_overrides=overrides,
                     textures=scene.textures, base_dir=base_dir,
                     emit_mtlx=emit_mtlx, mtlx_materials=mtlx_materials,
                     mtlx_refs=mtlx_refs)


def _resolve_medium(stage, shp: PbrtShape, scene: PbrtScene, report, path, base_dir=None,
                    emitted_volumes: set | None = None) -> dict | None:
    if not shp.inside_medium:
        return None
    medium = scene.media.get(shp.inside_medium)
    if medium is None:
        return None
    if media_mod.is_supported_heterogeneous(medium):
        report.exact(f"medium:{medium.type} {path}", "grid coefficients carried via customData")
        overrides = media_mod.heterogeneous_overrides(medium, base_dir)
        _emit_volume_once(stage, scene, medium, overrides, report, emitted_volumes)
        return overrides
    if media_mod.is_supported_cloud(medium):
        # pbrt's built-in procedural cloud (analytic fBm density) — no grid
        # file, no Volume prim: the density parameters + world→medium-local
        # rows ride the bound material's skinnyOverrides and the renderer
        # evaluates pbrt's CloudMedium::Density in-shader (MEDIUM_CLOUD).
        report.exact(f"medium:{medium.type} {path}",
                     "procedural cloud parameters carried via customData")
        return media_mod.cloud_overrides(medium)
    if media_mod.is_heterogeneous(medium):
        report.skipped(f"medium:{medium.type} {path}", "heterogeneous media unsupported")
        return None
    report.approx(f"medium:homogeneous {path}", "coefficients carried via customData")
    return media_mod.homogeneous_overrides(medium)


def _emit_volume_once(stage, scene: PbrtScene, medium, overrides: dict, report,
                      emitted_volumes: set | None) -> None:
    """Author the one ``UsdVol.Volume`` prim for *medium*, the first time any
    shape references it (multiple shapes may share the same named medium)."""
    if emitted_volumes is None or medium.name in emitted_volumes:
        return
    emitted_volumes.add(medium.name)
    grid_asset = overrides.get("volume_grid_asset")
    if not grid_asset:
        return
    vol_path = f"/World/volume_{emit.sanitize(medium.name)}"
    emit.add_volume(
        stage, vol_path, medium.ctm,
        grid_asset=grid_asset, field_name=overrides.get("volume_grid_field", "density"),
    )
    vol_prim = stage.GetPrimAtPath(vol_path)
    vol_prim.SetCustomDataByKey("skinnyOverrides", dict(overrides))
    report.exact(f"volume:{medium.type} {vol_path}", f"UsdVol.Volume referencing {grid_asset}")


def _author_material(stage, mat_path, pbrt_material, mesh_prim, report,
                     emissive_rgb=None, extra_overrides=None, textures=None, base_dir=None,
                     *, emit_mtlx: bool = False, mtlx_materials: dict | None = None,
                     mtlx_refs: list | None = None):
    if emit_mtlx:
        _author_material_mtlx(
            stage, mat_path, pbrt_material, mesh_prim, report,
            emissive_rgb=emissive_rgb, extra_overrides=extra_overrides,
            textures=textures, base_dir=base_dir,
            mtlx_materials=mtlx_materials, mtlx_refs=mtlx_refs,
        )
        return
    inputs, tex_inputs, status, notes = materials_mod.map_material(
        pbrt_material, emissive_rgb=emissive_rgb, textures=textures, base_dir=base_dir
    )
    overrides = dict(extra_overrides or {})
    if pbrt_material is not None and pbrt_material.type == "subsurface":
        overrides.update(media_mod.subsurface_overrides(pbrt_material.params))
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    for key, val in inputs.items():
        if key in tex_inputs:
            continue  # textured input is authored as a connection below
        if key in _OVERRIDE_ONLY_INPUTS:
            continue  # medium coeffs ride on skinnyOverrides, not the shader
        if key in ("diffuseColor", "emissiveColor", "specularColor"):
            shader.CreateInput(key, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in val]))
        else:
            shader.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(val))
    for usd_in, (tex_path, color_space, value_type) in tex_inputs.items():
        _author_texture(stage, shader, f"{mat_path}/{usd_in}_tex", usd_in, tex_path,
                        color_space, value_type)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if overrides:
        mat.GetPrim().SetCustomDataByKey("skinnyOverrides", overrides)
    # carry the exact pbrt material spec (type + raw params) for lossless recovery
    meta_mod.tag_prim(mat.GetPrim(), meta_mod.material_metadata(pbrt_material))
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(mat)
    mtype = pbrt_material.type if pbrt_material else "diffuse"
    detail = "; ".join(notes)
    report.add(f"material:{mtype} {mat_path}", status, detail)
    if tex_inputs:
        report.exact(f"textures {mat_path}", ", ".join(tex_inputs))


def _author_material_mtlx(stage, mat_path, pbrt_material, mesh_prim, report,
                          emissive_rgb=None, extra_overrides=None, textures=None,
                          base_dir=None, mtlx_materials=None, mtlx_refs=None):
    """Author a Material bound to a sidecar ``.mtlx`` standard_surface.

    Mirrors :func:`_author_material` but maps the pbrt material to the richer
    ``standard_surface`` slots (:func:`materials.map_material_mtlx`), binds the
    mesh, and records the standard_surface package + ``(prim_path, sm_name)``
    pair for :func:`mtlx_emit.write_mtlx_document` /
    :func:`mtlx_emit.author_mtlx_reference` (called once, after the stage is
    built). Crucially it authors **no** UsdPreviewSurface shader on the prim —
    per the loader contract the prim is later downgraded to a typeless ``over``
    so the ``.mtlx`` fallback fires regardless of the usdMtlx plugin.
    """
    inputs, tex_inputs, status, notes = materials_mod.map_material_mtlx(
        pbrt_material, emissive_rgb=emissive_rgb, textures=textures, base_dir=base_dir
    )
    overrides = dict(extra_overrides or {})
    if pbrt_material is not None and pbrt_material.type == "subsurface":
        overrides.update(media_mod.subsurface_overrides(pbrt_material.params))

    mat = UsdShade.Material.Define(stage, mat_path)
    # No UsdPreviewSurface shader, no surface output: ComputeBoundMaterial must
    # not resolve a shader on this prim (so the .mtlx fallback fires).
    if overrides:
        mat.GetPrim().SetCustomDataByKey("skinnyOverrides", overrides)
    meta_mod.tag_prim(mat.GetPrim(), meta_mod.material_metadata(pbrt_material))
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(mat)

    sm_name = Sdf.Path(mat_path).name
    if mtlx_materials is not None:
        mtlx_materials[sm_name] = {"inputs": inputs, "tex_inputs": tex_inputs}
    if mtlx_refs is not None:
        mtlx_refs.append((mat_path, sm_name))

    mtype = pbrt_material.type if pbrt_material else "diffuse"
    detail = "; ".join(notes)
    report.add(f"material:{mtype} {mat_path}", status, detail)
    if tex_inputs:
        report.exact(f"textures {mat_path}", ", ".join(tex_inputs))


def _author_texture(stage, shader, tex_prim_path, usd_in, image_path, color_space, value_type):
    """Author a UsdUVTexture node and connect it to *usd_in* on *shader*.

    Output channel and input type come from *value_type* (the `_TEXTURABLE` map's
    single source of truth): "color3f" -> `.rgb`/Color3f, "float" -> `.r`/Float.
    """
    scalar = value_type == "float"
    tex = UsdShade.Shader.Define(stage, tex_prim_path)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image_path))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
    out = tex.CreateOutput(
        "r" if scalar else "rgb",
        Sdf.ValueTypeNames.Float if scalar else Sdf.ValueTypeNames.Float3,
    )
    si = shader.CreateInput(
        usd_in, Sdf.ValueTypeNames.Float if scalar else Sdf.ValueTypeNames.Color3f
    )
    si.ConnectToSource(out)


# --------------------------------------------------------------------------- #
# camera
# --------------------------------------------------------------------------- #
def _emit_camera(stage, path, scene: PbrtScene, report, base_dir=None) -> None:
    cam = scene.camera
    film = scene.film
    xres = film.float("xresolution", 1280.0) if film else 1280.0
    yres = film.float("yresolution", 720.0) if film else 720.0
    aspect = (xres / yres) if yres else (16.0 / 9.0)

    notes: list[str] = []
    intr = camera_mod.perspective_to_camera(cam.params, aspect, notes)
    lens = None
    if cam.type == "perspective":
        status = "exact"
    elif cam.type == "realistic":
        lens = camera_mod.realistic_lens(cam.params, base_dir, notes)
        status = "exact" if lens else "approx"
    else:
        notes.append(f"camera type '{cam.type}' approximated as perspective")
        status = "approx"

    usd_cam = UsdGeom.Camera.Define(stage, path)
    usd_cam.CreateFocalLengthAttr(float(intr["focal_length_mm"]))
    usd_cam.CreateVerticalApertureAttr(float(intr["vertical_aperture_mm"]))
    usd_cam.CreateHorizontalApertureAttr(float(intr["horizontal_aperture_mm"]))
    # pbrt film exposure controls (change pbrt-radiometric-parity): author ISO and
    # exposure time on the camera prim so the renderer applies the imaging ratio
    # exposure_time·iso/100 as a live output scale (instead of baking it into
    # emitters). The standard UsdGeom `exposure` attr mirrors exposure_time so a
    # non-skinny USD consumer still sees a film exposure. usd_loader._extract_camera
    # reads these back into CameraOverride.iso / .exposure_time.
    film_iso = film.float("iso", 100.0) if film else 100.0
    exposure_time = (cam.params.float("shutterclose", 1.0)
                     - cam.params.float("shutteropen", 0.0))
    exposure_time = max(exposure_time, 0.0)
    imaging_ratio = exposure_time * film_iso / 100.0
    cam_prim = usd_cam.GetPrim()
    cam_prim.CreateAttribute("skinny:film:iso", Sdf.ValueTypeNames.Float).Set(float(film_iso))
    cam_prim.CreateAttribute(
        "skinny:film:exposureTime", Sdf.ValueTypeNames.Float).Set(float(exposure_time))
    # The standard UsdGeom `exposure` attr is a log2 (stops) gain, so author
    # log2(imaging_ratio) — a generic USD consumer with no skinny:film:* support
    # still reproduces the same brightness. Skipped when the ratio is non-positive.
    if imaging_ratio > 0.0:
        usd_cam.CreateExposureAttr(float(math.log2(imaging_ratio)))
    if "fstop" in intr:
        usd_cam.CreateFStopAttr(float(intr["fstop"]))
    if "focus_distance" in intr:
        usd_cam.CreateFocusDistanceAttr(float(intr["focus_distance"]))
    UsdGeom.Xformable(usd_cam).AddTransformOp().Set(
        emit.to_gf_matrix(T.to_skinny(cam.camera_to_world))
    )
    if lens:
        _author_lens(stage, path, lens)
        notes.append(f"realistic lens: {len(lens)} elements")

    camera_md = meta_mod.camera_metadata(cam)
    if T.is_orientation_reversing(cam.camera_to_world):
        # e.g. a `Scale -1 1 1` before LookAt makes the camera basis improper;
        # skinny reconstructs a right-handed camera from position+forward, so the
        # image comes out horizontally mirrored vs pbrt. Flag it (carry in
        # metadata) rather than silently shipping a mirrored render.
        camera_md["mirrored"] = True
        notes.append("improper camera transform (e.g. Scale -1): image mirrored horizontally vs pbrt")
        status = "approx"
    meta_mod.tag_prim(usd_cam.GetPrim(), camera_md)
    report.add(f"camera:{cam.type} {path}", status, "; ".join(notes))


def _author_lens(stage, cam_path, elements) -> None:
    """Author one Xform child per lens surface with skinny:lens:* attributes."""
    for el in elements:
        child = UsdGeom.Xform.Define(stage, f"{cam_path}/lens_{el['order']}")
        prim = child.GetPrim()
        prim.CreateAttribute("skinny:lens:role", Sdf.ValueTypeNames.String).Set(el["role"])
        prim.CreateAttribute("skinny:lens:radius", Sdf.ValueTypeNames.Float).Set(el["radius"])
        prim.CreateAttribute("skinny:lens:thickness", Sdf.ValueTypeNames.Float).Set(el["thickness"])
        prim.CreateAttribute("skinny:lens:ior", Sdf.ValueTypeNames.Float).Set(el["ior"])
        prim.CreateAttribute("skinny:lens:aperture", Sdf.ValueTypeNames.Float).Set(el["aperture"])
        prim.CreateAttribute("skinny:lens:order", Sdf.ValueTypeNames.Int).Set(el["order"])
