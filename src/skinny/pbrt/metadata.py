"""Carry the exact pbrt v4 source semantics as USD metadata.

The UsdPreviewSurface / UsdLux output is a *portable approximation*. This module
additionally records the original pbrt parameters losslessly so skinny (or a
pbrt round-trip) can recover the exact spec even where the USD schema is lossy
(spectra, exact BxDFs, integrator, sensor):

* per-prim ``customData["pbrt"]`` on each Material / Light / Camera,
* stage-wide ``customLayerData["pbrt"]`` for integrator / sampler / film / color
  space.

Each entity dict is ``{type, [name], params, paramTypes}`` where ``params`` maps
each pbrt parameter to a native USD-friendly value and ``paramTypes`` records its
pbrt parameter type (``float`` / ``rgb`` / ``spectrum`` / ``blackbody`` /
``bool`` / ``string`` / ``texture`` / …) so the value is self-describing.
"""

from __future__ import annotations

from .parser import Param, ParamSet

_PBRT_KEY = "pbrt"


def _param_native(p: Param):
    """Reduce a pbrt Param to a native USD-friendly value (or None if empty)."""
    t = p.type
    vals = p.values
    if not vals:
        return None
    if t in ("integer", "int"):
        return int(round(float(vals[0]))) if len(vals) == 1 else [int(round(float(v))) for v in vals]
    if t == "bool":
        out = [str(v).lower() == "true" for v in vals]
        return out[0] if len(out) == 1 else out
    if t in ("string", "texture"):
        return str(vals[0])
    if t == "blackbody":
        return float(vals[0])
    if t == "spectrum":
        # named spectrum -> string; inline sampled spectrum -> [l, v, l, v, ...]
        return str(vals[0]) if isinstance(vals[0], str) else [float(v) for v in vals]
    # numeric vector/scalar (float, rgb, color, point*, vector*, normal*)
    nums = []
    for v in vals:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            return str(vals[0])
    return nums[0] if len(nums) == 1 else nums


def paramset_to_dicts(ps: ParamSet) -> tuple[dict, dict]:
    """Return (params, paramTypes) native dicts for a ParamSet."""
    params: dict = {}
    types: dict = {}
    if ps is None:
        return params, types
    for name, p in ps.params.items():
        native = _param_native(p)
        if native is None:
            continue
        params[name] = native
        types[name] = p.type
    return params, types


def _entity_md(type_str: str, ps: ParamSet, **extra) -> dict:
    params, types = paramset_to_dicts(ps)
    md: dict = {"type": type_str}
    md.update({k: v for k, v in extra.items() if v})
    if params:
        md["params"] = params
        md["paramTypes"] = types
    return md


def material_metadata(material) -> dict:
    if material is None:
        return {"type": "diffuse"}
    return _entity_md(material.type, material.params, name=material.name)


def light_metadata(light) -> dict:
    return _entity_md(light.type, light.params)


def camera_metadata(camera) -> dict:
    return _entity_md(camera.type, camera.params)


def arealight_metadata(area_params) -> dict:
    """pbrt ``AreaLightSource "diffuse"`` emission spec (L / scale / twosided)."""
    return _entity_md("diffuse", area_params)


def scene_metadata(scene) -> dict:
    """Stage-wide pbrt config for ``customLayerData["pbrt"]``."""
    md: dict = {"colorSpace": scene.color_space}
    if scene.integrator is not None:
        itype = scene.integrator[0]
        md["integrator"] = _entity_md(itype, scene.integrator[1])
        # pbrt's SPPM maps 1:1 onto skinny's SPPM integrator (change
        # photon-mapping-sppm). Record a normalized skinny selection so a loader
        # / the parity harness can pick the integrator + seed its initial radius
        # without re-parsing pbrt param names. `radius` seeds the SPPM search
        # radius; `photonsperiteration` the photons-per-pass override.
        if itype in ("sppm", "photonmap"):
            iparams, _ = paramset_to_dicts(scene.integrator[1])
            sel: dict = {"integrator": "sppm"}
            if "radius" in iparams:
                sel["radius"] = float(iparams["radius"])
            if "photonsperiteration" in iparams:
                sel["photons"] = int(iparams["photonsperiteration"])
            md["skinny"] = sel
    if scene.sampler is not None:
        md["sampler"] = _entity_md(scene.sampler[0], scene.sampler[1])
    if scene.film is not None:
        fparams, ftypes = paramset_to_dicts(scene.film)
        if fparams:
            md["film"] = {"params": fparams, "paramTypes": ftypes}
    return md


def tag_prim(prim, md: dict) -> None:
    """Attach a pbrt metadata dict to a prim's customData."""
    prim.SetCustomDataByKey(_PBRT_KEY, md)


def tag_arealight(prim, md: dict) -> None:
    """Attach pbrt area-light emission metadata to a (mesh) prim's customData."""
    prim.SetCustomDataByKey("pbrtAreaLight", md)


def tag_stage(stage, md: dict) -> None:
    """Attach stage-wide pbrt metadata to the root layer's customLayerData."""
    layer = stage.GetRootLayer()
    cld = dict(layer.customLayerData)
    cld[_PBRT_KEY] = md
    layer.customLayerData = cld
