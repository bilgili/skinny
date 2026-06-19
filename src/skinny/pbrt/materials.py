"""pbrt material -> UsdPreviewSurface mapping (design D4).

Roughness calibration chain (parity-critical):

* pbrt v4: ``alpha = sqrt(roughness)`` when ``remaproughness`` (default), else
  ``alpha = roughness``.
* skinny GGX: ``alpha = usd_roughness**2`` (UsdPreviewSurface roughness passes
  straight through to the GGX perceptual roughness).
* therefore ``usd_roughness = sqrt(alpha)``.

Anisotropic ``uroughness``/``vroughness`` are reduced to an isotropic alpha via
the geometric mean and flagged.
"""

from __future__ import annotations

import math
import os

from . import spectra
from .parser import ParamSet

# pbrt material input -> (UsdPreviewSurface input, is_scalar)
_TEXTURABLE = {
    "reflectance": ("diffuseColor", False),
    "roughness": ("roughness", True),
}


def resolve_texture(name: str, textures, base_dir=None, _depth: int = 0):
    """Resolve a pbrt texture name to (abs_path, color_space) or None.

    Handles ``imagemap`` directly and unwraps a ``scale`` texture to its inner
    image. Other texture classes (checkerboard/mix/constant) are unsupported.
    """
    if textures is None or _depth > 4:
        return None
    tex = textures.get(name)
    if tex is None:
        return None
    if tex.klass == "imagemap":
        fname = tex.params.string("filename", None)
        if not fname:
            return None
        if base_dir and not os.path.isabs(fname):
            fname = os.path.join(base_dir, fname)
        encoding = tex.params.string("encoding", None)
        color_space = "raw" if (encoding == "linear" or tex.datatype == "float") else "sRGB"
        return fname, color_space
    if tex.klass == "scale":
        inner = tex.params.get("tex")
        if inner is not None and inner.type == "texture":
            return resolve_texture(inner.string, textures, base_dir, _depth + 1)
    return None


def references_texture(pbrt_material, textures, base_dir=None) -> bool:
    """True if any of *pbrt_material*'s inputs resolves to an imagemap texture.

    Used to decide whether a UV-less shape needs synthesized default UVs so the
    bound texture can sample (matching pbrt) rather than read a constant point.
    """
    if pbrt_material is None or textures is None:
        return False
    p = pbrt_material.params
    for pbrt_in in _TEXTURABLE:
        pp = p.get(pbrt_in)
        if (pp is not None and pp.type == "texture"
                and resolve_texture(pp.string, textures, base_dir) is not None):
            return True
    return False


def pbrt_roughness_to_alpha(roughness: float, remap: bool = True) -> float:
    """pbrt v4 roughness -> GGX alpha."""
    r = max(float(roughness), 0.0)
    return math.sqrt(r) if remap else r


def alpha_to_usd_roughness(alpha: float) -> float:
    """skinny GGX uses alpha = roughness**2, so usd_roughness = sqrt(alpha)."""
    return math.sqrt(max(alpha, 0.0))


def _resolve_roughness(params, notes: list[str]) -> float:
    remap = params.bool("remaproughness", True)
    if "uroughness" in params or "vroughness" in params:
        ur = params.float("uroughness", params.float("roughness", 0.0))
        vr = params.float("vroughness", params.float("roughness", 0.0))
        au = pbrt_roughness_to_alpha(ur, remap)
        av = pbrt_roughness_to_alpha(vr, remap)
        notes.append("anisotropic roughness reduced to isotropic (geometric mean)")
        alpha = math.sqrt(max(au, 1e-8) * max(av, 1e-8))
    else:
        alpha = pbrt_roughness_to_alpha(params.float("roughness", 0.0), remap)
    return alpha_to_usd_roughness(alpha)


def _conductor_basecolor(params, notes: list[str]):
    if "reflectance" in params:
        return spectra.param_to_rgb(params.get("reflectance")) or [0.9, 0.9, 0.9]
    eta_p, k_p = params.get("eta"), params.get("k")
    # named spectra (strings) -> tabulated metal IOR
    if eta_p is not None and eta_p.type == "spectrum" and isinstance(eta_p.values[0], str):
        refl = spectra.named_metal_reflectance_rgb(eta_p.values[0])
        if refl is not None:
            return list(refl)
    if eta_p is not None and k_p is not None:
        try:
            eta = spectra.param_to_rgb(eta_p) or eta_p.floats[:3]
            k = spectra.param_to_rgb(k_p) or k_p.floats[:3]
            return list(spectra.fresnel_conductor_rgb(eta, k))
        except Exception:  # noqa: BLE001 - fall through to default
            pass
    notes.append("conductor IOR unresolved; defaulted to copper")
    return list(spectra.named_metal_reflectance_rgb("copper"))


def map_material(pbrt_material, *, emissive_rgb=None, textures=None, base_dir=None):
    """Map a pbrt material to UsdPreviewSurface inputs.

    Returns (inputs, tex_inputs, status, notes): ``inputs`` are constant
    UsdPreviewSurface inputs, ``tex_inputs`` maps a UsdPreviewSurface input name
    to ``(image_path, color_space)`` for texture-connected inputs, and
    ``status`` is one of report.EXACT/APPROX/SKIPPED.
    """
    from .report import APPROX, EXACT

    notes: list[str] = []
    inputs: dict = {
        "diffuseColor": [0.5, 0.5, 0.5],
        "metallic": 0.0,
        "roughness": 0.5,
        "opacity": 1.0,
    }
    status = EXACT
    mtype = pbrt_material.type if pbrt_material else "diffuse"
    # an emissive shape may carry no material; use an empty set so reads default
    p = pbrt_material.params if pbrt_material else ParamSet()

    if mtype in ("", "none"):
        inputs["diffuseColor"] = [0.5, 0.5, 0.5]
    elif mtype == "diffuse":
        inputs["diffuseColor"] = spectra.param_to_rgb(p.get("reflectance")) or [0.5, 0.5, 0.5]
        inputs["roughness"] = 1.0
    elif mtype == "conductor":
        inputs["metallic"] = 1.0
        inputs["diffuseColor"] = _conductor_basecolor(p, notes)
        inputs["roughness"] = _resolve_roughness(p, notes)
    elif mtype in ("dielectric", "thindielectric"):
        inputs["diffuseColor"] = [1.0, 1.0, 1.0]
        inputs["opacity"] = 0.0  # open skinny's transmission/refraction gate
        inputs["ior"] = p.float("eta", 1.5) if p else 1.5
        inputs["roughness"] = _resolve_roughness(p, notes) if p else 0.0
        if mtype == "thindielectric":
            status = APPROX
            notes.append("thindielectric approximated as thin dielectric")
    elif mtype == "coateddiffuse":
        inputs["diffuseColor"] = spectra.param_to_rgb(p.get("reflectance")) or [0.5, 0.5, 0.5]
        inputs["roughness"] = 1.0
        inputs["clearcoat"] = 1.0
        inputs["clearcoatRoughness"] = _resolve_roughness(p, notes)
    elif mtype == "coatedconductor":
        inputs["metallic"] = 1.0
        inputs["diffuseColor"] = _conductor_basecolor(p, notes)
        inputs["roughness"] = _resolve_roughness(p, notes)
        inputs["clearcoat"] = 1.0
        inputs["clearcoatRoughness"] = p.float("interface.roughness", 0.0) if p else 0.0
    elif mtype == "diffusetransmission":
        inputs["diffuseColor"] = spectra.param_to_rgb(p.get("reflectance")) or [0.25, 0.25, 0.25]
        inputs["opacity"] = 0.5
        status = APPROX
        notes.append("diffusetransmission approximated as diffuse + partial opacity")
    elif mtype == "subsurface":
        inputs["diffuseColor"] = [1.0, 1.0, 1.0]
        inputs["opacity"] = 0.0
        inputs["ior"] = p.float("eta", 1.33) if p else 1.33
        status = APPROX
        notes.append("subsurface -> dielectric boundary + homogeneous interior (customData)")
    else:
        status = APPROX
        notes.append(f"unknown material '{mtype}' best-effort as diffuse grey")

    if emissive_rgb is not None:
        inputs["emissiveColor"] = list(emissive_rgb)

    # texture-connected inputs (imagemap reflectance/roughness)
    tex_inputs: dict = {}
    for pbrt_in, (usd_in, _scalar) in _TEXTURABLE.items():
        pp = p.get(pbrt_in)
        if pp is not None and pp.type == "texture":
            res = resolve_texture(pp.string, textures, base_dir)
            if res is not None:
                tex_inputs[usd_in] = res
            else:
                notes.append(f"texture '{pp.string}' on {pbrt_in} unresolved/unsupported")

    return inputs, tex_inputs, status, notes
