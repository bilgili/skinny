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
from dataclasses import dataclass

from . import spectra
from .parser import ParamSet

# pbrt material input -> (UsdPreviewSurface input, value_type). ``value_type`` is
# the single source of truth for how the texture connects: "color3f" drives the
# UsdUVTexture ``.rgb`` output, "float" the scalar ``.r`` output.
_TEXTURABLE = {
    "reflectance": ("diffuseColor", "color3f"),
    "roughness": ("roughness", "float"),
}
# UsdPreviewSurface input -> connection value_type, derived from _TEXTURABLE so
# the map stays the one source of truth (consumed by api._author_texture).
_USD_INPUT_KIND = {usd_in: vt for usd_in, vt in _TEXTURABLE.values()}


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


@dataclass(frozen=True)
class ParamValue:
    """A material parameter resolved the pbrt way: a constant, optionally with a
    bound texture. Mirrors pbrt's ``GetFloatTexture``/``GetSpectrumTexture``, which
    promote a constant to a ``*ConstantTexture`` so a material never branches on
    "float or texture". ``const`` is the scalar/rgb fallback; ``tex`` is the
    resolved ``(image_path, color_space)`` when the param is a supported texture.
    """

    const: object
    tex: tuple | None = None

    @property
    def is_tex(self) -> bool:
        return self.tex is not None


def get_float_texture(params, name, default, *, textures=None, base_dir=None, notes=None):
    """Resolve a FloatTexture-able parameter like pbrt's ``GetFloatTexture``.

    Constant -> ``ParamValue(value)``; named texture -> ``ParamValue(default, tex)``;
    absent -> ``ParamValue(default)``. Never raises on a texture-typed param: pbrt
    ``ErrorExit``s on an unknown/unsupported texture, skinny falls back to the
    default and records an APPROX note (best-effort translator).
    """
    p = params.get(name)
    if p is None:
        return ParamValue(float(default))
    if p.type == "texture":
        res = resolve_texture(p.string, textures, base_dir)
        if res is not None:
            return ParamValue(float(default), res)
        if notes is not None:
            notes.append(f"texture '{p.string}' on {name} unresolved/unsupported; used default")
        return ParamValue(float(default))
    return ParamValue(p.float)


def get_spectrum_texture(params, name, default_rgb, *, textures=None, base_dir=None,
                         notes=None, illuminant=False):
    """Resolve a SpectrumTexture-able parameter like pbrt's ``GetSpectrumTexture``.

    Constant spectrum -> ``ParamValue(rgb)``; named texture -> ``ParamValue(
    default_rgb, tex)``; absent -> ``ParamValue(default_rgb)``. Texture-safe.
    """
    p = params.get(name)
    if p is None:
        return ParamValue(list(default_rgb))
    if p.type == "texture":
        res = resolve_texture(p.string, textures, base_dir)
        if res is not None:
            return ParamValue(list(default_rgb), res)
        if notes is not None:
            notes.append(f"texture '{p.string}' on {name} unresolved/unsupported; used default")
        return ParamValue(list(default_rgb))
    rgb = spectra.param_to_rgb(p, illuminant=illuminant)
    return ParamValue(list(rgb) if rgb is not None else list(default_rgb))


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


def _resolve_roughness(params, notes: list[str], *, textures=None, base_dir=None) -> ParamValue:
    """Resolve roughness as a ParamValue (const usd_roughness, optional texture).

    A texture-bound ``roughness``/``uroughness``/``vroughness`` connects the
    texture and uses a mid scalar fallback (the perceptual sqrt remap cannot be
    applied to a USD texture connection without an extra node — flagged approx).
    The all-constant path reproduces the prior isotropic/anisotropic chain.
    """
    remap = params.bool("remaproughness", True)
    # resolve all three via the texture-safe accessor (never float() a texture name)
    rough = get_float_texture(params, "roughness", 0.0, textures=textures, base_dir=base_dir, notes=notes)
    urough = get_float_texture(params, "uroughness", rough.const, textures=textures, base_dir=base_dir, notes=notes)
    vrough = get_float_texture(params, "vroughness", rough.const, textures=textures, base_dir=base_dir, notes=notes)
    tex = rough.tex or urough.tex or vrough.tex
    if tex is not None:
        notes.append(
            "roughness texture connected; perceptual remap not applied to texture (approx)"
        )
        return ParamValue(0.5, tex)
    if "uroughness" in params or "vroughness" in params:
        au = pbrt_roughness_to_alpha(urough.const, remap)
        av = pbrt_roughness_to_alpha(vrough.const, remap)
        notes.append("anisotropic roughness reduced to isotropic (geometric mean)")
        alpha = math.sqrt(max(au, 1e-8) * max(av, 1e-8))
    else:
        alpha = pbrt_roughness_to_alpha(rough.const, remap)
    return ParamValue(alpha_to_usd_roughness(alpha))


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
    to ``(image_path, color_space, value_type)`` for texture-connected inputs
    (``value_type`` in {"color3f","float"}), and ``status`` is one of
    report.EXACT/APPROX/SKIPPED.

    Every textureable parameter is resolved through the promoting accessors
    (``get_float_texture``/``get_spectrum_texture``, mirroring pbrt's
    ``GetFloatTexture``/``GetSpectrumTexture``), so a texture-bound value yields a
    connection on its *own* USD input while a constant yields a plain input -- one
    uniform pass, no float-vs-texture branching, nothing assumed to be diffuse.
    """
    from .report import APPROX, EXACT

    notes: list[str] = []
    inputs: dict = {
        "diffuseColor": [0.5, 0.5, 0.5],
        "metallic": 0.0,
        "roughness": 0.5,
        "opacity": 1.0,
    }
    tex_inputs: dict = {}
    status = EXACT
    mtype = pbrt_material.type if pbrt_material else "diffuse"
    # an emissive shape may carry no material; use an empty set so reads default
    p = pbrt_material.params if pbrt_material else ParamSet()

    def put(usd_in, pv):
        """Apply a ParamValue: constant input + optional texture connection."""
        inputs[usd_in] = pv.const
        if pv.is_tex:
            path, color_space = pv.tex
            tex_inputs[usd_in] = (path, color_space, _USD_INPUT_KIND.get(usd_in, "float"))

    def reflectance(default):
        return get_spectrum_texture(
            p, "reflectance", default, textures=textures, base_dir=base_dir, notes=notes
        )

    def roughness():
        return _resolve_roughness(p, notes, textures=textures, base_dir=base_dir)

    def scalar(name, default):
        # USD has no texture input for these (ior / clearcoatRoughness); a texture
        # binding degrades to the scalar default with a note.
        pv = get_float_texture(p, name, default, textures=textures, base_dir=base_dir, notes=notes)
        if pv.is_tex:
            notes.append(f"{name} texture not supported on USD input; used scalar default")
        return pv.const

    if mtype in ("", "none"):
        inputs["diffuseColor"] = [0.5, 0.5, 0.5]
    elif mtype == "diffuse":
        put("diffuseColor", reflectance([0.5, 0.5, 0.5]))
        inputs["roughness"] = 1.0
    elif mtype == "conductor":
        inputs["metallic"] = 1.0
        put("diffuseColor", reflectance(_conductor_basecolor(p, notes)))
        put("roughness", roughness())
    elif mtype in ("dielectric", "thindielectric"):
        inputs["diffuseColor"] = [1.0, 1.0, 1.0]
        inputs["opacity"] = 0.0  # open skinny's transmission/refraction gate
        inputs["ior"] = scalar("eta", 1.5)
        put("roughness", roughness())
        if mtype == "thindielectric":
            status = APPROX
            notes.append("thindielectric approximated as thin dielectric")
    elif mtype == "coateddiffuse":
        put("diffuseColor", reflectance([0.5, 0.5, 0.5]))
        inputs["roughness"] = 1.0
        inputs["clearcoat"] = 1.0
        rv = roughness()
        inputs["clearcoatRoughness"] = rv.const
        if rv.is_tex:
            notes.append("coat roughness texture not connected on USD clearcoatRoughness; used scalar")
    elif mtype == "coatedconductor":
        inputs["metallic"] = 1.0
        put("diffuseColor", reflectance(_conductor_basecolor(p, notes)))
        put("roughness", roughness())
        inputs["clearcoat"] = 1.0
        inputs["clearcoatRoughness"] = scalar("interface.roughness", 0.0)
    elif mtype == "diffusetransmission":
        put("diffuseColor", reflectance([0.25, 0.25, 0.25]))
        inputs["opacity"] = 0.5
        status = APPROX
        notes.append("diffusetransmission approximated as diffuse + partial opacity")
    elif mtype == "subsurface":
        inputs["diffuseColor"] = [1.0, 1.0, 1.0]
        inputs["opacity"] = 0.0
        inputs["ior"] = scalar("eta", 1.33)
        status = APPROX
        notes.append("subsurface -> dielectric boundary + homogeneous interior (customData)")
    else:
        status = APPROX
        notes.append(f"unknown material '{mtype}' best-effort as diffuse grey")

    if emissive_rgb is not None:
        inputs["emissiveColor"] = list(emissive_rgb)

    # an unresolved/unsupported texture binding degrades to its scalar/rgb default
    # (handled in the accessors) -> surface as APPROX.
    if status == EXACT and any("unresolved/unsupported" in n for n in notes):
        status = APPROX

    return inputs, tex_inputs, status, notes
