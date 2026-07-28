"""pbrt material mapping: one resolver, two emit adapters.

:func:`resolve_material` is the **single owner** of pbrt-param interpretation —
which params each material type reads, with what defaults, texture promotion,
named-spectrum substitution, the roughness calibration chain and the subsurface
coefficient precedence. It returns a target-agnostic :class:`ResolvedMaterial`
(ordered neutral lobes + status + notes). :func:`map_material`
(UsdPreviewSurface) and :func:`map_material_mtlx` (MaterialX standard_surface)
are thin adapters over it and read no ``ParamSet`` themselves, so a new pbrt
param is wired exactly once. A hostless gate in
``tests/pbrt/test_material_resolve.py`` fails the build if a param read
reappears in an adapter.

What stays in an adapter is only what the target's vocabulary forces: input
renaming, anisotropy reduction policy, transmission encoding (``opacity`` gate
vs ``transmission``), emission encoding (``emissiveColor`` vs unit ``emission``
weight + ``emission_color``), ``value_type`` derivation, and *dropping* lobes
the target cannot express. Lobe ORDER is load-bearing: each adapter seeds its
base inputs then applies lobes in resolved order, which reproduces both targets'
historical input ordering (and hence their emitted bytes).

The ``flavor`` argument on the resolver is a deliberate wart: reading a param
has side effects (an unresolved texture appends a note and escalates
EXACT→APPROX, and notes/statuses are import output), so the handful of reads
that are one-sided or divergent between the two pipelines today are gated
rather than unified. Each gate names its follow-up.

Roughness calibration chain (parity-critical):

* pbrt v4: ``alpha = sqrt(roughness)`` when ``remaproughness`` (default), else
  ``alpha = roughness``.
* skinny GGX: ``alpha = usd_roughness**2`` (UsdPreviewSurface roughness passes
  straight through to the GGX perceptual roughness).
* therefore ``usd_roughness = sqrt(alpha)``.

Anisotropic ``uroughness``/``vroughness`` resolve **unreduced** (both alphas);
UsdPreviewSurface collapses them to the isotropic geometric mean and flags it,
standard_surface keeps a mean + ``specular_anisotropy``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from . import spectra
from .data import spectral_tables as _st
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

#: Authoring flavours the resolver is invoked with. The flavour exists only to
#: freeze reads that are one-sided or divergent between the two pipelines today
#: (see :func:`resolve_material`); a read is *not* performed under a flavour
#: whose mapper does not perform it today, because the promoting accessors
#: append notes and escalate EXACT→APPROX as a side effect of reading, and notes
#: and statuses are import output.
USD = "usd"
MTLX = "mtlx"


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

    A named ``spectrum`` value has no scalar form of its own, but a **named
    glass** (e.g. a dielectric ``"spectrum eta" "glass-BK7"``) does: its d-line
    (589.3 nm) index. That is what pbrt renders such a dielectric with in RGB, so
    it is used here in place of the generic default — otherwise every named glass
    would render at the caller's fallback (1.5), turning e.g. `glass-LASF9`
    (n=1.850) into a plain crown. The *dispersive* identity is preserved
    separately on ``skinnyOverrides`` (:func:`material_spectral_overrides`) and
    only the spectral build consumes it.
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
    if p.type == "spectrum" and p.values and isinstance(p.values[0], str):
        return ParamValue(_named_spectrum_scalar(p.string, name, default, notes))
    return ParamValue(p.float)


#: Float params whose value is a refractive index, so a named glass resolves to
#: its d-line IOR. Every OTHER float param routed through `get_float_texture` is a
#: roughness (`roughness`, `uroughness`, `vroughness`, `conductor.roughness`,
#: `interface.roughness`) — handing one a glass IOR would silently write 1.51673
#: into a roughness lane instead of degrading to the caller's default with a note.
_IOR_PARAM_NAMES = frozenset({"eta", "interface.eta"})


def _named_spectrum_scalar(spectrum_name, param_name, default, notes) -> float:
    """Scalar for a named-spectrum-valued float param, reporting any substitution.

    A recognised glass on an **IOR-bearing** param yields its d-line IOR silently
    (it is exact, not a fallback). Anything else degrades to *default* with an
    APPROX note that names what was unrecognised — a spectrum **file** reference is
    called out as such rather than mis-reported as an unknown glass (pbrt reads a
    file when a name misses its table; skinny has no reader).
    """
    if param_name in _IOR_PARAM_NAMES and _st.glass_is_known(spectrum_name):
        return float(_st.named_glass_ior_d(spectrum_name))
    if notes is not None:
        if spectra.looks_like_spectrum_file(spectrum_name):
            notes.append(
                f"spectrum file '{spectrum_name}' on {param_name} not read "
                f"(unsupported); used default {default}")
        else:
            notes.append(
                f"named spectrum '{spectrum_name}' on {param_name} unrecognised; "
                f"used default {default}")
    return float(default)


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


@dataclass(frozen=True)
class ResolvedRoughness:
    """Roughness resolved to its *unreduced* form; the reduction is adapter policy.

    Exactly one shape is populated:

    * ``pv`` — the isotropic (or texture-bound) result, already calibrated
      through ``pbrt_roughness_to_alpha`` → ``alpha_to_usd_roughness``;
    * ``alpha_u``/``alpha_v`` — the two GGX alphas of an anisotropic
      ``uroughness``/``vroughness`` pair, *before* any collapse, so each adapter
      applies its own policy (UsdPreviewSurface collapses to the isotropic
      geometric mean, standard_surface keeps a mean + ``specular_anisotropy``).
      The alphas — not the perceptual roughnesses — are carried because the
      UsdPreviewSurface collapse operates on alphas; round-tripping through
      ``sqrt`` would perturb the last bits.
    """

    pv: ParamValue | None = None
    alpha_u: float | None = None
    alpha_v: float | None = None

    @property
    def is_aniso(self) -> bool:
        return self.pv is None


def _resolve_roughness(params, notes: list[str], *, textures=None, base_dir=None,
                       flavor: str) -> ResolvedRoughness:
    """Resolve ``roughness``/``uroughness``/``vroughness`` for either target.

    A texture-bound axis connects the texture and uses a mid scalar fallback
    (the perceptual sqrt remap cannot be applied to a texture connection without
    an extra node — flagged approx). The all-constant path yields either the
    isotropic calibrated roughness or the unreduced alpha pair.

    ``flavor`` only gates the anisotropy *note*: the UsdPreviewSurface target
    loses the two axes in its collapse and says so, standard_surface represents
    them faithfully and has nothing to report.
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
        return ResolvedRoughness(ParamValue(0.5, tex))
    if "uroughness" in params or "vroughness" in params:
        if flavor == USD:
            notes.append("anisotropic roughness reduced to isotropic (geometric mean)")
        return ResolvedRoughness(
            alpha_u=pbrt_roughness_to_alpha(urough.const, remap),
            alpha_v=pbrt_roughness_to_alpha(vrough.const, remap),
        )
    return ResolvedRoughness(
        ParamValue(alpha_to_usd_roughness(pbrt_roughness_to_alpha(rough.const, remap)))
    )


def _usd_roughness(rr: ResolvedRoughness) -> ParamValue:
    """UsdPreviewSurface reduction: anisotropic axes collapse to their geometric mean."""
    if not rr.is_aniso:
        return rr.pv
    alpha = math.sqrt(max(rr.alpha_u, 1e-8) * max(rr.alpha_v, 1e-8))
    return ParamValue(alpha_to_usd_roughness(alpha))


def _mtlx_roughness(rr: ResolvedRoughness) -> tuple[ParamValue, float]:
    """standard_surface reduction: mean perceptual roughness + ``specular_anisotropy``.

    ``specular_anisotropy`` in ``[0, 1)`` encodes the axis ratio (0 == isotropic).
    """
    if not rr.is_aniso:
        return rr.pv, 0.0
    ru = alpha_to_usd_roughness(rr.alpha_u)
    rv = alpha_to_usd_roughness(rr.alpha_v)
    hi, lo = max(ru, rv), min(ru, rv)
    return ParamValue(0.5 * (ru + rv)), (0.0 if hi <= 1e-8 else (1.0 - lo / hi))


def _conductor_basecolor(params, notes: list[str]):
    if "reflectance" in params:
        return spectra.param_to_rgb(params.get("reflectance")) or [0.9, 0.9, 0.9]
    # `coatedconductor` names its conductor IOR `conductor.eta`/`conductor.k`
    # (pbrt-v4). Read both spellings — material_spectral_overrides already does,
    # so reading only `eta` here would give an RGB copper base colour while the
    # spectral override named a different metal.
    eta_p = params.get("eta") or params.get("conductor.eta")
    k_p = params.get("k") or params.get("conductor.k")
    # named spectra (strings) -> tabulated metal IOR
    if eta_p is not None and eta_p.type == "spectrum" and isinstance(eta_p.values[0], str):
        refl = spectra.named_metal_reflectance_rgb(eta_p.values[0])
        if refl is not None:
            return list(refl)
        if spectra.looks_like_spectrum_file(eta_p.values[0]):
            notes.append(f"spectrum file '{eta_p.values[0]}' on conductor eta not read "
                         f"(unsupported); defaulted to copper")
        else:
            notes.append(f"named spectrum '{eta_p.values[0]}' on conductor eta "
                         f"unrecognised; defaulted to copper")
        return list(spectra.named_metal_reflectance_rgb("copper"))
    if eta_p is not None and k_p is not None:
        try:
            eta = spectra.param_to_rgb(eta_p) or eta_p.floats[:3]
            k = spectra.param_to_rgb(k_p) or k_p.floats[:3]
            return list(spectra.fresnel_conductor_rgb(eta, k))
        except Exception:  # noqa: BLE001 - fall through to default
            pass
    notes.append("conductor IOR unresolved; defaulted to copper")
    return list(spectra.named_metal_reflectance_rgb("copper"))


def material_spectral_overrides(pbrt_material) -> dict:
    """Preserve a named-conductor / dispersive-glass identity for spectral mode.

    Additive to the RGB reduction (which stays unchanged): when a conductor's eta
    resolves to a named metal (``au``/``ag``/``al``/``cu``/``cuzn``/``mgo``/
    ``tio2``) this rides ``conductor_metal`` on ``skinnyOverrides``; when a
    dielectric carries a named/dispersive eta it rides ``glass_dispersion``. An
    inline ``spectrum`` on a material rides nothing — see the note below. The
    spectral GPU path binds the exact vendored
    eta/k or Cauchy curve from these keys. Returns ``{}`` for a plain-RGB
    conductor (unknown/absent named eta) or a scalar-IOR dielectric, so an
    RGB-only scene authors no new override (byte-identical import). The scalar
    ``ior`` and the RGB ``diffuseColor`` are untouched.
    """
    if pbrt_material is None:
        return {}
    mtype = pbrt_material.type
    p = pbrt_material.params
    if mtype in ("conductor", "coatedconductor"):
        # `coatedconductor` names its conductor IOR `conductor.eta` (pbrt-v4).
        eta_p = p.get("eta") or p.get("conductor.eta")
        key = spectra.named_conductor_key(eta_p)
        if key is not None:
            return {"conductor_metal": key}
    elif mtype in ("dielectric", "thindielectric"):
        key = spectra.named_glass_key(p.get("eta"))
        if key is not None:
            return {"glass_dispersion": key}
    return {}

# NOTE: an inline non-constant `spectrum` on a *material* (e.g.
# `"spectrum reflectance" [400 .1 700 .9]`) is deliberately NOT preserved here.
# Nothing consumes it: `skinnyOverrides["spectral"]` is read only by
# `usd_loader._extract_light_spd`, and only for distant *light* prims — there is no
# per-material SPD field, buffer, or shader path, so writing the payload would
# serialize data that looks like a working feature and silently isn't. Materials
# spectrally upsample from their RGB reduction instead. Wiring real per-material
# reflectance SPDs needs a loader field + packer + binding + shader change and is
# its own change (see `pbrt-named-spectra` design, Non-Goals).


def subsurface_medium_overrides(p, eta=None) -> dict:
    """Resolve pbrt `subsurface` inputs → medium-coefficient override keys
    (`subsurface_sigma_a/_s` mm⁻¹, `subsurface_g`, `subsurface_eta`) via the
    pbrt-v4 precedence (skinny.pbrt.subsurface). These ride on parameter_overrides
    for the renderer to pack into the volumetric medium.

    This is the ONE reader of a `subsurface` material's medium params. Three
    callers share it: both mappers (through :func:`resolve_material`, which passes
    the `eta` it already resolved for the `ior` lobe) and
    :func:`skinny.pbrt.media.subsurface_overrides`, which gets a raw `ParamSet`
    from the authoring pass and so passes no `eta`. `media` adds what the USD
    stage owns on top — the mm-per-scene-unit division and the `ior` carry — and
    reads no pbrt param itself, so the precedence order, the defaults and the
    named-spectrum rule cannot drift between the two modules.
    """
    from .subsurface import subsurface_coefficients, ETA_DEFAULT, SCALE_DEFAULT

    name = p.string("name", None)
    sigma_a = p.rgb("sigma_a", None)
    sigma_s = p.rgb("sigma_s", None)
    reflectance = p.rgb("reflectance", None)
    mfp = p.rgb("mfp", None)
    g_f = p.floats("g", [0.0])
    if eta is None:
        # `eta` accepts three pbrt param types — a numeric constant, a named
        # spectrum (a recognised glass resolves to its d-line index) and a texture
        # binding. `get_float_texture` is the accessor that handles all three;
        # `floats` raises on the latter two. `notes=None`: a caller that resolved
        # `eta` for the `ior` lobe reported any degradation there already, and one
        # read must not produce two notes.
        eta = get_float_texture(p, "eta", ETA_DEFAULT).const
    scale_f = p.floats("scale", [SCALE_DEFAULT])
    coeffs = subsurface_coefficients(
        name=name, sigma_a=sigma_a, sigma_s=sigma_s,
        reflectance=reflectance, mfp=mfp,
        g=float(g_f[0]) if g_f else 0.0,
        eta=float(eta),
        scale=float(scale_f[0]) if scale_f else SCALE_DEFAULT,
    )
    return {
        "subsurface_sigma_a": list(coeffs["sigma_a"]),
        "subsurface_sigma_s": list(coeffs["sigma_s"]),
        "subsurface_g": float(coeffs["g"]),
        "subsurface_eta": float(coeffs["eta"]),
    }


@dataclass
class ResolvedMaterial:
    """Target-agnostic result of interpreting a pbrt material's parameters.

    ``lobes`` is an **ordered** mapping from a small target-neutral vocabulary
    (``base_color``, ``roughness``, ``metallic``, ``ior``, ``transmission``,
    ``coat_*``, ``subsurface*``, ``emission_rgb``, the ``subsurface_sigma_*``
    override keys, …) to the resolved value: a :class:`ParamValue` for the
    texture-connectable lobes, a :class:`ResolvedRoughness` for roughness, a
    plain scalar/list otherwise. Order is load-bearing — each adapter seeds its
    own base inputs and then applies the lobes in this order, which is what
    reproduces both targets' historical input ordering (and hence their bytes).

    ``notes`` are in read order (accessor notes interleaved with branch notes)
    and ``status`` already carries the branch escalations plus the
    unresolved-texture postlude.
    """

    lobes: dict
    status: str
    notes: list[str]


def resolve_material(pbrt_material, *, emissive_rgb=None, textures=None, base_dir=None,
                     flavor: str) -> ResolvedMaterial:
    """Interpret a pbrt material's parameters — the single owner of that job.

    Which params a material type reads, with what defaults, texture promotion,
    named-spectrum substitution, the roughness calibration chain and the
    subsurface coefficient precedence all live here; :func:`map_material` and
    :func:`map_material_mtlx` are thin emit adapters over the result and read no
    ``ParamSet`` themselves.

    *flavor* (``USD`` / ``MTLX``) gates the handful of reads that are one-sided
    or divergent between the two pipelines today. It is a wart, not a design
    goal: reading a param has side effects (an unresolved texture appends a note
    and escalates EXACT→APPROX), so resolving a param the UsdPreviewSurface path
    never reads today would change its report output. Each gate is commented
    with the follow-up that removes it.
    """
    from .report import APPROX, EXACT

    notes: list[str] = []
    lobes: dict = {}
    status = EXACT
    mtype = pbrt_material.type if pbrt_material else "diffuse"
    # an emissive shape may carry no material; use an empty set so reads default
    p = pbrt_material.params if pbrt_material else ParamSet()
    mtlx = flavor == MTLX

    def reflectance(default):
        return get_spectrum_texture(
            p, "reflectance", default, textures=textures, base_dir=base_dir, notes=notes
        )

    def roughness():
        return _resolve_roughness(p, notes, textures=textures, base_dir=base_dir, flavor=flavor)

    def scalar(name, default):
        # Neither target has a texture input for these (ior / coat IOR / coat
        # roughness); a texture binding degrades to the scalar default with a
        # note worded for the target that drops it.
        pv = get_float_texture(p, name, default, textures=textures, base_dir=base_dir, notes=notes)
        if pv.is_tex:
            target = "standard_surface input" if mtlx else "USD input"
            notes.append(f"{name} texture not supported on {target}; used scalar default")
        return pv.const

    if mtype in ("", "none"):
        lobes["base_color"] = ParamValue([0.5, 0.5, 0.5])
    elif mtype == "interface":
        # pbrt null/boundary material: no BSDF lobes at all, the shape exists
        # only to bound a MediumInterface. Encode as a lobe-less flat material
        # (zero base colour; the UsdPreviewSurface adapter leaves opacity at its
        # fully-opaque default so the flat opacity=0 glass/refraction branch does
        # NOT fire). api._author_material additionally sets the
        # "volume_interface" skinnyOverrides marker so the renderer-side routing
        # predicate does not have to sniff lobe values.
        lobes["base_color"] = ParamValue([0.0, 0.0, 0.0])
        lobes["metallic"] = 0.0
        lobes["roughness"] = ResolvedRoughness(ParamValue(1.0))
        notes.append("interface -> null boundary material (no BSDF lobes); routes to volume path")
    elif mtype == "diffuse":
        lobes["base_color"] = reflectance([0.5, 0.5, 0.5])
        lobes["roughness"] = ResolvedRoughness(ParamValue(1.0))
    elif mtype == "conductor":
        lobes["metallic"] = 1.0
        base = _conductor_basecolor(p, notes)
        lobes["base_color"] = reflectance(base)
        lobes["roughness"] = roughness()
        # conductors carry a specular tint matching the base reflectance
        lobes["specular_color"] = list(base)
    elif mtype in ("dielectric", "thindielectric"):
        lobes["base_color"] = ParamValue([1.0, 1.0, 1.0])
        lobes["transmission"] = 1.0
        lobes["transmission_color"] = [1.0, 1.0, 1.0]
        lobes["ior"] = scalar("eta", 1.5)
        lobes["roughness"] = roughness()
        if mtype == "thindielectric":
            lobes["thin_walled"] = True
            status = APPROX
            notes.append(
                "thindielectric approximated as thin-walled transmissive surface" if mtlx
                else "thindielectric approximated as thin dielectric"
            )
    elif mtype == "coateddiffuse":
        lobes["base_color"] = reflectance([0.5, 0.5, 0.5])
        lobes["roughness"] = ResolvedRoughness(ParamValue(1.0))
        lobes["coat"] = 1.0
        lobes["coat_color"] = [1.0, 1.0, 1.0]
        if mtlx:
            # FLAVOUR GATE (one-sided read; follow-up: wire interface.eta on the
            # UsdPreviewSurface path too). UsdPreviewSurface has no coat IOR
            # input, so today's map_material never reads `interface.eta` — and
            # must keep not reading it, or a texture-bound one would add a note.
            lobes["coat_ior"] = scalar("interface.eta", 1.5)
        # pbrt `coateddiffuse` carries its interface (coat) roughness in the
        # top-level `roughness` param. (The earlier `interface.roughness` lookup
        # never matched, defaulting the coat roughness to 0 and diverging from
        # the UsdPreviewSurface export.) Reuse the shared calibration chain.
        lobes["coat_roughness"] = roughness()
    elif mtype == "coatedconductor":
        lobes["metallic"] = 1.0
        base = _conductor_basecolor(p, notes)
        lobes["base_color"] = reflectance(base)
        lobes["specular_color"] = list(base)
        if mtlx:
            # FLAVOUR GATE (drift 1; follow-up: make the UsdPreviewSurface path
            # read `conductor.roughness` too, with fixture regen). pbrt-v4 spells
            # the base metal's roughness `conductor.roughness`; only the mtlx
            # pipeline reads that spelling today, the USD one reads the
            # top-level `roughness` (which drives the coat) instead.
            rv = get_float_texture(p, "conductor.roughness", 0.0,
                                   textures=textures, base_dir=base_dir, notes=notes)
            if rv.is_tex:
                lobes["roughness"] = ResolvedRoughness(ParamValue(0.5, rv.tex))
                notes.append(
                    "roughness texture connected; perceptual remap not applied to texture (approx)"
                )
            else:
                remap = p.bool("remaproughness", True)
                lobes["roughness"] = ResolvedRoughness(
                    ParamValue(alpha_to_usd_roughness(pbrt_roughness_to_alpha(rv.const, remap)))
                )
        else:
            lobes["roughness"] = roughness()
        lobes["coat"] = 1.0
        lobes["coat_color"] = [1.0, 1.0, 1.0]
        if mtlx:
            # FLAVOUR GATE (one-sided read; same follow-up as coateddiffuse).
            lobes["coat_ior"] = scalar("interface.eta", 1.5)
        lobes["coat_roughness"] = scalar("interface.roughness", 0.0)
    elif mtype == "diffusetransmission":
        lobes["base_color"] = reflectance([0.25, 0.25, 0.25])
        lobes["transmission"] = 0.5
        if mtlx:
            # FLAVOUR GATE (one-sided read; follow-up: the UsdPreviewSurface path
            # drops `transmittance` entirely — it has no transmission colour
            # input — so it must not read it either).
            lobes["transmission_color"] = get_spectrum_texture(
                p, "transmittance", [0.25, 0.25, 0.25],
                textures=textures, base_dir=base_dir, notes=notes,
            ).const
        status = APPROX
        notes.append(
            "diffusetransmission approximated as base + partial transmission" if mtlx
            else "diffusetransmission approximated as diffuse + partial opacity"
        )
    elif mtype == "subsurface":
        from .subsurface import ETA_DEFAULT

        lobes["base_color"] = ParamValue([1.0, 1.0, 1.0])
        lobes["subsurface"] = 1.0
        # ONE `eta` read for the whole branch. `pack_flat_material` takes the
        # boundary IOR from the `ior` lane and never reads `subsurface_eta`, so
        # the two must hold one value — passing it down makes that structural
        # rather than a coincidence of both sites spelling the default 1.33.
        lobes["ior"] = scalar("eta", ETA_DEFAULT)
        if mtlx:
            # FLAVOUR GATE (one-sided reads; follow-up: UsdPreviewSurface has no
            # subsurface colour/radius inputs and reads neither today).
            lobes["subsurface_color"] = get_spectrum_texture(
                p, "reflectance", [1.0, 1.0, 1.0],
                textures=textures, base_dir=base_dir, notes=notes,
            ).const
            lobes["subsurface_radius"] = list(p.rgb("radius", [1.0, 1.0, 1.0]))
        # Stage-2 (pbrt-subsurface-volumetric): carry the volumetric medium
        # coefficients (σ_a, σ_s, g, eta) via the pbrt precedence, so the renderer
        # can shade a real interior random walk. The std_surface weight/color/radius
        # above stay for the preview closure + back-compat; on the
        # UsdPreviewSurface side the opacity=0 boundary remains the transitional
        # fallback. Emitted identically by both adapters.
        lobes.update(subsurface_medium_overrides(p, eta=lobes["ior"]))
        status = APPROX
        notes.append(
            "subsurface -> standard_surface subsurface + volumetric medium coeffs (σ_a/σ_s/g)"
            if mtlx else
            "subsurface -> dielectric boundary + volumetric medium coeffs (σ_a/σ_s/g)"
        )
    else:
        status = APPROX
        notes.append(f"unknown material '{mtype}' best-effort as diffuse grey")

    if emissive_rgb is not None:
        lobes["emission_rgb"] = list(emissive_rgb)

    # an unresolved/unsupported texture binding degrades to its scalar/rgb default
    # (handled in the accessors) -> surface as APPROX.
    if status == EXACT and any("unresolved/unsupported" in n for n in notes):
        status = APPROX

    return ResolvedMaterial(lobes, status, notes)


#: neutral lobe -> UsdPreviewSurface input, for the lobes that are a plain
#: rename. Lobes absent from this map and not special-cased in the adapter are
#: dropped: UsdPreviewSurface is the subset target (no coat colour/IOR, no
#: specular tint, no transmission colour, no subsurface colour/radius).
_USD_LOBES = {
    "base_color": "diffuseColor",
    "metallic": "metallic",
    "ior": "ior",
    "coat": "clearcoat",
    "subsurface_sigma_a": "subsurface_sigma_a",
    "subsurface_sigma_s": "subsurface_sigma_s",
    "subsurface_g": "subsurface_g",
    "subsurface_eta": "subsurface_eta",
}

#: neutral lobe -> standard_surface input, likewise.
_MTLX_LOBES = {
    "base_color": "base_color",
    "metallic": "metalness",
    "ior": "specular_IOR",
    "specular_color": "specular_color",
    "transmission": "transmission",
    "transmission_color": "transmission_color",
    "thin_walled": "thin_walled",
    "coat": "coat",
    "coat_color": "coat_color",
    "coat_ior": "coat_IOR",
    "subsurface": "subsurface",
    "subsurface_color": "subsurface_color",
    "subsurface_radius": "subsurface_radius",
    "subsurface_sigma_a": "subsurface_sigma_a",
    "subsurface_sigma_s": "subsurface_sigma_s",
    "subsurface_g": "subsurface_g",
    "subsurface_eta": "subsurface_eta",
}


def map_material_mtlx(pbrt_material, *, emissive_rgb=None, textures=None, base_dir=None):
    """Map a pbrt material to Autodesk ``standard_surface`` inputs.

    A thin emit adapter over :func:`resolve_material` (which owns every pbrt
    param read): this function only translates the resolved lobes into
    standard_surface vocabulary. Where :func:`map_material` targets the
    UsdPreviewSurface subset (base_color/roughness/metallic/opacity/ior), this
    fills the richer ``standard_surface`` slots that ``pack_std_surface_params``
    / ``_STD_SURFACE_TO_FLAT`` / ``_load_mtlx_materials`` read —
    ``transmission``/``transmission_color``, ``coat``/``coat_color``/
    ``coat_IOR``, ``subsurface``/``subsurface_color``/``subsurface_radius``,
    ``specular_anisotropy``, ``thin_walled`` — so the exported ``.mtlx`` carries
    pbrt values UsdPreviewSurface drops.

    Returns ``(inputs, tex_inputs, status, notes)`` with the **same shape** as
    :func:`map_material`: ``inputs`` are constant standard_surface inputs,
    ``tex_inputs`` maps a standard_surface input name to
    ``(image_path, color_space, value_type)`` (``value_type`` in
    {"color3f","float"}), ``status`` is one of report.EXACT/APPROX/SKIPPED.

    The roughness calibration chain is bit-for-bit identical to
    :func:`map_material` (both consume the same :class:`ResolvedRoughness`);
    anisotropic ``uroughness``/``vroughness`` map to ``specular_roughness`` +
    ``specular_anisotropy`` instead of collapsing to the geometric mean.
    """
    res = resolve_material(pbrt_material, emissive_rgb=emissive_rgb,
                           textures=textures, base_dir=base_dir, flavor=MTLX)
    inputs: dict = {
        "base_color": [0.5, 0.5, 0.5],
        "metalness": 0.0,
        "specular_roughness": 0.5,
    }
    tex_inputs: dict = {}

    def put(std_in, pv):
        """Apply a ParamValue: constant input + optional texture connection.

        ``value_type`` follows the constant's kind (rgb -> color3f, scalar ->
        float) so the connection matches map_material's tex_inputs shape.
        """
        inputs[std_in] = pv.const
        if pv.is_tex:
            path, color_space = pv.tex
            value_type = "color3f" if isinstance(pv.const, (list, tuple)) else "float"
            tex_inputs[std_in] = (path, color_space, value_type)

    for key, val in res.lobes.items():
        if key == "base_color":
            put("base_color", val)
        elif key == "roughness":
            pv, aniso = _mtlx_roughness(val)
            put("specular_roughness", pv)
            if aniso != 0.0:
                inputs["specular_anisotropy"] = aniso
        elif key == "coat_roughness":
            inputs["coat_roughness"] = _mtlx_roughness(val)[0].const \
                if isinstance(val, ResolvedRoughness) else val
        elif key == "emission_rgb":
            # standard_surface emission is weight(scalar) x emission_color; the
            # round-trip (_load_mtlx_materials) recovers emissiveColor = emission
            # * emission_color and ONLY when emission > 0. Author the unit weight
            # too, else an area-light material's radiance is dropped and the
            # scene renders black (emission_color alone never round-trips).
            inputs["emission"] = 1.0
            inputs["emission_color"] = list(val)
        else:
            std_in = _MTLX_LOBES.get(key)
            if std_in is not None:
                inputs[std_in] = val

    return inputs, tex_inputs, res.status, res.notes


def map_material(pbrt_material, *, emissive_rgb=None, textures=None, base_dir=None):
    """Map a pbrt material to UsdPreviewSurface inputs.

    A thin emit adapter over :func:`resolve_material` (which owns every pbrt
    param read): this function only translates the resolved lobes into
    UsdPreviewSurface vocabulary, collapsing what the subset target cannot
    express (anisotropy to its geometric mean, transmission to ``opacity``,
    coat colour / specular tint / subsurface colour dropped entirely).

    Returns (inputs, tex_inputs, status, notes): ``inputs`` are constant
    UsdPreviewSurface inputs, ``tex_inputs`` maps a UsdPreviewSurface input name
    to ``(image_path, color_space, value_type)`` for texture-connected inputs
    (``value_type`` in {"color3f","float"}), and ``status`` is one of
    report.EXACT/APPROX/SKIPPED.

    Every textureable parameter is resolved (in the resolver) through the
    promoting accessors (``get_float_texture``/``get_spectrum_texture``,
    mirroring pbrt's ``GetFloatTexture``/``GetSpectrumTexture``), so a
    texture-bound value yields a connection on its *own* USD input while a
    constant yields a plain input -- one uniform pass, no float-vs-texture
    branching, nothing assumed to be diffuse.
    """
    res = resolve_material(pbrt_material, emissive_rgb=emissive_rgb,
                           textures=textures, base_dir=base_dir, flavor=USD)
    inputs: dict = {
        "diffuseColor": [0.5, 0.5, 0.5],
        "metallic": 0.0,
        "roughness": 0.5,
        "opacity": 1.0,
    }
    tex_inputs: dict = {}
    notes = res.notes

    def put(usd_in, pv):
        """Apply a ParamValue: constant input + optional texture connection."""
        inputs[usd_in] = pv.const
        if pv.is_tex:
            path, color_space = pv.tex
            tex_inputs[usd_in] = (path, color_space, _USD_INPUT_KIND.get(usd_in, "float"))

    for key, val in res.lobes.items():
        if key == "base_color":
            put("diffuseColor", val)
        elif key == "roughness":
            put("roughness", _usd_roughness(val))
        elif key == "transmission":
            # UsdPreviewSurface has no transmission input: a transmissive lobe
            # opens skinny's refraction gate through opacity instead (a fully
            # transmissive dielectric is opacity 0).
            inputs["opacity"] = 1.0 - float(val)
        elif key == "subsurface":
            # the opacity=0 dielectric boundary is the transitional fallback;
            # the real transport rides the subsurface_* medium coefficients.
            inputs["opacity"] = 0.0
        elif key == "coat_roughness":
            pv = _usd_roughness(val) if isinstance(val, ResolvedRoughness) else ParamValue(val)
            inputs["clearcoatRoughness"] = pv.const
            if pv.is_tex:
                notes.append(
                    "coat roughness texture not connected on USD clearcoatRoughness; used scalar")
        elif key == "emission_rgb":
            inputs["emissiveColor"] = list(val)
        else:
            usd_in = _USD_LOBES.get(key)
            if usd_in is not None:
                inputs[usd_in] = val

    return inputs, tex_inputs, res.status, notes
