"""pbrt participating media + subsurface -> best-effort skinny overrides (D9).

Homogeneous media carry their coefficients onto the bound material as
``customData.skinnyOverrides`` (which the loader merges into
``Material.parameter_overrides``); skinny's volume path can consume them where
supported. Heterogeneous (grid/VDB) media are detected and flagged unsupported
rather than emitted as a wrong homogeneous stand-in.
"""

from __future__ import annotations

from . import spectra

_HETEROGENEOUS = {"uniformgrid", "grid", "nanovdb", "vdb", "cloud", "rgbgrid"}


def is_heterogeneous(medium) -> bool:
    return medium is not None and medium.type.lower() in _HETEROGENEOUS


def homogeneous_overrides(medium) -> dict:
    """Return a skinnyOverrides dict for a homogeneous medium."""
    p = medium.params
    scale = p.float("scale", 1.0)
    sigma_a = spectra.param_to_rgb(p.get("sigma_a")) or [1.0, 1.0, 1.0]
    sigma_s = spectra.param_to_rgb(p.get("sigma_s")) or [1.0, 1.0, 1.0]
    return {
        "pbrt_medium": medium.name,
        "volume_sigma_a": [c * scale for c in sigma_a],
        "volume_sigma_s": [c * scale for c in sigma_s],
        "volume_g": float(p.float("g", 0.0)),
    }


def subsurface_overrides(params) -> dict:
    """Volumetric medium coefficients for a pbrt ``subsurface`` material.

    Resolved via the pbrt-v4 precedence (:mod:`skinny.pbrt.subsurface`): explicit
    ``sigma_a``/``sigma_s`` (× ``scale``); else a named preset (``Skin1`` …); else
    ``reflectance`` + ``mfp`` diffuse-albedo inversion. Emitted onto
    ``skinnyOverrides`` so the loader merges them into
    ``Material.parameter_overrides``, where the renderer reads them to pack the
    inline homogeneous medium and route the material to MATERIAL_TYPE_SUBSURFACE
    (the interior random walk). Keys match the renderer's packer
    (``subsurface_sigma_a/_s/_g``); the boundary IOR is carried as ``ior`` —
    ``resolveMedium`` reuses the flat ``ior`` slot for the medium eta.
    """
    from .subsurface import subsurface_coefficients, ETA_DEFAULT, SCALE_DEFAULT

    g_f = params.floats("g", [0.0])
    eta_f = params.floats("eta", [ETA_DEFAULT])
    scale_f = params.floats("scale", [SCALE_DEFAULT])
    coeffs = subsurface_coefficients(
        name=params.string("name", None),
        sigma_a=params.rgb("sigma_a", None),
        sigma_s=params.rgb("sigma_s", None),
        reflectance=params.rgb("reflectance", None),
        mfp=params.rgb("mfp", None),
        g=float(g_f[0]) if g_f else 0.0,
        eta=float(eta_f[0]) if eta_f else ETA_DEFAULT,
        scale=float(scale_f[0]) if scale_f else SCALE_DEFAULT,
    )
    return {
        "subsurface_sigma_a": list(coeffs["sigma_a"]),
        "subsurface_sigma_s": list(coeffs["sigma_s"]),
        "subsurface_g": float(coeffs["g"]),
        "subsurface_eta": float(coeffs["eta"]),
        "ior": float(coeffs["eta"]),
    }
