"""pbrt participating media + subsurface -> best-effort skinny overrides (D9).

Homogeneous media carry their coefficients onto the bound material as
``customData.skinnyOverrides`` (which the loader merges into
``Material.parameter_overrides``); skinny's volume path can consume them where
supported. Heterogeneous (grid/VDB) media are detected; ``nanovdb`` grids are
supported (imported as a UsdVol volume, see :mod:`skinny.pbrt.emit`) and carry
their own override set (:func:`heterogeneous_overrides`). Other heterogeneous
types (``uniformgrid``, ``rgbgrid``, …) are flagged unsupported rather than
emitted as a wrong homogeneous stand-in.
"""

from __future__ import annotations

import os

from . import spectra

_HETEROGENEOUS = {"uniformgrid", "grid", "nanovdb", "vdb", "cloud", "rgbgrid"}

# Heterogeneous medium types this importer actually decodes to a UsdVol volume.
# Everything else in _HETEROGENEOUS keeps the "unsupported, skipped" path.
_SUPPORTED_HETEROGENEOUS = {"nanovdb"}


def is_heterogeneous(medium) -> bool:
    return medium is not None and medium.type.lower() in _HETEROGENEOUS


def is_supported_heterogeneous(medium) -> bool:
    """True for a heterogeneous medium this importer can actually emit.

    Only pbrt's ``nanovdb`` grid type (a ``.nvdb`` ``filename`` param) is
    supported today; ``uniformgrid``/``rgbgrid``/… still report as skipped via
    :func:`is_heterogeneous`.
    """
    return (
        medium is not None
        and medium.type.lower() in _SUPPORTED_HETEROGENEOUS
        and bool(medium.params.string("filename", None))
    )


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


def heterogeneous_overrides(medium, scene_dir: str | None = None) -> dict:
    """Return a skinnyOverrides dict for a supported (``nanovdb``) medium.

    Mirrors :func:`homogeneous_overrides`: ``scale`` folds into both σ (the
    grid's own density value is a separate multiplier the renderer applies at
    sample time, same convention as pbrt's ``GridMedium``). Adds the grid asset
    path (resolved against *scene_dir*, absolute POSIX) and the grid field name
    the renderer/loader reads back.
    """
    p = medium.params
    scale = p.float("scale", 1.0)
    sigma_a = spectra.param_to_rgb(p.get("sigma_a")) or [1.0, 1.0, 1.0]
    sigma_s = spectra.param_to_rgb(p.get("sigma_s")) or [1.0, 1.0, 1.0]
    filename = p.string("filename", "")
    if scene_dir and filename and not os.path.isabs(filename):
        filename = os.path.join(scene_dir, filename)
    grid_path = os.path.abspath(filename).replace(os.sep, "/") if filename else ""
    return {
        "pbrt_medium": medium.name,
        "volume_sigma_a": [c * scale for c in sigma_a],
        "volume_sigma_s": [c * scale for c in sigma_s],
        "volume_g": float(p.float("g", 0.0)),
        "volume_grid_asset": grid_path,
        "volume_grid_field": p.string("gridname", "density"),
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
    from .emit import PBRT_STAGE_METERS_PER_UNIT
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
    # pbrt media coefficients are mm⁻¹ interpreted per *scene unit* (τ = σ·L). The
    # renderer's walk computes τ = σ_packed · L_world · mm_per_unit, and the loader
    # derives mm_per_unit = metersPerUnit · 1000 = 1000 for an imported pbrt stage.
    # Pre-divide σ by that factor so σ_packed · mm_per_unit recovers the original
    # pbrt coefficients — otherwise the interior is ~1000× too dense (the sssdragon
    # renders opaque gold/brown instead of translucent). g/eta/ior are unitless.
    mm_per_unit = PBRT_STAGE_METERS_PER_UNIT * 1000.0
    return {
        "subsurface_sigma_a": [c / mm_per_unit for c in coeffs["sigma_a"]],
        "subsurface_sigma_s": [c / mm_per_unit for c in coeffs["sigma_s"]],
        "subsurface_g": float(coeffs["g"]),
        "subsurface_eta": float(coeffs["eta"]),
        "ior": float(coeffs["eta"]),
    }
