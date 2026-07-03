"""pbrt participating media + subsurface -> best-effort skinny overrides (D9).

Homogeneous media carry their coefficients onto the bound material as
``customData.skinnyOverrides`` (which the loader merges into
``Material.parameter_overrides``); skinny's volume path can consume them where
supported. Heterogeneous (grid/VDB) media are detected; ``nanovdb`` grids are
supported (imported as a UsdVol volume, see :mod:`skinny.pbrt.emit`) and carry
their own override set (:func:`heterogeneous_overrides`); pbrt's procedural
``cloud`` medium is supported analytically (:func:`cloud_overrides` — no grid
file, the renderer evaluates pbrt's fBm density in-shader, change
pbrt-cloud-procedural-medium). Other heterogeneous types (``uniformgrid``,
``rgbgrid``, …) are flagged unsupported rather than emitted as a wrong
homogeneous stand-in.
"""

from __future__ import annotations

import os

import numpy as np

from . import spectra
from . import transform as T

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


def is_supported_cloud(medium) -> bool:
    """True for pbrt's built-in procedural ``cloud`` medium with the default
    ``[0,1]³`` bounds (``p0``/``p1`` unauthored or default).

    The shader's ``MEDIUM_CLOUD`` case clips density to medium-local ``[0,1]³``
    (matching pbrt's ``SampleRay`` bounds intersection for the default cube);
    non-default authored bounds would need a separate clip volume, so they stay
    on the recorded unsupported-skip path.
    """
    if medium is None or medium.type.lower() != "cloud":
        return False
    p0 = [float(v) for v in medium.params.floats("p0", [0.0, 0.0, 0.0])]
    p1 = [float(v) for v in medium.params.floats("p1", [1.0, 1.0, 1.0])]
    return p0 == [0.0, 0.0, 0.0] and p1 == [1.0, 1.0, 1.0]


def _base_overrides(medium, scale: float = 1.0) -> dict:
    """Shared skinnyOverrides skeleton for every medium kind: the RGB-resolved
    σ_a/σ_s (× *scale*), the HG ``g``, and the medium name. The homogeneous and
    grid media fold pbrt's ``scale`` param here; the procedural cloud has no
    ``scale`` (pbrt ``CloudMedium::Create`` omits it) and passes ``scale=1.0``.
    """
    p = medium.params
    sigma_a = spectra.param_to_rgb(p.get("sigma_a")) or [1.0, 1.0, 1.0]
    sigma_s = spectra.param_to_rgb(p.get("sigma_s")) or [1.0, 1.0, 1.0]
    return {
        "pbrt_medium": medium.name,
        "volume_sigma_a": [float(c) * scale for c in sigma_a],
        "volume_sigma_s": [float(c) * scale for c in sigma_s],
        "volume_g": float(p.float("g", 0.0)),
    }


def cloud_overrides(medium) -> dict:
    """Return a skinnyOverrides dict for pbrt's procedural ``cloud`` medium.

    Extends :func:`_base_overrides` (no ``scale`` — pbrt ``CloudMedium::Create``
    has none) with the procedural density parameters (``density``, ``wispiness``
    default 1, ``frequency`` default 5) and the world→medium-local affine —
    rows 0..2 of ``ctm⁻¹ @ B`` (the inverse of the point bake ``B @ CTM``, see
    :func:`skinny.pbrt.emit.bake_world_mesh`) — the same row convention the
    nanovdb grid packs, so the renderer/shader consume both identically. pbrt
    evaluates ``Density`` in medium-local space and clips to the default
    ``[0,1]³`` bounds; for those bounds medium-local coords ≡ normalized uvw.
    """
    p = medium.params
    world_to_local = np.linalg.inv(np.asarray(medium.ctm, np.float64)) @ T.B
    return {
        **_base_overrides(medium),
        "volume_cloud": True,
        "cloud_density": float(p.float("density", 1.0)),
        "cloud_wispiness": float(p.float("wispiness", 1.0)),
        "cloud_frequency": float(p.float("frequency", 5.0)),
        "volume_world_to_uvw": [float(v) for v in world_to_local[:3, :].ravel()],
    }


def homogeneous_overrides(medium) -> dict:
    """Return a skinnyOverrides dict for a homogeneous medium."""
    return _base_overrides(medium, scale=medium.params.float("scale", 1.0))


def heterogeneous_overrides(medium, scene_dir: str | None = None) -> dict:
    """Return a skinnyOverrides dict for a supported (``nanovdb``) medium.

    Extends :func:`_base_overrides` (``scale`` folds into both σ — the grid's own
    density value is a separate multiplier the renderer applies at sample time,
    same convention as pbrt's ``GridMedium``) with the grid asset path (resolved
    against *scene_dir*, absolute POSIX) and the grid field name the
    renderer/loader reads back.
    """
    p = medium.params
    filename = p.string("filename", "")
    if scene_dir and filename and not os.path.isabs(filename):
        filename = os.path.join(scene_dir, filename)
    grid_path = os.path.abspath(filename).replace(os.sep, "/") if filename else ""
    return {
        **_base_overrides(medium, scale=p.float("scale", 1.0)),
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
