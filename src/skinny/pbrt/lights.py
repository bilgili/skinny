"""pbrt light -> UsdLux mapping (design D6).

``distant`` -> DistantLight, ``point`` -> small SphereLight, ``infinite`` ->
DomeLight, ``spot`` -> best-effort SphereLight (flagged). Area lights
(``AreaLightSource`` on a shape) are handled in the shape path as an emissive
material, not here. Radiance is reduced to a (color, intensity) split that the
loader recombines as ``color * intensity * 2**exposure``.
"""

from __future__ import annotations

import os

import numpy as np
from pxr import Sdf, UsdLux

from . import spectra
from . import transform as T
from .emit import sanitize, to_gf_matrix
from .metadata import light_metadata


def _color_intensity(rgb, scale: float = 1.0):
    rgb = np.asarray(rgb, dtype=np.float64) * float(scale)
    lum = float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    if lum <= 0:
        return [0.0, 0.0, 0.0], 0.0
    return (rgb / lum).tolist(), lum


def _orient_z_to(direction) -> np.ndarray:
    """4x4 rotation whose local +Z maps to *direction* (unit)."""
    z = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(z)
    if n == 0:
        return T.identity()
    z = z / n
    up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, z
    return m


def add_light(stage, parent_path: str, light, report, asset_dir: str | None = None,
              exposure_scale: float = 1.0, base_dir: str | None = None) -> bool:
    """Author one UsdLux light. Returns True if a light was created.

    *asset_dir* (when writing to disk) is where synthesized / converted `.hdr`
    maps are written for ``infinite`` lights; *base_dir* is the scene directory
    used to resolve a referenced env map. *exposure_scale* folds the pbrt film
    imagingRatio into the emitted radiance.
    """
    p = light.params
    ltype = light.type
    scale = p.float("scale", 1.0) * exposure_scale
    name = sanitize(f"light_{ltype}_{id(light) & 0xFFFF:x}")
    path = f"{parent_path}/{name}"
    pbrt_md = light_metadata(light)

    if ltype == "distant":
        frm = np.array(p.floats("from", [0.0, 0.0, 0.0]))
        to = np.array(p.floats("to", [0.0, 0.0, 1.0]))
        dir_local = to - frm
        dir_world = light.ctm[:3, :3] @ dir_local
        toward_source = -(T.B[:3, :3] @ dir_world)
        prim = UsdLux.DistantLight.Define(stage, path)
        prim.GetPrim().SetCustomDataByKey("pbrt", pbrt_md)
        _set_color_intensity(prim, spectra.param_to_rgb(p.get("L"), illuminant=True)
                             or [1.0, 1.0, 1.0], scale)
        prim.AddTransformOp().Set(to_gf_matrix(_orient_z_to(toward_source)))
        report.exact(f"light:distant {path}")
        return True

    if ltype in ("point", "spot"):
        frm = np.array(p.floats("from", [0.0, 0.0, 0.0]))
        pos = T.B[:3, :3] @ (light.ctm[:3, :3] @ frm + light.ctm[:3, 3])
        prim = UsdLux.SphereLight.Define(stage, path)
        prim.GetPrim().SetCustomDataByKey("pbrt", pbrt_md)
        prim.CreateRadiusAttr(0.05)
        _set_color_intensity(prim, spectra.param_to_rgb(p.get("I"), illuminant=True)
                             or [1.0, 1.0, 1.0], scale)
        prim.AddTransformOp().Set(to_gf_matrix(T.translate(*pos)))
        if ltype == "spot":
            report.approx(f"light:spot {path}", "no skinny spotlight; emitted as point/sphere")
        else:
            report.approx(f"light:point {path}", "point emitted as small sphere light")
        return True

    if ltype == "infinite":
        prim = UsdLux.DomeLight.Define(stage, path)
        prim.GetPrim().SetCustomDataByKey("pbrt", pbrt_md)
        fname = p.string("filename", None)
        if fname:
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".hdr":
                prim.CreateTextureFileAttr().Set(Sdf.AssetPath(fname))
                report.exact(f"light:infinite {path}")
            elif ext in (".exr", ".pfm") and asset_dir is not None:
                converted = _convert_env_to_hdr(fname, base_dir, asset_dir, name, scale, report, path)
                prim.CreateTextureFileAttr().Set(Sdf.AssetPath(converted or fname))
            else:
                prim.CreateTextureFileAttr().Set(Sdf.AssetPath(fname))
                report.approx(f"light:infinite {path}",
                              f"env map {ext} referenced; loader expects .hdr")
        else:
            rgb = spectra.param_to_rgb(p.get("L"), illuminant=True) or [1.0, 1.0, 1.0]
            rgb = [c * scale for c in rgb]
            if asset_dir is not None:
                # skinny's dome path needs an actual .hdr asset for a uniform env
                from .hdr import write_constant_hdr

                hdr_name = sanitize(f"{name}_const") + ".hdr"
                write_constant_hdr(os.path.join(asset_dir, hdr_name), rgb)
                prim.CreateTextureFileAttr().Set(Sdf.AssetPath(hdr_name))
                report.exact(f"light:infinite {path}", "constant env baked to .hdr")
            else:
                color, intensity = _color_intensity(rgb)
                prim.CreateColorAttr(tuple(color))
                prim.CreateIntensityAttr(intensity)
                report.approx(f"light:infinite {path}", "constant infinite (no map; in-memory)")
        return True

    report.skipped(f"light:{ltype}", "unsupported light type")
    return False


def _convert_env_to_hdr(fname, base_dir, asset_dir, name, scale, report, path):
    """Resample a pbrt ``.exr``/``.pfm`` infinite-light map to an `.hdr` for skinny."""
    src = fname if os.path.isabs(fname) else os.path.join(base_dir or "", fname)
    try:
        from .envmap import load_env_image

        img = load_env_image(src)
    except Exception as exc:  # noqa: BLE001 - missing libs / unreadable map
        report.approx(f"light:infinite {path}",
                      f"could not convert env {os.path.basename(fname)}: {exc}")
        return None
    if scale != 1.0:
        img = img * scale
    from .hdr import write_hdr

    hdr_name = sanitize(f"{name}_env") + ".hdr"
    write_hdr(os.path.join(asset_dir, hdr_name), img)
    report.exact(f"light:infinite {path}",
                 f"{os.path.splitext(fname)[1]} env converted to .hdr")
    return hdr_name


def _set_color_intensity(prim, rgb, scale) -> None:
    color, intensity = _color_intensity(rgb, scale)
    prim.CreateColorAttr(tuple(color))
    prim.CreateIntensityAttr(intensity)
