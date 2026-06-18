"""Minimal Radiance ``.hdr`` writer for synthesizing constant environment maps.

pbrt's textureless ``infinite`` light is a uniform-radiance environment. skinny's
dome-light path needs an actual `.hdr` asset, so we emit a tiny constant
equirect map. Flat (non-RLE) RGBE so any Radiance reader accepts it.
"""

from __future__ import annotations

import math

import numpy as np


def _float_to_rgbe(r: float, g: float, b: float) -> bytes:
    m = max(r, g, b)
    if m <= 1e-32:
        return bytes((0, 0, 0, 0))
    mant, e = math.frexp(m)  # m = mant * 2**e, mant in [0.5, 1)
    scale = mant * 256.0 / m
    return bytes(
        (
            min(255, int(r * scale)),
            min(255, int(g * scale)),
            min(255, int(b * scale)),
            e + 128,
        )
    )


def write_constant_hdr(path: str, rgb, width: int = 8, height: int = 4) -> None:
    """Write a constant-colour equirectangular Radiance `.hdr` of *rgb*."""
    px = _float_to_rgbe(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        b"-Y %d +X %d\n" % (height, width)
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(px * (width * height))


def _floats_to_rgbe_array(img: np.ndarray) -> np.ndarray:
    """Vectorized linear-RGB (H,W,3) -> RGBE (H,W,4) uint8."""
    img = np.asarray(img, dtype=np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    m = np.maximum(np.maximum(r, g), b)
    mask = m > 1e-32
    safe_m = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe_m)  # m = mant * 2**exp, mant in [0.5, 1)
    scale = np.where(mask, mant * 256.0 / safe_m, 0.0)
    out = np.zeros(img.shape[:2] + (4,), dtype=np.uint8)
    out[..., 0] = np.clip(r * scale, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(g * scale, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(b * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, exp + 128, 0).astype(np.uint8)
    return out


def write_hdr(path: str, img) -> None:
    """Write a full equirectangular linear-RGB image as a flat-RGBE Radiance `.hdr`."""
    rgbe = _floats_to_rgbe_array(img)
    h, w = rgbe.shape[:2]
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        b"-Y %d +X %d\n" % (h, w)
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rgbe.tobytes())
