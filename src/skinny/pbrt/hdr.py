"""Minimal Radiance ``.hdr`` writer for synthesizing constant environment maps.

pbrt's textureless ``infinite`` light is a uniform-radiance environment. skinny's
dome-light path needs an actual `.hdr` asset, so we emit a tiny constant
equirect map. Flat (non-RLE) RGBE so any Radiance reader accepts it.
"""

from __future__ import annotations

import math


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
