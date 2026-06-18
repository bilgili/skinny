"""Read environment maps for the pbrt importer (`.pfm` native, `.exr` via libs).

Used to convert a pbrt ``infinite`` light's `.exr`/`.pfm` map into the `.hdr`
that skinny's dome-light path loads. `.pfm` has a pure-Python reader (no deps);
`.exr` falls back to the lazy OpenEXR/imageio reader in :mod:`metrics`.
"""

from __future__ import annotations

import os

import numpy as np


def read_pfm(path: str) -> np.ndarray:
    """Read a Portable FloatMap to an (H, W, 3) float64 array (top-to-bottom)."""
    with open(path, "rb") as fh:
        header = fh.readline().strip()
        color = header == b"PF"
        dims = fh.readline().split()
        w, h = int(dims[0]), int(dims[1])
        scale = float(fh.readline().strip())
        endian = "<" if scale < 0 else ">"
        count = w * h * (3 if color else 1)
        data = np.frombuffer(fh.read(count * 4), dtype=endian + "f4")
    data = data.reshape(h, w, 3 if color else 1)
    data = np.flipud(data)  # PFM scanlines run bottom-to-top
    if not color:
        data = np.repeat(data, 3, axis=2)
    return data.astype(np.float64)


def load_env_image(path: str) -> np.ndarray:
    """Load an HDR environment map to (H, W, 3) float64. Dispatches by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pfm":
        return read_pfm(path)
    if ext == ".exr":
        from .metrics import read_exr

        return read_exr(path)
    raise ValueError(f"unsupported environment map format: {ext}")
