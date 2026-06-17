"""Image-parity metrics for the pbrt corpus (design D8).

* :func:`relmse` — relative MSE on linear data, robust to a residual global
  scale and to Monte-Carlo noise.
* :func:`flip` — a FLIP-style perceptual difference computed on identically
  tonemapped copies (a simplified opponent-colour approximation of the full
  FLIP metric; identical images score 0).
* :func:`align_exposure` — least-squares global scalar to align absolute
  exposure before relMSE.
* :func:`read_exr` — lazy EXR reader (used by the parity harness, not at import).

The math functions are pure numpy so they test without a GPU or pbrt binary.
"""

from __future__ import annotations

import numpy as np


def _as_rgb(img) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] >= 3:
        return a[..., :3]
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    return a


def relmse(a, b, eps: float = 1e-2) -> float:
    """Relative MSE: ``mean((a-b)**2 / (b**2 + eps))`` on linear RGB."""
    a = _as_rgb(a)
    b = _as_rgb(b)
    return float(np.mean((a - b) ** 2 / (b**2 + eps)))


def align_exposure(a, b):
    """Scale *a* by the least-squares scalar that best matches *b*; return scaled a."""
    a = _as_rgb(a)
    b = _as_rgb(b)
    denom = float(np.sum(a * a))
    if denom <= 0:
        return a
    s = float(np.sum(a * b)) / denom
    return a * s


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _tonemap(img, exposure: float) -> np.ndarray:
    lin = _as_rgb(img) * (2.0**exposure)
    return _srgb_encode(lin)


def _to_opponent(rgb: np.ndarray) -> np.ndarray:
    # simple luma + red-green + yellow-blue opponent channels
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    rg = r - g
    yb = 0.5 * (r + g) - b
    return np.stack([luma, rg, yb], axis=-1)


def flip(a, b, exposure: float = 0.0) -> float:
    """FLIP-style perceptual difference in [0, ~1]; 0 for identical images."""
    ta = _to_opponent(_tonemap(a, exposure))
    tb = _to_opponent(_tonemap(b, exposure))
    # weighted opponent distance, normalised to roughly [0,1]
    w = np.array([1.0, 0.5, 0.25])
    d = np.sqrt(np.sum(w * (ta - tb) ** 2, axis=-1))
    return float(np.mean(np.clip(d, 0.0, 1.0)))


def read_exr(path: str) -> np.ndarray:
    """Read a linear EXR to an (H, W, 3) float array. Lazy backend import."""
    try:
        import imageio.v3 as iio

        img = iio.imread(path)
        return _as_rgb(np.asarray(img, dtype=np.float64))
    except Exception:  # noqa: BLE001 - try OpenEXR next
        pass
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(path)
    header = exr.header()
    win = header["dataWindow"]
    w = win.max.x - win.min.x + 1
    h = win.max.y - win.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    chans = []
    for c in ("R", "G", "B"):
        raw = exr.channel(c, pt)
        chans.append(np.frombuffer(raw, dtype=np.float32).reshape(h, w))
    return np.stack(chans, axis=-1).astype(np.float64)
