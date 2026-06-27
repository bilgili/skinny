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

The standardized battery (:class:`ImageMetrics` + :func:`compute_metrics`) is the
single entry point every harness call-site uses to report a number — error vs a
reference (MSE/RMSE/MAE/relMSE/PSNR/FLIP) plus single-image quality stats
(variance, a noise-σ estimate, and a firefly outlier fraction). No call-site
should compute "error" or "noise" with its own inline formula.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
    """Read a linear EXR to an (H, W, 3) float array. Lazy backend import.

    Prefers the OpenEXR package (imageio's auto plugin selection mis-reads EXR);
    falls back to imageio's EXR plugin only if OpenEXR is unavailable.
    """
    try:
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
    except ImportError:
        import imageio.v3 as iio

        img = iio.imread(path, extension=".exr")
        return _as_rgb(np.asarray(img, dtype=np.float64))


# ─── standardized battery ─────────────────────────────────────────────────
#
# One canonical set of metrics so every parity / regression call-site reports
# the same numbers the same way. Error metrics are computed on a pair (the
# render and a reference); single-image stats need only the render.

_LUMA_W = np.array([0.299, 0.587, 0.114])  # Rec.601, matching :func:`_to_opponent`


def luminance(img) -> np.ndarray:
    """Rec.601 luminance of an (H, W, 3) image → (H, W)."""
    return _as_rgb(img) @ _LUMA_W


def mean_ratio(img, ref) -> float:
    """Absolute mean-luminance ratio mean(L(img)) / mean(L(ref)).

    The exposure-blind gate aligns *img* to *ref* before measuring error, so it
    is insensitive to a global brightness offset. This ratio is the un-aligned
    absolute-radiance check: 1.0 is a perfect match, >1 too bright, <1 too dim.
    Returns ``inf`` if the reference has ~zero mean luminance.
    """
    ref_mean = float(np.mean(luminance(ref)))
    if abs(ref_mean) < 1e-12:
        return float("inf")
    return float(np.mean(luminance(img))) / ref_mean


def mse(a, b) -> float:
    """Mean squared error on linear RGB (no exposure alignment)."""
    a, b = _as_rgb(a), _as_rgb(b)
    return float(np.mean((a - b) ** 2))


def rmse(a, b) -> float:
    """Root mean squared error."""
    return math.sqrt(mse(a, b))


def mae(a, b) -> float:
    """Mean absolute error on linear RGB."""
    a, b = _as_rgb(a), _as_rgb(b)
    return float(np.mean(np.abs(a - b)))


def psnr(a, b, peak: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB.

    *peak* defaults to the reference (*b*) maximum — appropriate for unbounded
    linear-HDR data, where a fixed 1.0 peak is meaningless. Identical images
    return ``inf``.
    """
    a, b = _as_rgb(a), _as_rgb(b)
    e = mse(a, b)
    if e <= 0.0:
        return float("inf")
    if peak is None:
        peak = float(np.max(b))
    if peak <= 0.0:
        return float("-inf")
    return float(10.0 * math.log10((peak * peak) / e))


def variance(img) -> float:
    """Global variance of the image luminance (a coarse contrast/noise proxy)."""
    return float(np.var(luminance(img)))


def noise_sigma(img) -> float:
    """Immerkær (1996) fast noise standard-deviation estimate on luminance.

    Convolves with the Laplacian-of-Laplacian mask and scales; robust to image
    content, so it tracks Monte-Carlo noise rather than scene structure. Returns
    0 for a sub-3px image.
    """
    L = luminance(img)
    h, w = L.shape
    if h < 3 or w < 3:
        return 0.0
    # |sum of the 3x3 mask [[1,-2,1],[-2,4,-2],[1,-2,1]] over the interior|
    d = (
        L[:-2, :-2] - 2 * L[:-2, 1:-1] + L[:-2, 2:]
        - 2 * L[1:-1, :-2] + 4 * L[1:-1, 1:-1] - 2 * L[1:-1, 2:]
        + L[2:, :-2] - 2 * L[2:, 1:-1] + L[2:, 2:]
    )
    return float(np.sum(np.abs(d)) * math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)))


def firefly_fraction(img, factor: float = 8.0, floor: float = 1e-3) -> float:
    """Fraction of pixels that are firefly outliers.

    A pixel is a firefly when its luminance exceeds the median of its 3×3
    neighbourhood by more than *factor* and is above an absolute *floor* (so flat
    dark regions don't register). Pure numpy 3×3 median via reflected padding.
    """
    L = luminance(img)
    if L.shape[0] < 3 or L.shape[1] < 3:
        return 0.0
    p = np.pad(L, 1, mode="reflect")
    neigh = np.stack(
        [p[i : i + L.shape[0], j : j + L.shape[1]] for i in range(3) for j in range(3)],
        axis=0,
    )
    med = np.median(neigh, axis=0)
    flagged = (L > floor) & (L > factor * np.maximum(med, floor))
    return float(np.mean(flagged))


@dataclass
class ImageMetrics:
    """Canonical metric battery for one rendered image (vs an optional reference).

    Error fields are ``None`` when no reference is supplied; single-image quality
    stats are always populated.
    """

    # error vs reference (exposure-aligned), None without a reference
    mse: float | None
    rmse: float | None
    mae: float | None
    relmse: float | None
    psnr: float | None
    flip: float | None
    # single-image quality stats (no reference needed)
    variance: float
    noise_sigma: float
    firefly_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        def f(x):
            return "—" if x is None else f"{x:.4g}"

        return (
            f"relMSE={f(self.relmse)} PSNR={f(self.psnr)} FLIP={f(self.flip)} "
            f"MSE={f(self.mse)} MAE={f(self.mae)} | var={self.variance:.4g} "
            f"noiseσ={self.noise_sigma:.4g} fireflies={self.firefly_fraction:.4g}"
        )


def compute_metrics(img, ref=None, *, align: bool = True, relmse_eps: float = 1e-2,
                    flip_exposure: float = 0.0) -> ImageMetrics:
    """Compute the full :class:`ImageMetrics` battery for *img*.

    When *ref* is given, error metrics are computed on the exposure-aligned pair
    (``align_exposure(img, ref)`` vs ``ref``, matching the parity convention).
    When *ref* is ``None``, only the single-image quality stats are filled.
    """
    src = _as_rgb(img)
    if ref is None:
        return ImageMetrics(
            mse=None, rmse=None, mae=None, relmse=None, psnr=None, flip=None,
            variance=variance(src), noise_sigma=noise_sigma(src),
            firefly_fraction=firefly_fraction(src),
        )
    ref = _as_rgb(ref)
    a = align_exposure(src, ref) if align else src
    return ImageMetrics(
        mse=mse(a, ref), rmse=rmse(a, ref), mae=mae(a, ref),
        relmse=float(np.mean((a - ref) ** 2 / (ref**2 + relmse_eps))),
        psnr=psnr(a, ref), flip=flip(a, ref, flip_exposure),
        variance=variance(src), noise_sigma=noise_sigma(src),
        firefly_fraction=firefly_fraction(src),
    )
