"""Film exposure controls and the HDR image writers — device-free.

Split out of ``renderer.py`` (change renderer-pure-core-extraction). These are
writers for render output; the readers used for scene intake live in ``pbrt/``
(``hdr.py``, ``envmap.py``) and stay there — different direction, different
consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

def _write_exr(path: str, rgb: np.ndarray) -> None:
    """Write float32 RGB to a scanline EXR via the Academy OpenEXR bindings."""
    import OpenEXR

    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {
        "R": np.ascontiguousarray(rgb[..., 0]),
        "G": np.ascontiguousarray(rgb[..., 1]),
        "B": np.ascontiguousarray(rgb[..., 2]),
    }
    OpenEXR.File(header, channels).write(path)


def _write_hdr_rgbe(path: str, rgb: np.ndarray) -> None:
    """Write float32 RGB to a Radiance .hdr (RGBE) file. No external deps."""
    rgb = np.maximum(rgb, 0.0).astype(np.float32, copy=False)
    h, w, _ = rgb.shape
    max_c = rgb.max(axis=2)
    mantissa, exponent = np.frexp(max_c)
    safe = max_c > 1e-32
    scale = np.where(safe, mantissa * 256.0 / np.where(safe, max_c, 1.0), 0.0)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(3):
        rgbe[..., i] = np.clip(
            np.round(rgb[..., i] * scale), 0.0, 255.0,
        ).astype(np.uint8)
    rgbe[..., 3] = np.where(
        safe, np.clip(exponent + 128, 0, 255), 0,
    ).astype(np.uint8)
    header = (
        b"#?RADIANCE\n"
        b"FORMAT=32-bit_rle_rgbe\n"
        b"\n"
        + f"-Y {h} +X {w}\n".encode("ascii")
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(rgbe.tobytes())


@dataclass
class FilmParameters:
    """pbrt film exposure controls, live on the renderer (change
    pbrt-radiometric-parity).

    `iso` and `exposure_time` are read from the authored camera
    (`skinny:film:iso` / `skinny:film:exposureTime`) and define the imaging
    ratio `exposure_time · iso / 100`, a global linear output scale on the
    rendered radiance (applied to the linear-HDR readback and folded into the
    display exposure). Defaults (100 / 1.0) ⇒ ratio 1.0 ⇒ a byte-identical render.
    Exposed in the UI as `film.iso` / `film.exposure_time` so they retune live.
    """

    iso: float = 100.0
    exposure_time: float = 1.0

    def imaging_ratio(self) -> float:
        return float(self.exposure_time) * float(self.iso) / 100.0
