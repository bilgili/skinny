"""Loaders for per-model detail maps (normal / roughness / displacement).

Each map is read via Pillow, forced to RGBA8, and resized to a fixed square
resolution so one uploaded image size fits every model. 2048x2048 is a
compromise between memory footprint (16 MiB per map, 48 MiB for all three)
and the resolution needed to resolve pore-scale detail on an adult face.

Blank fallbacks are provided so the GPU descriptors always have valid data:

  * blank_normal_bytes()       — flat (0.5, 0.5, 1.0) normal, i.e. +Z in
                                  tangent space (no perturbation)
  * blank_roughness_bytes()    — mid-grey (0.5); shader falls back to the
                                  roughness slider whenever the detail-map
                                  enable flag is off, so this value is only
                                  used when an OBJ lacks a roughness image
  * blank_displacement_bytes() — mid-grey (0.5); zero offset after bias
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DETAIL_TEX_RES = 2048
_BYTES_PER_TEX = DETAIL_TEX_RES * DETAIL_TEX_RES * 4  # RGBA8


def _to_rgba_img(img: Image.Image) -> Image.Image:
    """Force to RGBA, preserving range from high-bit-depth single-channel modes.

    Pillow's plain `.convert("RGBA")` on an `I;16` image clips every value
    ≥ 255 to 255 — catastrophic for 16-bit displacement TIFs, where the mean
    raw value is typically ~28k/65k and *every* pixel ends up pinned at 255.
    Bridge such modes to 8-bit ourselves first.
    """
    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        arr8 = (np.asarray(img, dtype=np.uint16) >> 8).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA")
    if img.mode == "I":
        a = np.asarray(img, dtype=np.uint32)
        hi = max(int(a.max()), 1)
        arr8 = (a.astype(np.float32) * (255.0 / hi)).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA")
    if img.mode == "F":
        a = np.asarray(img, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        rng = max(hi - lo, 1e-8)
        arr8 = np.clip(((a - lo) / rng) * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr8, "L").convert("RGBA")
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img


def _resize_rgba(img: Image.Image) -> bytes:
    """Convert to RGBA and stretch to DETAIL_TEX_RES square. Lanczos for quality."""
    img = _to_rgba_img(img)
    if img.size != (DETAIL_TEX_RES, DETAIL_TEX_RES):
        img = img.resize((DETAIL_TEX_RES, DETAIL_TEX_RES), Image.Resampling.LANCZOS)
    return img.tobytes()


def load_texture_bytes(path: Path | None) -> bytes | None:
    """Load an image file and return RGBA8 bytes at DETAIL_TEX_RES. None on failure."""
    if path is None or not path.exists():
        return None
    # Pillow is happy with large (200MB) uncompressed TGAs — decode is single-
    # threaded but only runs on model switch, so it's fine here.
    Image.MAX_IMAGE_PIXELS = None   # some face scans trip the decompression bomb guard
    try:
        with Image.open(path) as img:
            img.load()
            return _resize_rgba(img)
    except Exception as exc:  # noqa: BLE001
        print(f"[skinny] failed to load detail map {path.name}: {exc}")
        return None


def _flat_rgba(r: int, g: int, b: int, a: int = 255) -> bytes:
    arr = np.empty((DETAIL_TEX_RES, DETAIL_TEX_RES, 4), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    arr[..., 3] = a
    return arr.tobytes()


def blank_normal_bytes() -> bytes:
    """Flat normal map: tangent-space +Z (no perturbation)."""
    return _flat_rgba(128, 128, 255)


def blank_roughness_bytes() -> bytes:
    return _flat_rgba(128, 128, 128)


def blank_displacement_bytes() -> bytes:
    return _flat_rgba(128, 128, 128)


def expected_bytes() -> int:
    return _BYTES_PER_TEX
