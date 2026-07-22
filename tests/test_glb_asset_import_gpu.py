"""GPU render gate for GLB-derived assets (change glb-asset-import).

Converts a small textured GLB with the in-repo converter and renders it on the
Metal backend, asserting the textured surface carries the texture's saturated
color — i.e. the texture resolves and applies end-to-end, not the flat white a
dropped binding would leave. Exact binding/UV correctness is covered hostlessly
in test_glb_asset_import.py; this is the on-GPU end-to-end lock-in.

Run (guarded, one Metal process at a time — ZERO-SWAP):
    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
        tests/test_glb_asset_import_gpu.py -m gpu -q
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


def _have_metal() -> bool:
    try:
        from skinny.backend_select import make_context  # noqa: F401
        import slangpy  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")
needs_metal = pytest.mark.skipif(not _have_metal(), reason="No Metal/slangpy runtime")

pytestmark = [needs_usd, needs_metal, pytest.mark.gpu]


def _textured_quad_glb(path: Path, rgb: tuple[int, int, int]) -> None:
    """A camera-facing unit quad with a solid strongly-colored baseColor texture."""
    import io as _io
    import json as _json
    from PIL import Image

    # Quad in the z=0 plane, facing +z, spanning [-1,1]².
    pos = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    nrm = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16)
    buf = pos.tobytes() + nrm.tobytes() + uv.tobytes() + idx.tobytes()
    while len(buf) % 4:
        buf += b"\x00"

    img_io = _io.BytesIO()
    Image.new("RGB", (8, 8), rgb).save(img_io, format="PNG")
    png = img_io.getvalue()

    off_pos, off_nrm = 0, pos.nbytes
    off_uv = off_nrm + nrm.nbytes
    off_idx = off_uv + uv.nbytes
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(buf) + len(png)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": off_pos, "byteLength": pos.nbytes},
            {"buffer": 0, "byteOffset": off_nrm, "byteLength": nrm.nbytes},
            {"buffer": 0, "byteOffset": off_uv, "byteLength": uv.nbytes},
            {"buffer": 0, "byteOffset": off_idx, "byteLength": idx.nbytes},
            {"buffer": 0, "byteOffset": len(buf), "byteLength": len(png)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-1, -1, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "images": [{"bufferView": 4, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
            "roughnessFactor": 1.0}, "doubleSided": True}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    all_buf = buf + png
    jb = _json.dumps(gltf).encode()
    while len(jb) % 4:
        jb += b" "
    while len(all_buf) % 4:
        all_buf += b"\x00"
    glb = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(jb) + 8 + len(all_buf))
    glb += struct.pack("<I4s", len(jb), b"JSON") + jb
    glb += struct.pack("<I4s", len(all_buf), b"BIN\x00") + all_buf
    path.write_bytes(glb)


def test_glb_textured_quad_renders_colored(tmp_path):
    """A red-textured quad must render red, not the flat white of a dropped
    texture."""
    from skinny.glb_import import convert_glb_to_usd
    from skinny.headless import HeadlessRenderer

    glb = tmp_path / "quad.glb"
    _textured_quad_glb(glb, (220, 30, 30))       # strong red baseColor
    usd = convert_glb_to_usd(glb, tmp_path / "out")

    with HeadlessRenderer(128, 128, backend="metal") as hr:
        img = hr.render_to_array(usd, samples=48, env_intensity=2.0)

    rgb = img[..., :3].astype(np.float32)
    # Center patch = the quad surface (env fills the corners).
    patch = rgb[48:80, 48:80].reshape(-1, 3)
    bright = patch[patch.sum(axis=1) > 60]        # ignore any unlit pixels
    assert len(bright) > 0, "quad surface not visible"
    mean = bright.mean(axis=0)
    # Red channel dominates: the baseColor texture is applied, not dropped to white.
    assert mean[0] > mean[1] * 1.5 and mean[0] > mean[2] * 1.5, (
        f"expected red-dominant surface, got mean RGB {mean.round(1)} "
        f"(flat white/gray => texture dropped)"
    )
