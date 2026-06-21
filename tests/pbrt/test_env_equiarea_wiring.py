"""Importer wiring: square infinite-light maps reproject; others pass through."""

from __future__ import annotations

import re

import numpy as np
import pytest

from skinny.pbrt import equiarea


def _write_pfm(path, img):
    h, w = img.shape[:2]
    with open(path, "wb") as fh:
        fh.write(b"PF\n")
        fh.write(f"{w} {h}\n".encode())
        fh.write(b"-1.0\n")
        fh.write(np.flipud(img).astype("<f4").tobytes())


def _hdr_dims(path):
    raw = path.read_bytes()
    m = re.search(rb"-Y (\d+) \+X (\d+)", raw)
    assert m, "no resolution line in hdr"
    return int(m.group(1)), int(m.group(2))  # (height, width)


def _import_env(tmp_path, img, fname="sky.pfm"):
    pytest.importorskip("pxr")
    from skinny.pbrt.api import import_pbrt

    _write_pfm(tmp_path / fname, img)
    scene = tmp_path / "s.pbrt"
    scene.write_text(
        "WorldBegin\n"
        f'LightSource "infinite" "string filename" "{fname}"\n'
        'Shape "sphere" "float radius" 1\n'
    )
    out = tmp_path / "s.usda"
    stage, report = import_pbrt(str(scene), out=str(out))
    from pxr import UsdLux

    domes = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
    tex = UsdLux.DomeLight(domes[0]).GetTextureFileAttr().Get()
    return tmp_path / str(tex.path), report


def test_square_map_reprojected_to_double_width(tmp_path):
    edge = 16
    rng = np.random.default_rng(3)
    img = (rng.random((edge, edge, 3)).astype(np.float32) + 0.05) * 4
    hdr_path, report = _import_env(tmp_path, img)
    h, w = _hdr_dims(hdr_path)
    assert (h, w) == (edge, 2 * edge)  # reprojected, not verbatim square
    assert any("equal-area" in (e.detail or "").lower() for e in report.entries)


def test_square_map_matches_resampler(tmp_path):
    edge = 16
    rng = np.random.default_rng(5)
    img = (rng.random((edge, edge, 3)).astype(np.float32) + 0.05) * 4
    hdr_path, _ = _import_env(tmp_path, img)
    expect = equiarea.equiarea_to_equirect(img.astype(np.float64), height=edge)
    # decode flat-RGBE hdr
    raw = hdr_path.read_bytes()
    body = raw[raw.index(b"\n", raw.index(b"+X")) + 1:]
    h, w = expect.shape[:2]
    rgbe = np.frombuffer(body, np.uint8).reshape(h, w, 4).astype(np.float64)
    f = np.where(rgbe[..., 3:4] > 0, np.ldexp(1.0, (rgbe[..., 3:4] - (128 + 8)).astype(int)), 0.0)
    decoded = rgbe[..., :3] * f
    assert np.allclose(decoded, expect, rtol=0.03, atol=0.03)


def test_nonsquare_map_passthrough(tmp_path):
    # already lat-long (2:1) -> stays as-is, dims unchanged
    img = (np.random.default_rng(7).random((8, 16, 3)).astype(np.float32) + 0.05) * 2
    hdr_path, report = _import_env(tmp_path, img)
    h, w = _hdr_dims(hdr_path)
    assert (h, w) == (8, 16)
    assert any("equirect" in (e.detail or "").lower() and "assumed" in (e.detail or "").lower()
               for e in report.entries)
