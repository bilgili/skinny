"""Tests for env-map conversion: PFM read + HDR write + infinite-light wiring (6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from skinny.pbrt import hdr
from skinny.pbrt.envmap import read_pfm


def _write_pfm(path, img):
    h, w = img.shape[:2]
    with open(path, "wb") as fh:
        fh.write(b"PF\n")
        fh.write(f"{w} {h}\n".encode())
        fh.write(b"-1.0\n")  # little-endian
        fh.write(np.flipud(img).astype("<f4").tobytes())


def test_pfm_roundtrip(tmp_path):
    img = np.random.default_rng(0).random((4, 6, 3)).astype(np.float32) * 5
    p = tmp_path / "e.pfm"
    _write_pfm(p, img)
    back = read_pfm(str(p))
    assert back.shape == (4, 6, 3)
    assert np.allclose(back, img, atol=1e-4)


def test_hdr_write_roundtrips_within_rgbe_precision(tmp_path):
    img = np.abs(np.random.default_rng(1).random((8, 8, 3))) * 3 + 0.05
    p = tmp_path / "e.hdr"
    hdr.write_hdr(str(p), img)
    # decode the flat-RGBE file back and check ~1% relative error
    raw = p.read_bytes()
    body = raw[raw.index(b"\n", raw.index(b"+X")) + 1:]
    rgbe = np.frombuffer(body, np.uint8).reshape(8, 8, 4).astype(np.float64)
    f = np.where(rgbe[..., 3:4] > 0, np.ldexp(1.0, (rgbe[..., 3:4] - (128 + 8)).astype(int)), 0.0)
    decoded = rgbe[..., :3] * f
    assert np.allclose(decoded, img, rtol=0.02, atol=0.02)


def test_infinite_pfm_env_converted_to_hdr(tmp_path):
    pytest.importorskip("pxr")
    from skinny.pbrt.api import import_pbrt

    env = np.ones((4, 8, 3), np.float32) * 0.5
    _write_pfm(tmp_path / "sky.pfm", env)
    scene = tmp_path / "s.pbrt"
    scene.write_text(
        "WorldBegin\n"
        'LightSource "infinite" "string filename" "sky.pfm"\n'
        'Shape "sphere" "float radius" 1\n'
    )
    out = tmp_path / "s.usda"
    stage, report = import_pbrt(str(scene), out=str(out))

    from pxr import UsdLux

    domes = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
    assert domes
    tex = UsdLux.DomeLight(domes[0]).GetTextureFileAttr().Get()
    assert str(tex.path).endswith(".hdr")
    assert (tmp_path / str(tex.path)).exists()  # the .hdr was written next to out
    assert any("converted to .hdr" in (e.detail or "") for e in report.entries)
