"""Render-level fidelity gates for the Stage-2 rich flat-BSDF inputs
(flat-lobes-rich-inputs), tasks 4.1 + 4.2.

These prove the new inputs reach *pixels* (not just the BSDF unit math in
tests/test_flat_rich_inputs.py): a colored ``transmission_color`` tints the
glass, ``diffuse_roughness`` (Oren-Nayar) and a tinted ``specular_color`` each
move the converged image off the untinted/Lambert baseline in the expected
direction.

The reference scene is the corpus glass scene, exported through the ``-mtlx``
sidecar (which carries the rich standard_surface inputs UsdPreviewSurface
cannot). We render it, edit one rich input in the sidecar, re-render, and assert
the differential. Note: pbrt's ``dielectric`` is achromatic, so there is no
pbrt-v4 colored-glass reference for a thin-surface transmission tint — the gate
is the white-vs-tinted differential, which is the concrete assertion task 4.1
calls for ("the tint reaches pixels … differs from the white-glass render").
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parent.parent
GLASS_PBRT = ROOT / "tests" / "pbrt" / "corpus" / "glass_arealight.pbrt"


def _have_vulkan() -> bool:
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")


def _render_usda(usda: Path, spp: int = 64, w: int = 96, h: int = 96) -> np.ndarray:
    from skinny.vk_context import VulkanContext
    from skinny.renderer import Renderer

    ctx = VulkanContext(window=None, width=w, height=h)
    try:
        r = Renderer(
            vk_ctx=ctx,
            shader_dir=ROOT / "src" / "skinny" / "shaders",
            hdr_dir=ROOT / "hdrs",
            tattoo_dir=ROOT / "tattoos",
            usd_scene_path=usda,
            execution_mode="megakernel",
        )
        try:
            d = 400
            while d > 0 and (r._usd_scene is None or len(r._usd_scene.instances) < 3):
                r.update(0.025)
                d -= 1
            assert r._usd_scene is not None and len(r._usd_scene.instances) >= 3
            for _ in range(16):
                r.update(0.04)
            for i in range(spp):
                r.frame_index = 7 + i
                r.render_headless()
            arr, _ = r.read_accumulation_hdr()
            return np.ascontiguousarray(arr, dtype=np.float32)
        finally:
            r.cleanup()
    finally:
        ctx.destroy()


def _export_glass_mtlx(tmp: Path) -> tuple[Path, Path]:
    """Export the corpus glass scene through -mtlx; return (usda, mtlx sidecar)."""
    from skinny.pbrt.api import import_pbrt

    out = tmp / "out.usda"
    import_pbrt(str(GLASS_PBRT), out=str(out), materialx=True)
    mtlx = out.with_suffix(".mtlx")
    assert mtlx.exists(), "expected a -mtlx sidecar"
    return out, mtlx


def _patch_mtlx(mtlx: Path, old: str, new: str) -> None:
    txt = mtlx.read_text()
    assert old in txt, f"sidecar did not contain {old!r}"
    mtlx.write_text(txt.replace(old, new))


def _relmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a[..., :3] - b[..., :3]
    return float(np.sqrt((d ** 2).sum() / ((b[..., :3] ** 2).sum() + 1e-9)))


@needs_vulkan
@pytest.mark.skipif(not GLASS_PBRT.exists(), reason="glass corpus scene missing")
class TestColoredGlassReachesPixels:
    """Task 4.1 — a non-white transmission_color tints the transmitted radiance."""

    def test_red_transmission_tints_glass(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            usda, mtlx = _export_glass_mtlx(tmp)
            white = _render_usda(usda)
            _patch_mtlx(
                mtlx,
                'name="transmission_color" type="color3" value="1, 1, 1"',
                'name="transmission_color" type="color3" value="1, 0.2, 0.2"',
            )
            red = _render_usda(usda)

        assert np.isfinite(white).all() and np.isfinite(red).all()
        # The tint must materially change the image.
        assert _relmse(white, red) > 0.1, "transmission_color did not reach pixels"
        # Expected direction: red transmission keeps R, suppresses G and B.
        sw = white[..., :3].sum(axis=(0, 1))
        sr = red[..., :3].sum(axis=(0, 1))
        assert sr[0] > 0.9 * sw[0], "red channel should be ~preserved"
        assert sr[1] < 0.75 * sw[1], "green should be suppressed by the tint"
        assert sr[2] < 0.75 * sw[2], "blue should be suppressed by the tint"


@needs_vulkan
@pytest.mark.skipif(not GLASS_PBRT.exists(), reason="glass corpus scene missing")
class TestDiffuseRoughnessAndSpecularColor:
    """Task 4.2 — diffuse_roughness (Oren-Nayar) and specular_color each move the
    image off the Lambert / untinted baseline."""

    # The diffuse floor material in the exported sidecar.
    _FLOOR = 'name="base_color" type="color3" value="0.4, 0.45, 0.5" />\n    <input name="metalness" type="float" value="0" />\n    <input name="specular_roughness" type="float" value="1" />'

    def test_diffuse_roughness_changes_floor(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            usda, mtlx = _export_glass_mtlx(tmp)
            lambert = _render_usda(usda)
            # Add Oren-Nayar roughness to the diffuse floor.
            _patch_mtlx(
                mtlx,
                self._FLOOR,
                self._FLOOR + '\n    <input name="diffuse_roughness" type="float" value="1" />',
            )
            oren = _render_usda(usda)

        assert np.isfinite(oren).all()
        assert _relmse(lambert, oren) > 0.01, (
            "diffuse_roughness (Oren-Nayar) did not change the diffuse floor")

    def test_specular_color_tints_metal_floor(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            usda, mtlx = _export_glass_mtlx(tmp)
            # Make the floor a white-spec metal first (baseline), then tint it.
            metal_floor = (
                'name="base_color" type="color3" value="0.4, 0.45, 0.5" />\n'
                '    <input name="metalness" type="float" value="1" />\n'
                '    <input name="specular_roughness" type="float" value="0.2" />'
            )
            _patch_mtlx(mtlx, self._FLOOR, metal_floor)
            white = _render_usda(usda)
            _patch_mtlx(
                mtlx, metal_floor,
                metal_floor + '\n    <input name="specular_color" type="color3" value="1, 0.3, 0.1" />',
            )
            tinted = _render_usda(usda)

        assert np.isfinite(tinted).all()
        assert _relmse(white, tinted) > 0.01, "specular_color did not reach pixels"
        sw = white[..., :3].sum(axis=(0, 1))
        st = tinted[..., :3].sum(axis=(0, 1))
        # Green/blue specular response suppressed relative to red.
        assert st[1] < sw[1] and st[2] < sw[2]
