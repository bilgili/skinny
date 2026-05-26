"""Tessellation of USD analytic gprims (Sphere/Cube/Cylinder/Cone/Capsule/Plane)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASIC_USDA = (
    PROJECT_ROOT / "assets" / "assets-main" / "test_assets" / "MaterialXTest"
    / "basic.usda"
)


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


def _bbox(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return positions.min(axis=0), positions.max(axis=0)


@needs_usd
class TestTessellateGprim:
    def _prim(self, type_name: str, **attrs):
        from pxr import Usd
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/g", type_name)
        for k, v in attrs.items():
            prim.GetAttribute(k).Set(v)
        # return stage too so it stays alive for the prim's lifetime
        return prim, stage

    def _check_common(self, ms) -> None:
        assert ms is not None
        assert ms.positions.dtype == np.float32
        assert ms.positions.ndim == 2 and ms.positions.shape[1] == 3
        assert ms.normals.shape == ms.positions.shape
        assert ms.uvs.shape == (ms.positions.shape[0], 2)
        assert ms.tri_idx.dtype == np.int32 and ms.tri_idx.shape[1] == 3
        assert ms.positions.shape[0] >= 3
        assert ms.tri_idx.shape[0] >= 1
        assert int(ms.tri_idx.max()) < ms.positions.shape[0]
        assert int(ms.tri_idx.min()) >= 0
        lengths = np.linalg.norm(ms.normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-3)

    def test_sphere_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Sphere")
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        radii = np.linalg.norm(ms.positions, axis=1)
        assert np.allclose(radii, 1.0, atol=1e-2)

    def test_sphere_radius(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Sphere", radius=2.0)
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        assert np.allclose(lo, -2.0, atol=1e-2)
        assert np.allclose(hi, 2.0, atol=1e-2)

    def test_cube_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Cube")  # size fallback 2.0
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        assert np.allclose(lo, -1.0, atol=1e-4)
        assert np.allclose(hi, 1.0, atol=1e-4)

    def test_cube_size(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Cube", size=4.0)
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        lo, hi = _bbox(ms.positions)
        assert np.allclose(lo, -2.0, atol=1e-4)
        assert np.allclose(hi, 2.0, atol=1e-4)

    def test_cylinder_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Cylinder", radius=0.5)  # height 2, axis Z
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        assert np.allclose([lo[2], hi[2]], [-1.0, 1.0], atol=1e-2)
        assert np.allclose([lo[0], hi[0]], [-0.5, 0.5], atol=1e-2)
        assert np.allclose([lo[1], hi[1]], [-0.5, 0.5], atol=1e-2)

    def test_cylinder_axis_x(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Cylinder", radius=0.5, axis="X")
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        lo, hi = _bbox(ms.positions)
        # long axis now X (h/2 = 1.0), cross-section ±0.5 in Y/Z
        assert np.allclose([lo[0], hi[0]], [-1.0, 1.0], atol=1e-2)
        assert np.allclose([lo[1], hi[1]], [-0.5, 0.5], atol=1e-2)
        assert np.allclose([lo[2], hi[2]], [-0.5, 0.5], atol=1e-2)

    def test_cone_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Cone", radius=0.5)  # height 2, axis Z
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        assert np.allclose([lo[2], hi[2]], [-1.0, 1.0], atol=1e-2)
        assert np.allclose([lo[0], hi[0]], [-0.5, 0.5], atol=1e-2)

    def test_capsule_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Capsule")  # radius 0.5, height 1, axis Z
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        # total length = height + 2*radius = 2 -> z in [-1, 1]
        assert np.allclose([lo[2], hi[2]], [-1.0, 1.0], atol=1e-2)
        assert np.allclose([lo[0], hi[0]], [-0.5, 0.5], atol=1e-2)
        assert np.allclose([lo[1], hi[1]], [-0.5, 0.5], atol=1e-2)

    def test_plane_default(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Plane")  # width 2, length 2, axis Z (normal)
        ms = tessellate_gprim(prim, Usd.TimeCode.Default())
        self._check_common(ms)
        lo, hi = _bbox(ms.positions)
        assert np.allclose([lo[0], hi[0]], [-1.0, 1.0], atol=1e-4)
        assert np.allclose([lo[1], hi[1]], [-1.0, 1.0], atol=1e-4)
        assert np.allclose([lo[2], hi[2]], [0.0, 0.0], atol=1e-4)

    def test_unsupported_returns_none(self):
        from pxr import Usd
        from skinny.usd_gprims import tessellate_gprim
        prim, _s = self._prim("Xform")
        assert tessellate_gprim(prim, Usd.TimeCode.Default()) is None


@needs_usd
@pytest.mark.skipif(not BASIC_USDA.exists(), reason="basic.usda asset missing")
def test_load_basic_usda_with_sphere():
    from skinny.usd_loader import load_scene_from_usd
    scene = load_scene_from_usd(BASIC_USDA)
    assert len(scene.instances) == 1
