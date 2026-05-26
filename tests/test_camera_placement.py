"""Up-axis correction on USD load + camera hero-angle framing."""

from __future__ import annotations

import numpy as np
import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


class TestUpAxisRotation:
    def test_y_up_returns_none(self):
        from skinny.usd_loader import _up_axis_rt
        assert _up_axis_rt("Y") is None

    def test_z_up_matrix_maps_z_to_y(self):
        from skinny.usd_loader import _up_axis_rt
        rt = _up_axis_rt("Z")
        assert rt is not None
        assert rt.shape == (3, 3)
        # Row-vector right-multiply: +Z world axis maps to +Y.
        np.testing.assert_allclose(np.array([0, 0, 1], np.float32) @ rt,
                                   np.array([0, 1, 0], np.float32), atol=1e-6)
        # +Y maps to -Z.
        np.testing.assert_allclose(np.array([0, 1, 0], np.float32) @ rt,
                                   np.array([0, 0, -1], np.float32), atol=1e-6)
        # +X unchanged.
        np.testing.assert_allclose(np.array([1, 0, 0], np.float32) @ rt,
                                   np.array([1, 0, 0], np.float32), atol=1e-6)
