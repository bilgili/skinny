"""Renderer-side mirrored (improper) camera support.

The pbrt importer flags an orientation-reversing camera with
``customData["pbrt"]["mirrored"] = True`` (see ``pbrt/api.py``). These tests
cover the loader threading that surfaces the flag as
``CameraOverride.mirrored`` and the renderer UBO packing that carries it to the
shader (change ``pbrt-mirrored-camera-flip``).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from skinny import usd_loader

_CORPUS = os.path.join(os.path.dirname(__file__), "corpus")
_MIRRORED = os.path.join(_CORPUS, "mirrored_arealight.pbrt")


def _stage_with_camera(mirrored_md):
    """In-memory stage with one UsdGeom.Camera; optionally tag pbrt metadata."""
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/World/Cam")
    cam.CreateFocalLengthAttr(50.0)
    cam.CreateVerticalApertureAttr(24.0)
    if mirrored_md is not None:
        cam.GetPrim().SetCustomDataByKey("pbrt", mirrored_md)
    return stage


def test_extract_camera_reads_mirrored_flag():
    stage = _stage_with_camera({"type": "perspective", "mirrored": True})
    ov = usd_loader._extract_camera(stage, Usd.TimeCode.Default())
    assert ov is not None
    assert ov.mirrored is True


def test_extract_camera_defaults_mirror_false():
    # No pbrt metadata at all -> not mirrored.
    stage = _stage_with_camera(None)
    ov = usd_loader._extract_camera(stage, Usd.TimeCode.Default())
    assert ov is not None
    assert ov.mirrored is False


def test_extract_camera_pbrt_md_without_mirror_is_false():
    # pbrt metadata present but no mirror key -> not mirrored.
    stage = _stage_with_camera({"type": "perspective"})
    ov = usd_loader._extract_camera(stage, Usd.TimeCode.Default())
    assert ov is not None
    assert ov.mirrored is False


def _render_toggle(mirror: bool, *, backend=None, integrator="path",
                   width=128, height=128, spp=128):
    """Render the asymmetric corpus scene with the cameraMirror flag forced on/off.

    Forces the flag directly (rather than via the scene's own improper-camera
    metadata) so both renders share the identical reconstructed camera basis —
    the only difference is the ndc.x flip — making the A/B a pure mirror test.
    """
    from skinny.backend_select import select_backend
    from skinny.headless import HeadlessRenderer, RenderOptions
    from skinny.pbrt.api import import_pbrt

    backend = backend or select_backend()
    with tempfile.TemporaryDirectory() as tmp:
        usd = os.path.join(tmp, "scene.usda")
        import_pbrt(_MIRRORED, out=usd)
        with HeadlessRenderer(width, height, backend=backend) as r:
            r._prepare(usd, RenderOptions(samples=spp, integrator=integrator))
            r.renderer.direct_light_index = 1   # drop skinny's default key light
            r.renderer.env_intensity = 0.0      # no infinite light in the scene
            r.renderer._camera_mirror = bool(mirror)
            r.renderer._last_state_hash = None
            r._accumulate(spp)
            arr, _ = r.renderer.read_accumulation_hdr()
    return np.asarray(arr, dtype=np.float64)[..., :3]


def _assert_mirror_is_flip(backend=None, integrator="path"):
    from skinny.pbrt import metrics

    off = _render_toggle(False, backend=backend, integrator=integrator)
    on = _render_toggle(True, backend=backend, integrator=integrator)

    # Non-trivial guard: the un-mirrored image is itself asymmetric, so flipping
    # it changes it materially (red↔green swap).
    assert metrics.relmse(off, np.fliplr(off)) > 0.1, "scene not asymmetric enough"

    # The flag is exactly a horizontal mirror: on ≈ fliplr(off) within MC noise.
    rm = metrics.relmse(on, np.fliplr(off))
    assert rm < 0.03, f"cameraMirror is not a clean horizontal flip (relMSE {rm:.4f})"


@pytest.mark.gpu
def test_camera_mirror_flag_is_horizontal_flip():
    """cameraMirror on == column-reverse (fliplr) of cameraMirror off (host default
    backend — native Metal on Apple Silicon)."""
    _assert_mirror_is_flip()


@pytest.mark.gpu
def test_camera_mirror_flag_is_horizontal_flip_vulkan():
    """Same flip under the Vulkan/MoltenVK backend — proves both .spv paths apply
    the mirror, not just the in-process Metal compile."""
    _assert_mirror_is_flip(backend="vulkan")


@pytest.mark.gpu
def test_camera_mirror_flag_is_horizontal_flip_bdpt():
    """BDPT camera connections (sampleWi) honor the mirror too — the bidirectional
    image still equals fliplr of the un-mirrored render."""
    _assert_mirror_is_flip(integrator="bdpt")
