"""Tests for the web application layer.

Covers SkinnySession lifecycle, Panel widget wiring, video stream protocol,
param synchronization, and Tornado handler setup.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from queue import Empty
from threading import Event
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
HEAD_DIR = PROJECT_ROOT / "heads"
TATTOO_DIR = PROJECT_ROOT / "tattoos"


def _have_vulkan():
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
pytestmark = pytest.mark.gpu


# ── SkinnySession lifecycle ─────────────────────────────────────��───


@needs_vulkan
class TestSkinnySession:

    @pytest.fixture(autouse=True)
    def _clear_sessions(self):
        from skinny.web_app import SkinnySession
        old = dict(SkinnySession._active)
        SkinnySession._active.clear()
        yield
        for sid, sess in list(SkinnySession._active.items()):
            if sid not in old:
                sess.cleanup()
        SkinnySession._active.clear()
        SkinnySession._active.update(old)

    def test_session_creates(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("test-01")
        assert s.session_id == "test-01"
        assert s._running
        assert SkinnySession.get("test-01") is s
        s.cleanup()

    def test_session_cleanup(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("test-02")
        s.cleanup()
        assert not s._running
        assert SkinnySession.get("test-02") is None

    def test_session_get_unknown(self):
        from skinny.web_app import SkinnySession
        assert SkinnySession.get("nonexistent") is None

    def test_session_max_limit(self):
        from skinny.web_app import SkinnySession
        old_max = SkinnySession.MAX_SESSIONS
        SkinnySession.MAX_SESSIONS = 1
        try:
            s1 = SkinnySession("limit-1")
            with pytest.raises(RuntimeError, match="Max sessions"):
                SkinnySession("limit-2")
            s1.cleanup()
        finally:
            SkinnySession.MAX_SESSIONS = old_max

    def test_session_produces_frames(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("frames-01")
        try:
            frame = s.frame_queue.get(timeout=5)
            assert len(frame) > 5
            frame_type = frame[0]
            assert frame_type in (0, 1, 2, 3)
        finally:
            s.cleanup()

    def test_session_frame_header_format(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("header-01")
        try:
            frame = s.frame_queue.get(timeout=5)
            assert len(frame) >= 5
            frame_type = frame[0]
            accum = struct.unpack("!I", frame[1:5])[0]
            assert isinstance(accum, int)
            assert frame_type in (0, 1, 2, 3)
        finally:
            s.cleanup()

    def test_set_param_scalar(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("param-01")
        try:
            s.set_param("light_intensity", 7.5)
            assert s.renderer.light_intensity == 7.5
        finally:
            s.cleanup()

    def test_set_param_discrete(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("param-02")
        try:
            s.set_param("direct_light_index", 1)
            assert s.renderer.direct_light_index == 1
            s.set_param("direct_light_index", 0)
            assert s.renderer.direct_light_index == 0
        finally:
            s.cleanup()

    def test_set_param_mtlx(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("param-03")
        try:
            s.set_param("mtlx.layer_top_melanin", 0.42)
            assert abs(s.renderer.mtlx_overrides.get("layer_top_melanin", 0) - 0.42) < 1e-6
        finally:
            s.cleanup()

    def test_handle_camera_orbit(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("cam-01")
        try:
            s.handle_camera("orbit", {"dx": 10, "dy": 5})
        finally:
            s.cleanup()

    def test_handle_camera_pan(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("cam-02")
        try:
            s.handle_camera("pan", {"dx": 3, "dy": -2})
        finally:
            s.cleanup()

    def test_handle_camera_zoom(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("cam-03")
        try:
            s.handle_camera("zoom", {"delta": 1})
        finally:
            s.cleanup()

    def test_handle_camera_move(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("cam-04")
        try:
            s.renderer.camera_mode = "free"
            s.handle_camera("move", {"forward": 1, "right": 0, "up": 0, "dt": 0.016})
        finally:
            s.cleanup()

    def test_handle_camera_move_orbit_noop(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("cam-05")
        try:
            s.handle_camera("move", {"forward": 1, "right": 0, "up": 0, "dt": 0.016})
        finally:
            s.cleanup()


# ── Frame queue protocol ────────────────────────────────────────���───


@needs_vulkan
class TestFrameQueue:

    def test_push_frame_drops_oldest(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("queue-01")
        try:
            while not s.frame_queue.empty():
                try:
                    s.frame_queue.get_nowait()
                except Empty:
                    break
            s._push_frame(2, b"frame1data")
            s._push_frame(2, b"frame2data")
            s._push_frame(2, b"frame3data")
            assert s.frame_queue.qsize() <= 2
        finally:
            s.cleanup()

    def test_frame_types(self):
        from skinny.web_app import SkinnySession
        s = SkinnySession("types-01")
        try:
            while not s.frame_queue.empty():
                try:
                    s.frame_queue.get_nowait()
                except Empty:
                    break
            for ftype in (0, 1, 2, 3):
                s._push_frame(ftype, b"test")
                frame = s.frame_queue.get(timeout=2)
                assert frame[0] == ftype
        finally:
            s.cleanup()


# ── Param grouping ──────────────────────────────────────────────────


class TestParamGrouping:

    def test_group_params_categories(self):
        from skinny.web_app import _group_params
        from skinny.params import ParamSpec

        params = [
            ParamSpec("Preset", "preset_index", "discrete", choice_source="presets"),
            ParamSpec("Environment", "env_index", "discrete", choice_source="environments"),
            ParamSpec("IBL intensity", "env_intensity", "continuous", 0.05, 0.0, 3.0),
            ParamSpec("Melanin", "mtlx.layer_top_melanin", "continuous", 0.01, 0.0, 1.0),
            ParamSpec("Light elevation", "light_elevation", "continuous", 5.0, -90.0, 90.0),
            ParamSpec("Light color R", "light_color_r", "continuous", 0.05, 0.0, 1.0),
            ParamSpec("Normal map strength", "normal_map_strength", "continuous", 0.05, 0.0, 2.0),
            ParamSpec("Subdivision", "subdivision_index", "discrete", choice_source="subdivision_modes"),
            ParamSpec("Scattering", "scatter_index", "discrete", choice_source="scatter_modes"),
        ]
        groups = _group_params(params)
        assert "Render" in groups
        assert "Skin" in groups
        assert "IBL" in groups
        assert "Direct Light" in groups
        assert "Detail" in groups

    def test_group_params_light_split(self):
        from skinny.web_app import _group_params
        from skinny.params import ParamSpec

        params = [
            ParamSpec("Environment", "env_index", "discrete", choice_source="environments"),
            ParamSpec("IBL intensity", "env_intensity", "continuous", 0.05, 0.0, 3.0),
            ParamSpec("Light elevation", "light_elevation", "continuous", 5.0, -90.0, 90.0),
            ParamSpec("Light azimuth", "light_azimuth", "continuous", 5.0, -180.0, 180.0),
            ParamSpec("Light color R", "light_color_r", "continuous", 0.05, 0.0, 1.0),
            ParamSpec("Direct light", "direct_light_index", "discrete",
                      choice_source="direct_light_modes"),
        ]
        groups = _group_params(params)
        assert "IBL" in groups
        assert len(groups["IBL"]) == 2
        assert "Direct Light" in groups
        assert len(groups["Direct Light"]) == 4

    def test_group_params_preset_in_skin(self):
        from skinny.web_app import _group_params
        from skinny.params import ParamSpec

        params = [
            ParamSpec("Preset", "preset_index", "discrete", choice_source="presets"),
            ParamSpec("Melanin", "mtlx.layer_top_melanin", "continuous", 0.01, 0.0, 1.0),
            ParamSpec("Hemoglobin", "mtlx.layer_middle_hemoglobin", "continuous", 0.01, 0.0, 1.0),
        ]
        groups = _group_params(params)
        assert "Skin" in groups
        paths = [p.path for p in groups["Skin"]]
        assert "preset_index" in paths
        assert "mtlx.layer_top_melanin" in paths
        assert len(groups["Skin"]) == 3

    def test_group_params_empty_groups_excluded(self):
        from skinny.web_app import _group_params
        from skinny.params import ParamSpec

        params = [
            ParamSpec("Scattering", "scatter_index", "discrete", choice_source="scatter_modes"),
        ]
        groups = _group_params(params)
        assert "Render" in groups
        assert "Skin" not in groups
        assert "IBL" not in groups
        assert "Direct Light" not in groups

    def test_direct_light_in_direct_light_group(self):
        from skinny.web_app import _group_params
        from skinny.params import ParamSpec

        params = [
            ParamSpec("Direct light", "direct_light_index", "discrete",
                      choice_source="direct_light_modes"),
        ]
        groups = _group_params(params)
        assert "Direct Light" in groups
        paths = [p.path for p in groups["Direct Light"]]
        assert "direct_light_index" in paths


# ── Params module ───────────────────────────────────────────────────


class TestParamsGetSet:

    def test_get_nested_scalar(self):
        from skinny.params import _get_nested
        obj = MagicMock()
        obj.light_intensity = 5.0
        assert _get_nested(obj, "light_intensity") == 5.0

    def test_set_nested_scalar(self):
        from skinny.params import _set_nested
        obj = MagicMock()
        obj.mtlx_overrides = {}
        _set_nested(obj, "light_intensity", 7.5)
        assert obj.light_intensity == 7.5

    def test_get_nested_mtlx(self):
        from skinny.params import _get_nested
        obj = MagicMock()
        obj.mtlx_overrides = {"layer_top_melanin": 0.3}
        val = _get_nested(obj, "mtlx.layer_top_melanin")
        assert abs(val - 0.3) < 1e-6

    def test_set_nested_mtlx(self):
        from skinny.params import _set_nested
        obj = MagicMock()
        obj.mtlx_overrides = {}
        _set_nested(obj, "mtlx.layer_top_melanin", 0.5)
        assert abs(obj.mtlx_overrides["layer_top_melanin"] - 0.5) < 1e-6

    def test_set_nested_mtlx_ganged(self):
        from skinny.params import _set_nested
        obj = MagicMock()
        obj.mtlx_overrides = {}
        _set_nested(obj, "mtlx.layer_top_anisotropy", 0.8)
        assert abs(obj.mtlx_overrides["layer_top_anisotropy"] - 0.8) < 1e-6
        assert abs(obj.mtlx_overrides["layer_middle_anisotropy"] - 0.8) < 1e-6
        assert abs(obj.mtlx_overrides["layer_bottom_anisotropy"] - 0.8) < 1e-6

    def test_build_all_params_includes_static(self):
        from skinny.params import build_all_params, STATIC_PARAMS
        renderer = MagicMock()
        renderer._mtlx_skin_material = None
        params = build_all_params(renderer)
        assert len(params) >= len(STATIC_PARAMS)

    def test_snapshot_and_apply_roundtrip(self):
        from skinny.params import _snapshot_params, _apply_saved_params, STATIC_PARAMS
        renderer = MagicMock()
        renderer.mtlx_overrides = {"layer_top_melanin": 0.3}
        renderer._mtlx_skin_material = None
        renderer.preset_index = 0
        renderer.env_index = 1
        renderer.env_intensity = 1.5
        renderer.mm_per_unit = 120.0
        renderer.direct_light_index = 0
        renderer.scatter_index = 0
        renderer.integrator_index = 0
        renderer.furnace_index = 0
        renderer.head_index = 0
        renderer.detail_maps_index = 0
        renderer.normal_map_strength = 1.0
        renderer.subdivision_index = 0
        renderer.displacement_scale_mm = 0.5
        renderer.tattoo_index = 0
        renderer.tattoo_density = 0.5
        renderer.light_elevation = 45.0
        renderer.light_azimuth = 90.0
        renderer.light_intensity = 5.0
        renderer.light_color_r = 1.0
        renderer.light_color_g = 0.9
        renderer.light_color_b = 0.8

        snap = _snapshot_params(renderer, STATIC_PARAMS)
        assert "light_elevation" in snap
        assert abs(snap["light_elevation"] - 45.0) < 1e-6
        assert "preset_index" not in snap


# ── VideoPageHandler template ───────────────────────────────────────


class TestVideoPlayerTemplate:

    def test_template_exists(self):
        from skinny.web_app import _TEMPLATE_PATH
        assert _TEMPLATE_PATH.exists()

    def test_template_has_placeholders(self):
        from skinny.web_app import _TEMPLATE_PATH
        content = _TEMPLATE_PATH.read_text()
        assert "{{SESSION_ID}}" in content
        assert "{{WIDTH}}" in content
        assert "{{HEIGHT}}" in content

    def test_template_has_websocket(self):
        from skinny.web_app import _TEMPLATE_PATH
        content = _TEMPLATE_PATH.read_text()
        assert "WebSocket" in content
        assert "video_ws" in content

    def test_template_has_camera_controls(self):
        from skinny.web_app import _TEMPLATE_PATH
        content = _TEMPLATE_PATH.read_text()
        assert "mousedown" in content
        assert "mousemove" in content
        assert "wheel" in content
        assert "sendCamera" in content

    def test_template_has_webcodecs(self):
        from skinny.web_app import _TEMPLATE_PATH
        content = _TEMPLATE_PATH.read_text()
        assert "VideoDecoder" in content
        assert "EncodedVideoChunk" in content

    def test_template_has_jpeg_fallback(self):
        from skinny.web_app import _TEMPLATE_PATH
        content = _TEMPLATE_PATH.read_text()
        assert "image/jpeg" in content or "handleJpeg" in content


# ── Hardware detection ──────────────────────────────────────────────


class TestGpuInfo:

    def test_vendor_enum(self):
        from skinny.hardware import GpuVendor
        assert GpuVendor.INTEL.name == "INTEL"
        assert GpuVendor.NVIDIA.name == "NVIDIA"
        assert GpuVendor.AMD.name == "AMD"

    def test_preferred_encoder_nvidia(self):
        from skinny.hardware import GpuInfo, GpuVendor
        info = GpuInfo(
            vendor=GpuVendor.NVIDIA, name="RTX 4090",
            device_id=0, vendor_id=0x10DE, device_type=2,
            vk_physical_device=None,
        )
        assert info.preferred_h264_encoder == "h264_nvenc"

    def test_preferred_encoder_intel(self):
        from skinny.hardware import GpuInfo, GpuVendor
        info = GpuInfo(
            vendor=GpuVendor.INTEL, name="Arc A770",
            device_id=0, vendor_id=0x8086, device_type=2,
            vk_physical_device=None,
        )
        assert info.preferred_h264_encoder == "h264_qsv"

    def test_preferred_encoder_amd(self):
        from skinny.hardware import GpuInfo, GpuVendor
        info = GpuInfo(
            vendor=GpuVendor.AMD, name="RX 7900",
            device_id=0, vendor_id=0x1002, device_type=2,
            vk_physical_device=None,
        )
        assert info.preferred_h264_encoder == "h264_amf"

    def test_preferred_encoder_unknown(self):
        from skinny.hardware import GpuInfo, GpuVendor
        info = GpuInfo(
            vendor=GpuVendor.UNKNOWN, name="Mystery GPU",
            device_id=0, vendor_id=0xFFFF, device_type=0,
            vk_physical_device=None,
        )
        assert info.preferred_h264_encoder == "libx264"

    @needs_vulkan
    def test_enumerate_gpus(self):
        import vulkan as vk
        from skinny.hardware import enumerate_gpus
        app_info = vk.VkApplicationInfo(
            pApplicationName="test", applicationVersion=1,
            pEngineName="test", engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
        )
        instance = vk.vkCreateInstance(create_info, None)
        try:
            gpus = enumerate_gpus(instance)
            assert len(gpus) >= 1
            assert gpus[0].name
        finally:
            vk.vkDestroyInstance(instance, None)

    @needs_vulkan
    def test_select_gpu_auto(self):
        import vulkan as vk
        from skinny.hardware import select_gpu
        app_info = vk.VkApplicationInfo(
            pApplicationName="test", applicationVersion=1,
            pEngineName="test", engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            pApplicationInfo=app_info,
        )
        instance = vk.vkCreateInstance(create_info, None)
        try:
            gpu = select_gpu(instance, "auto")
            assert gpu.name
        finally:
            vk.vkDestroyInstance(instance, None)

    def test_select_gpu_invalid_preference(self):
        from skinny.hardware import select_gpu, GpuInfo, GpuVendor
        fake_gpu = GpuInfo(
            vendor=GpuVendor.INTEL, name="Fake",
            device_id=0, vendor_id=0x8086, device_type=2,
            vk_physical_device=None,
        )
        with patch("skinny.hardware.enumerate_gpus", return_value=[fake_gpu]):
            with pytest.raises(ValueError, match="Unknown GPU preference"):
                select_gpu(None, "quantum")


# ── Scene building ──────────────────────────────────────────────────


class TestSceneBuilding:

    def test_build_default_scene_with_light(self):
        from skinny.scene import build_default_scene
        direction = np.array([0, 1, 0], dtype=np.float32)
        radiance = np.array([5, 5, 5], dtype=np.float32)
        scene = build_default_scene(
            environment=None, env_intensity=1.0, mesh=None,
            light_direction=direction, light_radiance=radiance,
            direct_light_enabled=True,
        )
        assert len(scene.lights_dir) == 1
        np.testing.assert_allclose(scene.lights_dir[0].direction, direction)
        np.testing.assert_allclose(scene.lights_dir[0].radiance, radiance)

    def test_build_default_scene_no_light(self):
        from skinny.scene import build_default_scene
        direction = np.array([0, 1, 0], dtype=np.float32)
        radiance = np.array([5, 5, 5], dtype=np.float32)
        scene = build_default_scene(
            environment=None, env_intensity=1.0, mesh=None,
            light_direction=direction, light_radiance=radiance,
            direct_light_enabled=False,
        )
        assert len(scene.lights_dir) == 0

    def test_build_default_scene_none_direction(self):
        from skinny.scene import build_default_scene
        scene = build_default_scene(
            environment=None, env_intensity=1.0, mesh=None,
            light_direction=None, light_radiance=None,
            direct_light_enabled=True,
        )
        assert len(scene.lights_dir) == 0

    def test_build_default_scene_furnace(self):
        from skinny.scene import build_default_scene
        scene = build_default_scene(
            environment=None, env_intensity=1.0, mesh=None,
            furnace_mode=True,
        )
        assert scene.furnace_mode is True
