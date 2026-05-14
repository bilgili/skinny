"""Tests for headless Vulkan rendering pipeline.

Covers VulkanContext headless mode, Renderer.render_headless(),
ReadbackBuffer, and the video encoder chain.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
HEAD_DIR = PROJECT_ROOT / "heads"
TATTOO_DIR = PROJECT_ROOT / "tattoos"

pytestmark = pytest.mark.gpu


def _have_vulkan():
    try:
        import vulkan as vk  # noqa: F401
        return True
    except Exception:
        return False


def _have_pyav():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


needs_vulkan = pytest.mark.skipif(not _have_vulkan(), reason="No Vulkan runtime")
needs_pyav = pytest.mark.skipif(not _have_pyav(), reason="PyAV not installed")


# ── VulkanContext headless ──────────────────────────────────────────


@needs_vulkan
class TestVulkanContextHeadless:

    @pytest.fixture(scope="class")
    def headless_ctx(self):
        from skinny.vk_context import VulkanContext
        ctx = VulkanContext(window=None, width=64, height=64)
        yield ctx
        ctx.destroy()

    def test_no_surface(self, headless_ctx):
        assert headless_ctx.surface is None

    def test_no_swapchain(self, headless_ctx):
        assert headless_ctx.swapchain_info is None

    def test_has_compute_queue(self, headless_ctx):
        assert headless_ctx.compute_queue is not None

    def test_no_present_queue(self, headless_ctx):
        assert headless_ctx.present_queue is None

    def test_gpu_info_populated(self, headless_ctx):
        from skinny.hardware import GpuInfo
        assert isinstance(headless_ctx.gpu_info, GpuInfo)
        assert headless_ctx.gpu_info.name

    def test_device_created(self, headless_ctx):
        assert headless_ctx.device is not None

    def test_command_buffers_allocate(self, headless_ctx):
        bufs = headless_ctx.allocate_command_buffers(2)
        assert len(bufs) == 2


# ── ReadbackBuffer ──────────────────────────────────────────────────


@needs_vulkan
class TestReadbackBuffer:

    @pytest.fixture(scope="class")
    def ctx(self):
        from skinny.vk_context import VulkanContext
        ctx = VulkanContext(window=None, width=16, height=16)
        yield ctx
        ctx.destroy()

    def test_read_returns_correct_size(self, ctx):
        from skinny.vk_compute import ReadbackBuffer
        rb = ReadbackBuffer(ctx, 16, 16)
        data = rb.read()
        assert len(data) == 16 * 16 * 4
        rb.destroy()

    def test_read_returns_bytes(self, ctx):
        from skinny.vk_compute import ReadbackBuffer
        rb = ReadbackBuffer(ctx, 8, 8)
        data = rb.read()
        assert isinstance(data, bytes)
        rb.destroy()


# ── Headless renderer ───────────────────────────────────────────────


@needs_vulkan
class TestRendererHeadless:

    @pytest.fixture(scope="class")
    def renderer_and_ctx(self):
        from skinny.vk_context import VulkanContext
        from skinny.renderer import Renderer
        ctx = VulkanContext(window=None, width=64, height=64)
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=SHADER_DIR,
            hdr_dir=HDR_DIR,
            head_dir=HEAD_DIR,
            tattoo_dir=TATTOO_DIR,
        )
        yield renderer, ctx
        renderer.cleanup()
        ctx.destroy()

    def test_render_headless_returns_rgba(self, renderer_and_ctx):
        renderer, ctx = renderer_and_ctx
        renderer.update(0.016)
        raw = renderer.render_headless()
        expected = 64 * 64 * 4
        assert len(raw) == expected

    def test_render_headless_nonzero(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.update(0.016)
        raw = renderer.render_headless()
        assert any(b != 0 for b in raw), "Frame should not be all black"

    def test_accumulation_increments(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.accum_frame = 0
        renderer.update(0.016)
        renderer.render_headless()
        renderer.update(0.016)
        renderer.render_headless()
        assert renderer.accum_frame > 0

    def test_render_multiple_frames(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        for _ in range(5):
            renderer.update(0.016)
            raw = renderer.render_headless()
            assert len(raw) == 64 * 64 * 4

    def test_scene_rebuilt_each_update(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.update(0.016)
        scene1 = renderer.scene
        renderer.update(0.016)
        scene2 = renderer.scene
        assert scene1 is not scene2

    def test_direct_light_toggle(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.direct_light_index = 0
        renderer.update(0.016)
        assert renderer.scene.lights_dir, "lights_dir should be populated when direct=On"

        renderer.direct_light_index = 1
        renderer.update(0.016)
        assert not renderer.scene.lights_dir, "lights_dir should be empty when direct=Off"

        renderer.direct_light_index = 0
        renderer.update(0.016)

    def test_light_direction_from_params(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.direct_light_index = 0
        renderer.light_elevation = 45.0
        renderer.light_azimuth = 90.0
        renderer.light_intensity = 10.0
        renderer.update(0.016)
        scene = renderer.scene
        assert len(scene.lights_dir) == 1
        d = scene.lights_dir[0].direction
        assert np.linalg.norm(d) > 0.99
        r = scene.lights_dir[0].radiance
        assert np.max(r) > 0, "Radiance should be nonzero"

    def test_pack_uniforms_size(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.update(0.016)
        data = renderer._pack_uniforms()
        assert len(data) == 240, f"Expected 240 bytes, got {len(data)}"

    def test_pack_uniforms_direct_flag(self, renderer_and_ctx):
        renderer, _ = renderer_and_ctx
        renderer.direct_light_index = 0
        renderer.update(0.016)
        data = renderer._pack_uniforms()
        use_direct = struct.unpack_from("I", data, 164)[0]
        assert use_direct == 1

        renderer.direct_light_index = 1
        renderer.update(0.016)
        data = renderer._pack_uniforms()
        use_direct = struct.unpack_from("I", data, 164)[0]
        assert use_direct == 0

        renderer.direct_light_index = 0
        renderer.update(0.016)


# ── VideoEncoder ────────────────────────────────────────────────────


class TestVideoEncoderHelpers:
    """Pure-function tests for Annex B / AVCC helpers (no GPU needed)."""

    def test_split_annex_b_four_byte(self):
        from skinny.video_encoder import _split_annex_b
        data = b'\x00\x00\x00\x01\x67\x42\xc0\x1e\x00\x00\x00\x01\x68\xce\x38\x80'
        nals = _split_annex_b(data)
        assert len(nals) == 2
        assert nals[0][0] == 0x67  # SPS
        assert nals[1][0] == 0x68  # PPS

    def test_split_annex_b_three_byte(self):
        from skinny.video_encoder import _split_annex_b
        data = b'\x00\x00\x01\x67\xAA\xBB\x00\x00\x01\x68\xCC'
        nals = _split_annex_b(data)
        assert len(nals) == 2

    def test_split_annex_b_empty(self):
        from skinny.video_encoder import _split_annex_b
        assert _split_annex_b(b'') == []

    def test_annex_b_to_avcc_length_prefix(self):
        from skinny.video_encoder import _annex_b_to_avcc
        data = b'\x00\x00\x00\x01\x67\x42\xc0\x1e'
        avcc = _annex_b_to_avcc(data)
        length = struct.unpack(">I", avcc[:4])[0]
        assert length == 4  # 0x67, 0x42, 0xc0, 0x1e
        assert avcc[4] == 0x67

    def test_build_avcc_description(self):
        from skinny.video_encoder import _build_avcc_description
        sps = bytes([0x67, 0x42, 0xC0, 0x1E, 0xDA, 0x02])
        pps = bytes([0x68, 0xCE, 0x38, 0x80])
        desc = _build_avcc_description(sps, pps)
        assert desc[0] == 1  # configurationVersion
        assert desc[1] == 0x42  # AVCProfileIndication
        assert desc[2] == 0xC0  # profile_compatibility
        assert desc[3] == 0x1E  # AVCLevelIndication
        assert desc[4] == 0xFF  # lengthSizeMinusOne=3 | reserved
        assert desc[5] == 0xE1  # numSPS=1 | reserved
        sps_len = struct.unpack(">H", desc[6:8])[0]
        assert sps_len == len(sps)
        pps_offset = 8 + len(sps)
        assert desc[pps_offset] == 1  # numPPS
        pps_len = struct.unpack(">H", desc[pps_offset + 1:pps_offset + 3])[0]
        assert pps_len == len(pps)


@needs_pyav
class TestVideoEncoderH264:

    @pytest.fixture()
    def encoder(self):
        from skinny.video_encoder import VideoEncoder
        enc = VideoEncoder(64, 64, gpu_info=None, fps=10)
        yield enc
        enc.close()

    def test_encoder_opens(self, encoder):
        assert encoder.is_h264

    def test_encoder_name(self, encoder):
        assert encoder.encoder_name in ("libx264", "h264_nvenc", "h264_qsv", "h264_amf")

    def test_avcc_description_present(self, encoder):
        assert encoder.avcc_description is not None
        assert len(encoder.avcc_description) > 10

    def test_encode_single_frame(self, encoder):
        raw = bytes(64 * 64 * 4)
        results = encoder.encode_h264(raw)
        assert isinstance(results, list)

    def test_encode_produces_keyframe(self, encoder):
        raw = bytes(64 * 64 * 4)
        found_key = False
        for _ in range(40):
            results = encoder.encode_h264(raw)
            for is_key, data in results:
                if is_key:
                    found_key = True
                    assert len(data) > 0
        assert found_key, "Should produce at least one keyframe within GOP"

    def test_force_keyframe(self, encoder):
        raw = bytes(64 * 64 * 4)
        for _ in range(5):
            encoder.encode_h264(raw)
        encoder.force_keyframe()
        results = encoder.encode_h264(raw)
        if results:
            assert results[0][0], "Forced frame should be keyframe"

    def test_flush(self, encoder):
        raw = bytes(64 * 64 * 4)
        encoder.encode_h264(raw)
        flushed = encoder.flush()
        assert isinstance(flushed, list)


class TestVideoEncoderJpeg:

    def test_encode_jpeg_returns_valid(self):
        pytest.importorskip("PIL")
        from skinny.video_encoder import VideoEncoder
        enc = VideoEncoder(32, 32, gpu_info=None, fps=10)
        raw_rgba = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8).tobytes()
        jpeg = enc.encode_jpeg(raw_rgba, quality=50)
        assert jpeg[:2] == b'\xff\xd8'  # JPEG SOI marker
        assert len(jpeg) > 100
        enc.close()

    def test_encode_jpeg_solid_color(self):
        pytest.importorskip("PIL")
        from skinny.video_encoder import VideoEncoder
        enc = VideoEncoder(16, 16, gpu_info=None, fps=10)
        raw = bytes([255, 0, 0, 255] * 16 * 16)
        jpeg = enc.encode_jpeg(raw, quality=90)
        assert jpeg[:2] == b'\xff\xd8'
        enc.close()


# ── MaterialX graph demo scene ─────────────────────────────────────


DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"


@needs_vulkan
@pytest.mark.skipif(not DEMO_SCENE.exists(), reason="three_materials_demo.usda missing")
class TestMaterialXGraphDemoRender:
    """End-to-end smoke render for the marble / wood / brass scene.

    Renders a few frames headless, then asserts each sphere region has
    non-near-black pixels. Catches regressions where a graph dispatch
    silently zeros out base_color / specular_color / coat_color.
    """

    # Sphere world-space x: -2.5 / 0 / +2.5; camera at (0, 1, 7) looking
    # ~down 5°. With a 256-wide film and the demo's 35mm focal length the
    # spheres land near these pixel centroids in screen space.
    WIDTH = 256
    HEIGHT = 256
    SPHERE_CENTERS_PX = {
        "marble": (60, 140),
        "wood":   (128, 140),
        "brass":  (196, 140),
    }
    REGION_HALF = 16    # sample a 33×33 patch around each centroid
    MIN_RGB = 0.05      # require any pixel in the patch to exceed this

    @pytest.fixture(scope="class")
    def rendered(self):
        from skinny.vk_context import VulkanContext
        from skinny.renderer import Renderer
        ctx = VulkanContext(window=None, width=self.WIDTH, height=self.HEIGHT)
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=SHADER_DIR,
            hdr_dir=HDR_DIR,
            tattoo_dir=TATTOO_DIR,
            usd_scene_path=DEMO_SCENE,
        )
        # USD load is async; pump the streaming pipeline until at least
        # the three spheres have materialized as instances. Hard cap at
        # ~5 s so a hung loader doesn't block CI.
        deadline = 200  # × 0.025 s = 5 s
        while deadline > 0 and (
            renderer._usd_scene is None
            or len(renderer._usd_scene.instances) < 3
        ):
            renderer.update(0.025)
            deadline -= 1
        # A few extra frames so accumulation stabilises and the path
        # tracer's first-bounce noise averages out.
        frames: list[bytes] = []
        for _ in range(8):
            renderer.update(0.04)
            frames.append(renderer.render_headless())
        raw = frames[-1]
        try:
            yield renderer, raw
        finally:
            renderer.cleanup()
            ctx.destroy()

    def _sample_patch(self, raw: bytes, cx: int, cy: int) -> np.ndarray:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.HEIGHT, self.WIDTH, 4)
        x0 = max(0, cx - self.REGION_HALF)
        x1 = min(self.WIDTH, cx + self.REGION_HALF + 1)
        y0 = max(0, cy - self.REGION_HALF)
        y1 = min(self.HEIGHT, cy + self.REGION_HALF + 1)
        # Drop alpha, normalise to [0, 1].
        return arr[y0:y1, x0:x1, :3].astype(np.float32) / 255.0

    def test_scene_streamed(self, rendered):
        renderer, _ = rendered
        assert renderer._usd_scene is not None
        assert len(renderer._usd_scene.instances) >= 3, \
            f"expected 3 spheres, got {len(renderer._usd_scene.instances)}"

    @pytest.mark.parametrize("sphere", ["marble", "wood", "brass"])
    def test_sphere_renders_non_black(self, rendered, sphere):
        _, raw = rendered
        cx, cy = self.SPHERE_CENTERS_PX[sphere]
        patch = self._sample_patch(raw, cx, cy)
        peak = patch.max(axis=(0, 1))   # max per channel across patch
        assert peak.max() > self.MIN_RGB, (
            f"{sphere} sphere at ({cx}, {cy}) is near-black "
            f"(peak={peak.tolist()}, min channel max={peak.max():.4f}); "
            f"graph dispatch likely not producing visible colour"
        )


# ── Headless render → encode pipeline ──────────────────────────────


@needs_vulkan
class TestHeadlessEncoding:

    @pytest.fixture(scope="class")
    def renderer_ctx(self):
        from skinny.vk_context import VulkanContext
        from skinny.renderer import Renderer
        ctx = VulkanContext(window=None, width=64, height=64)
        renderer = Renderer(
            vk_ctx=ctx,
            shader_dir=SHADER_DIR,
            hdr_dir=HDR_DIR,
            head_dir=HEAD_DIR,
            tattoo_dir=TATTOO_DIR,
        )
        yield renderer, ctx
        renderer.cleanup()
        ctx.destroy()

    def test_render_to_jpeg(self, renderer_ctx):
        pytest.importorskip("PIL")
        from skinny.video_encoder import VideoEncoder
        renderer, ctx = renderer_ctx
        enc = VideoEncoder(64, 64, gpu_info=ctx.gpu_info, fps=10)
        renderer.update(0.016)
        raw = renderer.render_headless()
        jpeg = enc.encode_jpeg(raw)
        assert jpeg[:2] == b'\xff\xd8'
        assert len(jpeg) > 100
        enc.close()

    @needs_pyav
    def test_render_to_h264(self, renderer_ctx):
        from skinny.video_encoder import VideoEncoder
        renderer, ctx = renderer_ctx
        enc = VideoEncoder(64, 64, gpu_info=ctx.gpu_info, fps=10)
        if not enc.is_h264:
            pytest.skip("No H264 encoder available")
        all_packets = []
        for _ in range(5):
            renderer.update(0.016)
            raw = renderer.render_headless()
            packets = enc.encode_h264(raw)
            all_packets.extend(packets)
        assert len(all_packets) > 0, "Should produce H264 packets"
        enc.close()
