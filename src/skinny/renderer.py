"""Core renderer — orchestrates Vulkan compute dispatch for skin ray tracing."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vulkan as vk

from PIL import Image, ImageDraw, ImageFont

from skinny.environment import Environment, load_environments
from skinny.head_textures import (
    DETAIL_TEX_RES,
    TextureStats,
    blank_displacement_bytes,
    blank_normal_bytes,
    blank_roughness_bytes,
    compute_texture_stats,
    expected_bytes as detail_expected_bytes,
    load_texture_bytes,
)
from skinny.mesh import (
    Mesh,
    MeshSource,
    bake_mesh,
    discover_mesh_sources,
    dummy_mesh,
    pick_auto_subdivision_level,
)
from skinny.presets import PRESETS, Preset
from skinny.settings import load_user_presets
from skinny.tattoos import TATTOO_HEIGHT, TATTOO_WIDTH, Tattoo, blank_tattoo_data, load_tattoos
from skinny.vk_context import VulkanContext
from skinny.vk_compute import (
    ComputePipeline,
    HudOverlay,
    SampledImage,
    StorageBuffer,
    StorageImage,
    UniformBuffer,
)

WORKGROUP_SIZE = 8
MAX_FRAMES_IN_FLIGHT = 2


@dataclass
class SkinParameters:
    """Physically-based skin parameters.

    Layered skin model: epidermis -> dermis -> subcutaneous fat.
    Absorption and scattering coefficients are spectral (RGB approximation).
    """

    # Epidermis
    melanin_fraction: float = 0.15
    epidermis_thickness_mm: float = 0.1

    # Dermis
    hemoglobin_fraction: float = 0.05
    blood_oxygenation: float = 0.75
    dermis_thickness_mm: float = 1.0

    # Subcutaneous
    subcut_thickness_mm: float = 3.0

    # Scattering
    scattering_coefficient: np.ndarray = field(
        default_factory=lambda: np.array([3.7, 4.4, 5.05], dtype=np.float32)
    )
    anisotropy_g: float = 0.8

    # Surface
    roughness: float = 0.35
    ior: float = 1.4

    # Sub-millimeter surface detail (pores + vellus hair). Defaults to 0 so
    # loading a pre-detail preset renders identically to pre-change output.
    pore_density: float = 0.0
    pore_depth: float = 0.0
    hair_density: float = 0.0
    hair_tilt: float = 0.0

    def pack(self) -> bytes:
        """Pack into std140-compatible bytes matching the Slang SkinParams struct.

        std140 layout (offsets in bytes):
          0: melaninFraction      (float)
          4: hemoglobinFraction   (float)
          8: bloodOxygenation     (float)
         12: epidermisThickness   (float)
         16: dermisThickness      (float)
         20: subcutThickness      (float)
         24: <8 bytes padding>    (align float3 to 16)
         32: scatteringCoeff      (float3, 12 bytes)
         44: anisotropy           (float, fills vec3 trailing slot)
         48: roughness            (float)
         52: ior                  (float)
         56: poreDensity          (float)
         60: poreDepth            (float)
         64: hairDensity          (float)
         68: hairTilt             (float)
         72: <8 bytes padding>    (struct rounds to 16)
        Total: 80 bytes
        """
        return struct.pack(
            "6f 2I 3f f 2f 4f 2I",
            self.melanin_fraction,
            self.hemoglobin_fraction,
            self.blood_oxygenation,
            self.epidermis_thickness_mm,
            self.dermis_thickness_mm,
            self.subcut_thickness_mm,
            0, 0,                                # 8 bytes padding
            *self.scattering_coefficient,
            self.anisotropy_g,
            self.roughness,
            self.ior,
            self.pore_density,
            self.pore_depth,
            self.hair_density,
            self.hair_tilt,
            0, 0,                                # 8 bytes struct tail padding
        )


def _perspective(fov_deg: float, aspect: float) -> np.ndarray:
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    near, far = 0.1, 100.0
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = far / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (near * far) / (near - far)
    return proj


def _look_at(pos: np.ndarray, forward: np.ndarray) -> np.ndarray:
    # Returns V^T — the math view matrix transposed. numpy .tobytes() writes
    # row-major; GLSL reads that as column-major, which transposes once more,
    # so the GPU ends up seeing V. The camera basis therefore lives in the
    # columns of this numpy array, not the rows. (Same convention as
    # _perspective above, which stores P^T for the same reason.)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / max(np.linalg.norm(right), 1e-6)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[3, 0] = -np.dot(right, pos)
    view[3, 1] = -np.dot(up, pos)
    view[3, 2] = np.dot(forward, pos)
    return view


@dataclass
class OrbitCamera:
    """Camera that rotates around a target point (default: centre of the SDF head).

    The head's y-extent is roughly [-0.94, +1.15] and its z-extent [-0.80, +0.97],
    so target=(0, 0.1, 0.05) pins the pivot to its visual centroid. If you orbit
    around the world origin instead, the head drifts noticeably around the frame
    because that origin sits near the jaw/throat rather than the head's middle.
    """

    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.1, 0.05], dtype=np.float32)
    )
    distance: float = 3.0
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0

    @property
    def position(self) -> np.ndarray:
        x = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.005
        self.pitch += dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 - delta * 0.1), 0.5, 50.0))

    def pan(self, dx: float, dy: float) -> None:
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        scale = self.distance * 0.002
        self.target = self.target + (-right * dx + up * dy) * scale

    def view_matrix(self) -> np.ndarray:
        pos = self.position
        f = self.target - pos
        return _look_at(pos, f / np.linalg.norm(f))

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect)

    def state_signature(self) -> tuple:
        return (
            "orbit",
            float(self.yaw), float(self.pitch), float(self.distance), float(self.fov),
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
        )


@dataclass
class FreeCamera:
    """FPS-style camera: WASD translates, mouse look rotates yaw/pitch."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32)
    )
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    move_speed: float = 1.5   # world units / second

    def forward(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array([
            np.sin(self.yaw) * cp,
            np.sin(self.pitch),
            -np.cos(self.yaw) * cp,
        ], dtype=np.float32)

    def _right_vec(self) -> np.ndarray:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(self.forward(), world_up)
        return r / max(np.linalg.norm(r), 1e-6)

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.005
        self.pitch -= dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def move(self, forward: float, right: float, up: float, dt: float) -> None:
        step = self.move_speed * dt
        self.position = (
            self.position
            + self.forward() * (forward * step)
            + self._right_vec() * (right * step)
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * (up * step)
        )

    def zoom(self, delta: float) -> None:
        # Scroll changes movement speed in free mode.
        self.move_speed = float(np.clip(self.move_speed * (1.0 + delta * 0.1), 0.05, 50.0))

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.forward())

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect)

    def state_signature(self) -> tuple:
        return (
            "free",
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.yaw), float(self.pitch), float(self.fov),
        )


class Renderer:
    """Sets up Vulkan resources and dispatches Slang compute shaders each frame."""

    def __init__(
        self,
        vk_ctx: VulkanContext,
        shader_dir: Path,
        hdr_dir: Path | None = None,
        head_dir: Path | None = None,
        tattoo_dir: Path | None = None,
    ) -> None:
        self.ctx = vk_ctx
        self.width = vk_ctx.width
        self.height = vk_ctx.height
        self.shader_dir = shader_dir
        self.skin = SkinParameters()

        # Biological presets (Fitzpatrick I–VI × M/F) plus any user-saved
        # presets discovered under ~/.skinny/presets/. Selecting a preset
        # pushes its values into self.skin via apply_preset() which goes
        # through the same _set_nested path as the keyboard/slider UI.
        self.presets: list[Preset] = list(PRESETS) + load_user_presets()
        self.preset_index = 0

        # Two cameras kept in parallel; `camera` returns whichever is active.
        # Orbit is the default so the head is framed and dragging rotates
        # around it; press 'C' to switch to free-fly (WASD + mouse look).
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode: str = "orbit"

        self.frame_index = 0
        self.time_elapsed = 0.0

        # HDR environment library — built-in presets + any .hdr files found
        # in hdr_dir. Switching is driven by `env_index`.
        self.environments: list[Environment] = load_environments(hdr_dir)
        self.env_index = 0
        self._last_env_index = -1

        # Progressive accumulation — running sample count across frames.
        # Reset to 0 (via reset_accumulation) whenever camera/skin/light changes.
        self.accum_frame = 0
        self._last_state_hash: int | None = None

        # HUD overlay
        self.show_hud = True
        self.hud_text_lines: list[str] = []  # set by the input layer each frame
        self._fps_smooth = 0.0
        self._hud_font = self._load_hud_font()

        # Light (spherical representation + derived direction/radiance).
        # Color channels are user-tunable so direct light can be warmed,
        # cooled, or tinted to match a gel. Defaults are the old hardcoded
        # (3.0, 2.8, 2.5) tungsten-ish ratio L2-normalised, preserving both
        # chromaticity and total radiance magnitude from before the split.
        self.light_elevation = 35.0   # degrees
        self.light_azimuth = 45.0     # degrees
        self.light_intensity = 5.0
        self.light_color_r = 0.624
        self.light_color_g = 0.583
        self.light_color_b = 0.520
        self._update_light()

        # Direct-light toggle. Exposed to the UI as a discrete choice so the
        # user can fall back to pure image-based lighting.
        self.direct_light_modes: list[str] = ["On", "Off"]
        self.direct_light_index = 0

        # Skin scattering model. Each entry is a bitmask consumed by
        # main_pass.slang (bit 0 = point-BSSRDF, bit 1 = volume march).
        # Defaults to both so the visual matches existing renders.
        self.scatter_modes: list[str] = [
            "BSSRDF + Volume",
            "BSSRDF only",
            "Volume only",
            "Off",
        ]
        self._scatter_mode_bits = [0b11, 0b01, 0b10, 0b00]
        self.scatter_index = 0

        # Sampling / integration strategy consumed by main_pass.slang.
        # MIS affects surface IBL. "Bidirectional" keeps the camera-path
        # renderer but also adds explicit light-connection samples in volume.
        # "Stored BDPT" additionally builds a light-side surface vertex and
        # connects the camera hit to it with a visibility-tested geometry term.
        self.integrator_modes: list[str] = [
            "Path tracing", "MIS", "Bidirectional", "Stored BDPT",
        ]
        self.integrator_index = 0

        # Scalar applied to every sampleEnvironment() lookup. With many HDR
        # environments the raw luminance swamps skin albedo once multiplied
        # through the SSS estimator; this lets the user rebalance direct vs.
        # indirect contribution.
        self.env_intensity = 1.0

        # Furnace / energy-conservation probe. In this mode the shader swaps
        # the head for a unit sphere, clamps the environment to white (L=1)
        # in every direction, disables analytic direct light, and paints any
        # pixel whose accumulated radiance exceeds 1.0 per channel in a loud
        # pink — so energy violations are visible by eye. Exposed as a
        # discrete UI slider (On/Off) instead of a CLI flag so it can be
        # toggled during a session without restarting.
        self.furnace_modes: list[str] = ["Off", "On"]
        self.furnace_index: int = 0

        # Scene-scale bridge between mm-valued skin params and world-unit
        # ray distances. 1 world unit = mm_per_unit millimetres. The SDF
        # Loomis head is roughly unit-scale (~2 units tall), so with a ~240
        # mm real head height the default is 120. Exposed as a slider so
        # mesh heads of other sizes can be dialled in without editing code.
        self.mm_per_unit = 120.0

        # Head-model library. Index 0 is always the analytic SDF from
        # sdf_head.slang; further entries are triangle meshes discovered in
        # `head_dir`. The GPU Gems 3 Ch.14 pipeline wants a laser-scanned
        # polygonal head — drop any OBJ into heads/ and it appears here.
        self.head_models: list[str] = ["SDF (Loomis)"]
        self._mesh_sources: list[MeshSource] = discover_mesh_sources(head_dir)
        for src in self._mesh_sources:
            self.head_models.append(f"Mesh: {src.name}")
        self.head_index = 0
        # Rebake-tracking: each (source, subdivision, displacement-scale)
        # combination produces a different GPU mesh, so we remember the
        # triple we last baked and rebuild when any of them change. -1 and
        # NaN sentinels force an initial bake on the first mesh selection.
        self._baked_source_idx: int = -1
        self._baked_subdivision: int = -1      # resolved (0/1/2), not the UI index
        self._baked_scale_mm: float = float("nan")
        self._baked_scale_world: float = float("nan")
        self._baked_mm_per_unit: float = float("nan")
        self._baked_normals: bool = False      # tracks Mesh.normals_baked
        self._baked_normal_strength: float = float("nan")   # bake-time strength
        self._dirty_since: float | None = None      # monotonic wall-clock
        # Texture bytes + stats cached per source index. Loading a 2K TIF/TGA
        # takes ~1 s; rebaking on slider drag would feel terrible without a
        # cache. The normal cache additionally feeds the normal-bake path in
        # bake_mesh; stats drive the auto-subdivision level.
        self._displacement_cache: dict[int, bytes | None] = {}
        self._normal_cache: dict[int, bytes | None] = {}
        self._source_stats: dict[int, TextureStats] = {}

        # Tattoo library — procedural presets plus any PNG/JPG in tattoo_dir.
        # Index 0 ("None") is an all-zero-alpha image so "no tattoo" is just
        # another selection, no special-casing on the GPU side.
        self.tattoos: list[Tattoo] = load_tattoos(tattoo_dir)
        self.tattoo_index = 0
        self._last_tattoo_index = -1
        self.tattoo_density = 1.0

        # Per-model detail maps (normal / roughness / displacement). When the
        # active head model has a corresponding texture file, it's uploaded
        # into these three SampledImages on mesh switch. The per-map
        # availability flags feed the UBO so the shader only reads a map
        # when it's actually meaningful — and the enable toggle below lets
        # the user fall back to the slider values at will.
        self.detail_maps_modes: list[str] = ["On", "Off"]
        self.detail_maps_index = 0          # 0 = maps on, 1 = use sliders
        self.normal_map_strength = 1.0      # multiplies tangent-space XY offset
        # Default displacement ≈ 1 mm peak so a model shipping with a disp map
        # actually gets displaced on first load. Models without a map are
        # unaffected — bake_mesh skips the offset step when bytes are absent.
        self.displacement_scale_mm = 1.0    # mm offset at (disp - 0.5); 0 = off
        self._detail_available = (False, False, False)  # (normal, rough, disp)

        # CPU mesh subdivision before displacement. Each level splits every
        # triangle into 4 via midpoint subdivision; two levels gets us ~16×
        # the original tri count, which is usually enough for displacement
        # to read as real geometry on a ~10k-tri face scan.
        #
        # "Auto" picks a level from per-texture frequency stats (see
        # head_textures.compute_texture_stats + mesh.pick_auto_subdivision_level)
        # so models with bold displacement + bumpy normals get more triangles
        # without manual tuning; flat maps stay cheap. Explicit Off/1×/2× are
        # still offered as overrides.
        self.subdivision_modes: list[str] = [
            "Auto", "Off", "1 level (4x)", "2 levels (16x)",
        ]
        self.subdivision_index = 0   # Auto by default
        self._max_subdivision_level = 2

        self._init_gpu()

    @property
    def camera(self):
        return self.orbit_camera if self.camera_mode == "orbit" else self.free_camera

    def reset_camera(self) -> None:
        """Snap both cameras back to a known-good frame on the head."""
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"

    def toggle_camera_mode(self) -> None:
        """Flip between orbit and free while preserving the current viewpoint."""
        if self.camera_mode == "orbit":
            # Orbit -> Free: match position and look direction.
            o = self.orbit_camera
            self.free_camera.position = o.position.astype(np.float32).copy()
            self.free_camera.yaw = -o.yaw
            self.free_camera.pitch = -o.pitch
            self.camera_mode = "free"
        else:
            # Free -> Orbit: pivot around the head's visual centre.
            head_centre = np.array([0.0, 0.1, 0.05], dtype=np.float32)
            pos = self.free_camera.position.astype(np.float32)
            offset = pos - head_centre
            dist = float(max(np.linalg.norm(offset), 0.5))
            self.orbit_camera.target = head_centre
            self.orbit_camera.distance = dist
            self.orbit_camera.pitch = float(np.arcsin(offset[1] / dist))
            self.orbit_camera.yaw = float(np.arctan2(offset[0], offset[2]))
            self.camera_mode = "orbit"

    def refresh_user_presets(self) -> None:
        """Re-scan ~/.skinny/presets/ and rebuild the preset list.

        Built-ins (first N entries) are preserved; user entries are replaced.
        Called after a save/delete from the Tk panel so the combobox list
        reflects on-disk reality.
        """
        self.presets = list(PRESETS) + load_user_presets()
        if self.preset_index >= len(self.presets):
            self.preset_index = 0

    def _update_light(self) -> None:
        az = np.radians(self.light_azimuth)
        el = np.radians(self.light_elevation)
        d = np.array([
            np.cos(el) * np.sin(az),
            np.sin(el),
            np.cos(el) * np.cos(az),
        ], dtype=np.float32)
        self.light_direction = d / np.linalg.norm(d)
        color = np.array(
            [self.light_color_r, self.light_color_g, self.light_color_b],
            dtype=np.float32,
        )
        self.light_radiance = color * self.light_intensity

    def _init_gpu(self) -> None:
        # Compute pipeline from Slang
        self.pipeline = ComputePipeline(
            self.ctx,
            self.shader_dir,
            entry_module="main_pass",
            entry_point="mainImage",
        )

        # Uniform buffer — FrameConstants + SkinParams + light
        self.uniform_size = 512  # generous, std140 aligned
        self.uniform_buffer = UniformBuffer(self.ctx, self.uniform_size)

        # Persistent HDR accumulation image (progressive convergence).
        self.accum_image = StorageImage(self.ctx, self.width, self.height)

        # Per-frame HUD overlay (R8 alpha mask rasterised by Pillow).
        self.hud_overlay = HudOverlay(self.ctx, self.width, self.height)

        # HDR environment texture (RGBA32F, equirectangular).
        from skinny.environment import ENV_HEIGHT, ENV_WIDTH
        self.env_image = SampledImage(self.ctx, ENV_WIDTH, ENV_HEIGHT)
        self._ensure_env_uploaded()

        # Tattoo texture (RGBA32F, spherical UV). Seeded with a blank so the
        # descriptor is valid even before the user flips off "None".
        self.tattoo_image = SampledImage(self.ctx, TATTOO_WIDTH, TATTOO_HEIGHT)
        self.tattoo_image.upload_sync(blank_tattoo_data())
        self._ensure_tattoo_uploaded()

        # Per-model detail maps — RGBA8, 2K square. Three images cover
        # normal / roughness / displacement respectively. Seeded with
        # blanks so the descriptors are valid on frame 1.
        self.normal_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
        )
        self.roughness_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
        )
        self.displacement_image = SampledImage(
            self.ctx, DETAIL_TEX_RES, DETAIL_TEX_RES,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, bytes_per_pixel=4,
        )
        self.normal_image.upload_sync(blank_normal_bytes())
        self.roughness_image.upload_sync(blank_roughness_bytes())
        self.displacement_image.upload_sync(blank_displacement_bytes())

        # Mesh storage buffers — always bound even when the SDF path is
        # active, so the shader's StructuredBuffer bindings are valid.
        # Size to the biggest mesh at the max subdivision level (each level
        # is bounded by 4× the triangles, so level 2 = 16× upper bound on
        # verts/tris/bvh nodes; add a generous per-source allowance).
        self._dummy_mesh = dummy_mesh()
        max_sub_mul = 4 ** self._max_subdivision_level   # 16 at level 2
        max_v = max(
            (src.positions.shape[0] for src in self._mesh_sources),
            default=self._dummy_mesh.num_vertices,
        ) * max_sub_mul
        max_t = max(
            (src.tri_idx.shape[0] for src in self._mesh_sources),
            default=self._dummy_mesh.num_triangles,
        ) * max_sub_mul
        # BVH node count is <= 2·tri_count with our leaf size of 4, but
        # we over-size to keep headroom — cheaper than reallocation on rebake.
        v_size = max_v * 32 + 256
        i_size = max_t * 12 + 256
        b_size = max(max_t * 32, self._dummy_mesh.num_nodes * 32) + 256
        self.vertex_buffer = StorageBuffer(self.ctx, v_size)
        self.index_buffer = StorageBuffer(self.ctx, i_size)
        self.bvh_buffer = StorageBuffer(self.ctx, b_size)
        # Upload the dummy mesh so the buffers are valid on first frame
        # even before the user picks a real mesh (or if none are present).
        self._upload_mesh(self._dummy_mesh)

        # Descriptor pool and sets
        self._create_descriptors()

        # Command buffers
        self.command_buffers = self.ctx.allocate_command_buffers(MAX_FRAMES_IN_FLIGHT)

        # Synchronisation
        swapchain_image_count = len(self.ctx.swapchain_info.images)
        self.image_available = [
            vk.vkCreateSemaphore(
                self.ctx.device, vk.VkSemaphoreCreateInfo(), None
            )
            for _ in range(MAX_FRAMES_IN_FLIGHT)
        ]
        # One render_finished semaphore per swapchain image to avoid
        # signalling a semaphore still in use by a prior present.
        self.render_finished = [
            vk.vkCreateSemaphore(
                self.ctx.device, vk.VkSemaphoreCreateInfo(), None
            )
            for _ in range(swapchain_image_count)
        ]
        self.in_flight_fences = [
            vk.vkCreateFence(
                self.ctx.device,
                vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT),
                None,
            )
            for _ in range(MAX_FRAMES_IN_FLIGHT)
        ]

        self.current_frame = 0

    def _create_descriptors(self) -> None:
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=MAX_FRAMES_IN_FLIGHT,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=MAX_FRAMES_IN_FLIGHT,
            ),
        ]
        # Storage-image descriptors per frame: swapchain + accumulation + HUD.
        pool_sizes[1] = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=MAX_FRAMES_IN_FLIGHT * 3,
        )
        pool_sizes.append(
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=MAX_FRAMES_IN_FLIGHT * 5,  # env + tattoo + n/r/d
            )
        )
        # Three storage buffers per frame: vertices, indices, BVH nodes.
        pool_sizes.append(
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=MAX_FRAMES_IN_FLIGHT * 3,
            )
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=MAX_FRAMES_IN_FLIGHT,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.ctx.device, pool_info, None
        )

        layouts = [self.pipeline.descriptor_set_layout] * MAX_FRAMES_IN_FLIGHT
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=MAX_FRAMES_IN_FLIGHT,
            pSetLayouts=layouts,
        )
        self.descriptor_sets = vk.vkAllocateDescriptorSets(
            self.ctx.device, alloc_info
        )

        # Write descriptors (UBO at binding 0, accumulation image at binding 2).
        # Binding 1 (swapchain image) is updated per-frame in render() because
        # the acquired image index changes.
        for ds in self.descriptor_sets:
            buf_info = vk.VkDescriptorBufferInfo(
                buffer=self.uniform_buffer.buffer,
                offset=0,
                range=self.uniform_size,
            )
            accum_info = vk.VkDescriptorImageInfo(
                imageView=self.accum_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            hud_info = vk.VkDescriptorImageInfo(
                imageView=self.hud_overlay.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            env_info = vk.VkDescriptorImageInfo(
                sampler=self.env_image.sampler,
                imageView=self.env_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            tattoo_info = vk.VkDescriptorImageInfo(
                sampler=self.tattoo_image.sampler,
                imageView=self.tattoo_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            normal_info = vk.VkDescriptorImageInfo(
                sampler=self.normal_image.sampler,
                imageView=self.normal_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            rough_info = vk.VkDescriptorImageInfo(
                sampler=self.roughness_image.sampler,
                imageView=self.roughness_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            disp_info = vk.VkDescriptorImageInfo(
                sampler=self.displacement_image.sampler,
                imageView=self.displacement_image.view,
                imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            vtx_info = vk.VkDescriptorBufferInfo(
                buffer=self.vertex_buffer.buffer, offset=0, range=self.vertex_buffer.size,
            )
            idx_info = vk.VkDescriptorBufferInfo(
                buffer=self.index_buffer.buffer, offset=0, range=self.index_buffer.size,
            )
            bvh_info = vk.VkDescriptorBufferInfo(
                buffer=self.bvh_buffer.buffer, offset=0, range=self.bvh_buffer.size,
            )
            writes = [
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=0,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    pBufferInfo=[buf_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=2,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[accum_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=3,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[hud_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=4,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[env_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=5,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vtx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=6,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[idx_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=7,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[bvh_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=8,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[tattoo_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=9,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[normal_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=10,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[rough_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=ds,
                    dstBinding=11,
                    dstArrayElement=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    pImageInfo=[disp_info],
                ),
            ]
            vk.vkUpdateDescriptorSets(self.ctx.device, len(writes), writes, 0, None)

    def _upload_mesh(self, mesh: Mesh) -> None:
        self.vertex_buffer.upload_sync(mesh.vertex_bytes)
        self.index_buffer.upload_sync(mesh.index_bytes)
        self.bvh_buffer.upload_sync(mesh.bvh_bytes)

    def _upload_detail_maps(self, src_idx: int | None) -> None:
        """Upload this source's detail maps (or blanks when absent).

        Caches decoded bytes for both the normal and displacement maps so the
        CPU bake (which needs both for auto-subdivision + normal-into-vertex
        baking) doesn't have to re-read TIF/TGA files on every rebake, and
        computes a one-shot TextureStats record driving Auto subdivision.
        SDF mode (src_idx=None) just restores blanks.
        """
        if src_idx is None:
            self.normal_image.upload_sync(blank_normal_bytes())
            self.roughness_image.upload_sync(blank_roughness_bytes())
            self.displacement_image.upload_sync(blank_displacement_bytes())
            self._detail_available = (False, False, False)
            return

        src = self._mesh_sources[src_idx]
        if src_idx in self._normal_cache:
            nrm = self._normal_cache[src_idx]
        else:
            nrm = load_texture_bytes(src.normal_map)
            self._normal_cache[src_idx] = nrm
        rgh = load_texture_bytes(src.roughness_map)
        if src_idx in self._displacement_cache:
            dsp = self._displacement_cache[src_idx]
        else:
            dsp = load_texture_bytes(src.displacement_map)
            self._displacement_cache[src_idx] = dsp

        if src_idx not in self._source_stats:
            self._source_stats[src_idx] = compute_texture_stats(nrm, dsp)

        self.normal_image.upload_sync(nrm if nrm is not None else blank_normal_bytes())
        self.roughness_image.upload_sync(rgh if rgh is not None else blank_roughness_bytes())
        self.displacement_image.upload_sync(dsp if dsp is not None else blank_displacement_bytes())

        self._detail_available = (nrm is not None, rgh is not None, dsp is not None)

    def _current_scale_world(self) -> float:
        """Convert the mm-valued displacement slider into world units."""
        mm_per_unit = max(float(self.mm_per_unit), 1e-6)
        return float(self.displacement_scale_mm) / mm_per_unit

    def _resolved_subdivision_level(self, src_idx: int) -> int:
        """Map the UI subdivision index to an actual level ∈ [0, max_level].

        subdivision_modes = ["Auto", "Off", "1 level (4x)", "2 levels (16x)"]
            index 0 → stats-driven (pick_auto_subdivision_level)
            index 1 → 0 (Off)
            index 2 → 1
            index 3 → 2
        """
        idx = int(self.subdivision_index)
        if idx <= 0:
            stats = self._source_stats.get(src_idx, TextureStats())
            return pick_auto_subdivision_level(
                stats.disp_activity, stats.normal_activity,
                max_level=self._max_subdivision_level,
            )
        return max(0, min(idx - 1, self._max_subdivision_level))

    def _bake_and_upload(self, src_idx: int) -> None:
        """Subdivide + displace + (optionally) bake normals + rebuild BVH."""
        src = self._mesh_sources[src_idx]
        disp_bytes = self._displacement_cache.get(src_idx)
        nrm_bytes  = self._normal_cache.get(src_idx)
        sub_levels = self._resolved_subdivision_level(src_idx)
        scale_world = self._current_scale_world()
        mesh = bake_mesh(
            src,
            subdivision_levels=sub_levels,
            displacement_bytes=disp_bytes,
            displacement_res=DETAIL_TEX_RES,
            displacement_scale_world=scale_world,
            normal_bytes=nrm_bytes,
            normal_res=DETAIL_TEX_RES,
            normal_map_strength=float(self.normal_map_strength),
        )
        self._upload_mesh(mesh)
        self._baked_source_idx      = src_idx
        self._baked_subdivision     = sub_levels
        self._baked_scale_mm        = float(self.displacement_scale_mm)
        self._baked_scale_world     = scale_world
        self._baked_mm_per_unit     = float(self.mm_per_unit)
        self._baked_normals         = bool(mesh.normals_baked)
        self._baked_normal_strength = float(self.normal_map_strength)
        self._dirty_since           = None

    def _rebake_if_needed(self, now: float) -> None:
        """Decide whether the mesh buffers need rebuilding this frame.

        Rebakes immediately on head-model or subdivision-level change, and
        after a 300 ms debounce on the displacement-scale slider (so the
        user can drag smoothly without triggering a ~second-long bake on
        every intermediate value).
        """
        self.head_index = int(np.clip(self.head_index, 0, len(self.head_models) - 1))

        # SDF mode: nothing to rebake; just clear detail maps if we were
        # previously showing a textured mesh.
        if self.head_index == 0:
            if self._baked_source_idx != -1:
                self._upload_detail_maps(None)
                self._baked_source_idx = -1
            return

        src_idx = self.head_index - 1
        if not (0 <= src_idx < len(self._mesh_sources)):
            return

        # Source change: upload this source's detail maps, then force a bake
        # with the current subdivision + scale settings.
        if src_idx != self._baked_source_idx:
            self._upload_detail_maps(src_idx)
            self._bake_and_upload(src_idx)
            return

        target_sub = self._resolved_subdivision_level(src_idx)
        target_scale_world = self._current_scale_world()

        sub_changed = target_sub != self._baked_subdivision
        scale_changed = abs(target_scale_world - self._baked_scale_world) > 1e-9
        # Strength only affects the mesh when normals were actually baked in
        # (bake_mesh skips the bake at level 0 with displacement off).
        strength_changed = (
            self._baked_normals
            and abs(float(self.normal_map_strength) - self._baked_normal_strength) > 1e-9
        )

        if sub_changed:
            # Dropdown change (or Auto resolving to a different level) —
            # commit immediately; no drag to debounce.
            self._bake_and_upload(src_idx)
            return

        if scale_changed or strength_changed:
            if self._dirty_since is None:
                self._dirty_since = now
            elif now - self._dirty_since > 0.3:
                self._bake_and_upload(src_idx)
        else:
            self._dirty_since = None

    def _ensure_tattoo_uploaded(self) -> None:
        """Re-upload the tattoo texture on selection change."""
        if self.tattoo_index == self._last_tattoo_index:
            return
        self.tattoo_index = int(np.clip(self.tattoo_index, 0, len(self.tattoos) - 1))
        self.tattoo_image.upload_sync(self.tattoos[self.tattoo_index].data)
        self._last_tattoo_index = self.tattoo_index

    def _pack_uniforms(self) -> bytes:
        aspect = self.width / self.height
        view_inv = np.linalg.inv(self.camera.view_matrix())
        proj_inv = np.linalg.inv(self.camera.projection_matrix(aspect))

        data = bytearray()
        # FrameConstants: viewInverse (mat4), projInverse (mat4), position (vec3),
        # fov, frameIndex, accumFrame, time, width, height
        data += view_inv.astype(np.float32).tobytes()       # 64 bytes
        data += proj_inv.astype(np.float32).tobytes()        # 64 bytes
        data += self.camera.position.tobytes()               # 12 bytes
        data += struct.pack("f", self.camera.fov)            # 4 bytes
        data += struct.pack("I", self.frame_index)           # 4 bytes
        data += struct.pack("I", self.accum_frame)           # 4 bytes
        data += struct.pack("f", self.time_elapsed)          # 4 bytes
        data += struct.pack("II", self.width, self.height)   # 8 bytes
        use_direct = 1 if self.direct_light_index == 0 else 0
        data += struct.pack("I", use_direct)                 # 4 bytes
        use_mesh = 1 if self.head_index > 0 else 0
        data += struct.pack("I", use_mesh)                   # 4 bytes
        data += struct.pack("f", float(self.tattoo_density)) # 4 bytes
        scatter_bits = self._scatter_mode_bits[
            int(np.clip(self.scatter_index, 0, len(self._scatter_mode_bits) - 1))
        ]
        data += struct.pack("I", scatter_bits)               # 4 bytes
        data += struct.pack("f", float(self.env_intensity))  # 4 bytes
        data += struct.pack("I", 1 if self.furnace_index != 0 else 0)  # 4 bytes
        data += struct.pack("f", float(self.mm_per_unit))        # 4 bytes
        # Detail-map controls — single `detailFlags` bitfield + two strengths.
        # Bit 0: master enable (mirror of the UI toggle, AND-ed with the
        #        per-map availability bits below so a missing map is always
        #        treated as off, even when the user toggles on).
        # Bit 1: normal map available
        # Bit 2: roughness map available
        # Bit 3: displacement map available
        # Bit 4: normal map already baked into vertex normals (shader skips
        #        its own normal-map sample for mesh hits so we don't
        #        double-apply the same perturbation).
        master = 1 if self.detail_maps_index == 0 else 0
        nrm_ok, rgh_ok, dsp_ok = self._detail_available
        flags = (
            master
            | ((1 if nrm_ok else 0) << 1)
            | ((1 if rgh_ok else 0) << 2)
            | ((1 if dsp_ok else 0) << 3)
            | ((1 if self._baked_normals else 0) << 4)
        )
        data += struct.pack("I", flags)                              # 4 bytes
        data += struct.pack("f", float(self.normal_map_strength))    # 4 bytes
        data += struct.pack("f", float(self.displacement_scale_mm))  # 4 bytes
        data += struct.pack(
            "I",
            int(np.clip(self.integrator_index, 0, len(self.integrator_modes) - 1)),
        )                                                            # 4 bytes
        # Pad to 16-byte alignment (struct of scalars → round up to vec4)
        while len(data) % 16 != 0:
            data += b"\x00"

        # SkinParams
        data += self.skin.pack()
        while len(data) % 16 != 0:
            data += b"\x00"

        # Light
        data += self.light_direction.tobytes()   # 12 bytes
        data += struct.pack("f", 0.0)            # pad
        data += self.light_radiance.tobytes()     # 12 bytes
        data += struct.pack("f", 0.0)            # pad

        return bytes(data)

    def _ensure_env_uploaded(self) -> None:
        """Upload current env to GPU if it has changed (called once per switch)."""
        if self.env_index == self._last_env_index:
            return
        self.env_index = int(np.clip(self.env_index, 0, len(self.environments) - 1))
        env = self.environments[self.env_index]
        self.env_image.upload_sync(env.data)
        self._last_env_index = self.env_index

    @property
    def env_name(self) -> str:
        if 0 <= self.env_index < len(self.environments):
            return self.environments[self.env_index].name
        return "(none)"

    @staticmethod
    def _load_hud_font() -> ImageFont.ImageFont:
        """Try a common monospace TTF; fall back to Pillow's bitmap default."""
        for candidate in (
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/lucon.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, 14)
            except OSError:
                continue
        return ImageFont.load_default()

    def _build_hud_bytes(self) -> bytes:
        """Rasterise hud_text_lines into an R8 alpha mask.

        Pixel value encoding consumed by main_pass.slang:
          0     : transparent
          150   : dim background panel (alpha ≈ 0.59)
          255   : text ink (white)

        TTF anti-aliasing is thresholded away so edges don't fall into the
        panel-alpha range (which would render text edges as dim panel).
        """
        base = Image.new("L", (self.width, self.height), 0)
        if not self.show_hud or not self.hud_text_lines:
            return base.tobytes()

        draw = ImageDraw.Draw(base)
        font = self._hud_font

        line_height = 18
        padding = 10
        max_text_w = 0
        for line in self.hud_text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            max_text_w = max(max_text_w, bbox[2] - bbox[0])

        panel_w = max_text_w + padding * 2
        panel_h = line_height * len(self.hud_text_lines) + padding * 2

        # Panel background
        draw.rectangle((0, 0, panel_w, panel_h), fill=150)

        # Text rendered onto a separate mask, then thresholded to binary so
        # AA edges become crisp ink rather than falling into the panel range.
        text_mask = Image.new("L", (self.width, self.height), 0)
        tdraw = ImageDraw.Draw(text_mask)
        for i, line in enumerate(self.hud_text_lines):
            tdraw.text(
                (padding, padding + i * line_height),
                line,
                fill=255,
                font=font,
            )
        text_mask = text_mask.point(lambda v: 255 if v >= 128 else 0)
        base.paste(255, (0, 0), mask=text_mask)

        return base.tobytes()

    def _current_state_hash(self) -> int:
        """Hash camera + skin + light state. Changes reset the accumulation."""
        parts = (
            self.camera.state_signature(),
            self.skin.melanin_fraction, self.skin.hemoglobin_fraction,
            self.skin.blood_oxygenation, self.skin.epidermis_thickness_mm,
            self.skin.dermis_thickness_mm, self.skin.subcut_thickness_mm,
            self.skin.anisotropy_g, self.skin.roughness, self.skin.ior,
            self.skin.pore_density, self.skin.pore_depth,
            self.skin.hair_density, self.skin.hair_tilt,
            float(self.light_elevation), float(self.light_azimuth),
            float(self.light_intensity),
            float(self.light_color_r), float(self.light_color_g), float(self.light_color_b),
            int(self.env_index),
            int(self.direct_light_index),
            int(self.head_index),
            int(self.tattoo_index),
            float(self.tattoo_density),
            int(self.scatter_index),
            int(self.integrator_index),
            float(self.env_intensity),
            int(self.furnace_index),
            float(self.mm_per_unit),
            int(self.detail_maps_index),
            float(self.normal_map_strength),
            float(self.displacement_scale_mm),
            int(self.subdivision_index),
            int(self.preset_index),
        )
        return hash(parts)

    def update(self, dt: float) -> None:
        self.time_elapsed += dt
        self.frame_index += 1

        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_smooth = (
                inst_fps if self._fps_smooth == 0 else self._fps_smooth * 0.9 + inst_fps * 0.1
            )

        # If the environment selection changed, re-upload the HDR texture.
        self._ensure_env_uploaded()
        # Rebake the head mesh if source / subdivision / displacement-scale
        # drifted from whatever we last built. Uses wall-clock time so slider
        # drags get debounced cleanly regardless of frame rate.
        self._rebake_if_needed(time.monotonic())
        # And for the tattoo texture.
        self._ensure_tattoo_uploaded()

        state = self._current_state_hash()
        if state != self._last_state_hash:
            self.accum_frame = 0
            self._last_state_hash = state
        else:
            self.accum_frame += 1

    def render(self) -> None:
        f = self.current_frame

        vk.vkWaitForFences(
            self.ctx.device, 1, [self.in_flight_fences[f]], vk.VK_TRUE, 2**64 - 1
        )
        vk.vkResetFences(self.ctx.device, 1, [self.in_flight_fences[f]])

        image_index = self.ctx.vkAcquireNextImageKHR(
            self.ctx.device,
            self.ctx.swapchain_info.swapchain,
            2**64 - 1,
            self.image_available[f],
            vk.VK_NULL_HANDLE,
        )

        # Upload uniforms and HUD staging
        self.uniform_buffer.upload(self._pack_uniforms())
        self.hud_overlay.upload(self._build_hud_bytes())

        # Update binding 1 (storage image) with the acquired swapchain image view
        img_info = vk.VkDescriptorImageInfo(
            imageView=self.ctx.swapchain_info.image_views[image_index],
            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
        )
        write = vk.VkWriteDescriptorSet(
            dstSet=self.descriptor_sets[f],
            dstBinding=1,
            dstArrayElement=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[img_info],
        )
        vk.vkUpdateDescriptorSets(self.ctx.device, 1, [write], 0, None)

        # Record command buffer
        cmd = self.command_buffers[f]
        vk.vkResetCommandBuffer(cmd, 0)
        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        # Transition swapchain image to GENERAL for compute write
        barrier = vk.VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.ctx.swapchain_info.images[image_index],
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier],
        )

        # Cross-frame memory dependency on the accumulation image: previous
        # frame's writes must be visible to this frame's reads.
        accum_mem_barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, [accum_mem_barrier], 0, None, 0, None,
        )

        # Copy this frame's HUD bytes from staging into the device-local image.
        self.hud_overlay.record_copy(cmd)

        # Bind pipeline and descriptors
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline.pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline.pipeline_layout,
            0, 1, [self.descriptor_sets[f]],
            0, None,
        )

        # Dispatch
        groups_x = (self.width + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        groups_y = (self.height + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        vk.vkCmdDispatch(cmd, groups_x, groups_y, 1)

        # Transition to PRESENT_SRC
        barrier2 = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=0,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            image=self.ctx.swapchain_info.images[image_index],
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, None, 0, None, 1, [barrier2],
        )

        vk.vkEndCommandBuffer(cmd)

        # Submit
        submit_info = vk.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.image_available[f]],
            pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT],
            commandBufferCount=1,
            pCommandBuffers=[cmd],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.render_finished[image_index]],
        )
        vk.vkQueueSubmit(self.ctx.compute_queue, 1, [submit_info], self.in_flight_fences[f])

        # Present
        present_info = vk.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.render_finished[image_index]],
            swapchainCount=1,
            pSwapchains=[self.ctx.swapchain_info.swapchain],
            pImageIndices=[image_index],
        )
        self.ctx.vkQueuePresentKHR(self.ctx.present_queue, present_info)

        self.current_frame = (f + 1) % MAX_FRAMES_IN_FLIGHT

    def cleanup(self) -> None:
        vk.vkDeviceWaitIdle(self.ctx.device)

        for sem in self.image_available + self.render_finished:
            vk.vkDestroySemaphore(self.ctx.device, sem, None)
        for fence in self.in_flight_fences:
            vk.vkDestroyFence(self.ctx.device, fence, None)

        vk.vkDestroyDescriptorPool(self.ctx.device, self.descriptor_pool, None)
        self.bvh_buffer.destroy()
        self.index_buffer.destroy()
        self.vertex_buffer.destroy()
        self.displacement_image.destroy()
        self.roughness_image.destroy()
        self.normal_image.destroy()
        self.tattoo_image.destroy()
        self.env_image.destroy()
        self.hud_overlay.destroy()
        self.accum_image.destroy()
        self.uniform_buffer.destroy()
        self.pipeline.destroy()
