"""Metal compute resource wrappers — backend-polymorphic sibling of ``vk_compute``.

These classes mirror the **public API** of the :mod:`skinny.vk_compute` resource
classes the renderer consumes — same class names, constructor signatures, and
upload-helper names — so the renderer builds its megakernel GPU resources on a
:class:`~skinny.metal_context.MetalContext` through the resource module resolved
once by :func:`skinny.backend_select.resource_module`, with no backend-specific
branches at the construction sites (design D1). Each wraps a SlangPy / slang-rhi
resource (``Buffer`` / ``Texture`` / ``Sampler``).

The megakernel :class:`ComputePipeline` compiles ``main_pass.slang`` (``mainImage``)
to Metal **in-process** via a SlangPy slang session (design D2/D7) after running
:func:`~skinny.megakernel_sources.emit_megakernel_sources`, reflects the global
binding layout, and dispatches by **binding resources through a
:class:`slangpy.ShaderCursor`** on a root shader object — buffers/textures/samplers
by name and the uniform block via ``set_data`` byte blobs only, never per-field
scalar writes (the P1 D4 fence-hang discipline).

Two Metal-target shader adaptations are reflected in the binding surface here
(design D8 / task 4.0a): the combined ``Sampler2D`` bindless pool is a plain
``Texture2D`` array + a shared ``commonSampler`` (binding 38), and the five
discrete combined samplers are split into ``Texture2D`` + a per-map ``SamplerState``
(bindings 39–43) — slang-rhi's Metal backend cannot bind a combined ``Sampler2D``.
The Vulkan path is byte-identical and unaffected.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Backend-agnostic megakernel-source emission + scene helpers live in the
# Vulkan-free `megakernel_sources` module; re-exported so the renderer can reach
# them through the resolved resource module (`self._gpu.*`) on either backend.
from skinny.megakernel_sources import (  # noqa: F401  (re-export)
    GRAPH_BINDING_BASE,
    emit_megakernel_aggregator,
    emit_megakernel_sources,
    python_material_ids,
    scan_python_materials,
)

# Metal trims the bindless flat-material texture pool to fit Apple Metal's
# 128-texture compute-argument limit alongside the discrete maps (design D8).
# MUST equal the array dimension in `shaders/bindings.slang`'s `#if SKINNY_METAL`
# branch (`Texture2D<float4> flatMaterialTextures[119]`). 119, not 120: the
# 128-texture argument table is otherwise exactly full, and the volumeDensity
# 3D grid (binding 26, nanovdb-volume-rendering) needs the last slot.
BINDLESS_TEXTURE_CAPACITY = 119

# Slang globals for the shared sampler + the per-map samplers added on the Metal
# target (design D8). Used by the renderer to bind them by name.
COMMON_SAMPLER_NAME = "commonSampler"
DISCRETE_MAP_SAMPLERS = {
    "envMap": "envMapSampler",
    "tattooMap": "tattooMapSampler",
    "normalMap": "normalMapSampler",
    "roughnessMap": "roughnessMapSampler",
    "displacementMap": "displacementMapSampler",
}


_shared_fallback_warned = False


def _warn_shared_fallback(reason) -> None:
    """One-shot notice that shared-storage mode degraded to staged uploads
    (change metal-neural-interop, design D3 fallback). The publisher keeps
    working — every write just pays the device-local staging copy."""
    global _shared_fallback_warned
    if _shared_fallback_warned:
        return
    _shared_fallback_warned = True
    print(f"[metal] shared-storage buffer unavailable ({reason}); "
          "falling back to staged uploads", flush=True)


def _buffer_usage(spy, *, indirect: bool = False):
    usage = (
        spy.BufferUsage.unordered_access
        | spy.BufferUsage.shader_resource
        | spy.BufferUsage.copy_source
        | spy.BufferUsage.copy_destination
    )
    if indirect:
        usage |= spy.BufferUsage.indirect_argument
    return usage


# Backend-neutral image-format tokens the renderer passes so a construction site
# is the same on either backend (vk_compute resolves the same tokens to VkFormat).
# Legacy Vulkan `VkFormat` int values are also accepted defensively (no `vulkan`
# import) so a stray int doesn't crash the Metal path.
_FORMAT_TOKENS = {
    "rgba32f": "rgba32_float",
    "rgba32_float": "rgba32_float",
    "rgba8_unorm": "rgba8_unorm",
    # slang-rhi spells the 8-bit sRGB format `rgba8_unorm_srgb` (Vulkan's
    # VK_FORMAT_R8G8B8A8_SRGB / the `rgba8_srgb` neutral token); the bare
    # `rgba8_srgb` is not a `slangpy.Format` member.
    "rgba8_srgb": "rgba8_unorm_srgb",
    "r8_unorm": "r8_unorm",
    "r32_float": "r32_float",
    "r16_float": "r16_float",
}
_VKFORMAT_INTS = {  # VkFormat enum values → slangpy.Format name
    9: "r8_unorm", 37: "rgba8_unorm", 43: "rgba8_unorm_srgb", 109: "rgba32_float",
}
_ADDRESS_TOKENS = {"repeat", "clamp", "mirror", "black", "useMetadata"}
_VK_ADDRESS_INTS = {0: "repeat", 1: "mirror", 2: "clamp", 3: "black"}


def _resolve_format(spy, fmt):
    """Map a backend-neutral format token (or ``None``/``slangpy.Format``/legacy
    VkFormat int) to a ``slangpy.Format``."""
    if fmt is None:
        return spy.Format.rgba32_float
    if isinstance(fmt, str):
        return getattr(spy.Format, _FORMAT_TOKENS.get(fmt, "rgba32_float"))
    if isinstance(fmt, int):  # legacy VkFormat int
        return getattr(spy.Format, _VKFORMAT_INTS.get(fmt, "rgba32_float"))
    return fmt  # already a slangpy.Format


def _resolve_address_mode(mode, default: str) -> str:
    if isinstance(mode, str):
        return mode if mode in _ADDRESS_TOKENS else default
    if isinstance(mode, int):  # legacy VkSamplerAddressMode int
        return _VK_ADDRESS_INTS.get(mode, default)
    return default


def _make_sampler(ctx, *, address_u: str = "repeat", address_v: str = "clamp"):
    """Create a linear sampler, mapping the Vulkan default address modes
    (repeat U, clamp-to-edge V) onto slang-rhi's sampler desc. Falls back to a
    bare default sampler if the keyword surface differs across slangpy builds."""
    spy = ctx._spy
    dev = ctx.device

    def _addr(name):
        # slang-rhi spells clamp-to-edge "clamp_to_edge"; accept "clamp" too.
        amode = getattr(spy, "TextureAddressingMode", None)
        if amode is None:
            return None
        return {
            "repeat": getattr(amode, "wrap", getattr(amode, "repeat", None)),
            "clamp": getattr(amode, "clamp_to_edge", getattr(amode, "clamp", None)),
        }.get(name)

    try:
        kw = {
            "min_filter": spy.TextureFilteringMode.linear,
            "mag_filter": spy.TextureFilteringMode.linear,
        }
        au, av = _addr(address_u), _addr(address_v)
        if au is not None and av is not None:
            kw.update(address_u=au, address_v=av, address_w=av)
        return dev.create_sampler(**kw)
    except Exception:  # noqa: BLE001 — sampler desc surface varies; default is fine
        return dev.create_sampler()


class StorageBuffer:
    """Device-local storage buffer (SSBO) — Metal sibling of
    ``vk_compute.StorageBuffer(ctx, size_bytes, *, indirect=False, external=False)``.

    ``external`` is accepted for signature parity but is a no-op on Metal — there
    are no exported memory handles (capability flag stays ``False``). The Metal
    interop seam is ``shared`` instead (change metal-neural-interop, design D3):
    a shared-mode buffer allocates from the host-visible heap
    (``MemoryType.upload`` → ``MTLStorageModeShared`` on Apple-Silicon UMA) so
    ``write_in_place`` lands host bytes the next dispatch reads with no staging
    round-trip. slangpy 0.42 exposes no mapped pointer, so the in-place write is
    one host memcpy via ``copy_from_numpy`` (probe: task 1.1). A host shadow
    backs ``upload_range`` so partial writes compose."""

    def __init__(self, ctx, size_bytes: int, *, indirect: bool = False,
                 external: bool = False, shared: bool = False) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.size = max(int(size_bytes), 16)  # GPU drivers dislike zero-sized buffers
        self.external = False  # no Metal external-memory export (P1 D5)
        self.shared = False
        self._shadow = bytearray(self.size)
        if shared:
            try:
                self.buffer = ctx.device.create_buffer(
                    size=self.size,
                    usage=_buffer_usage(spy, indirect=indirect),
                    memory_type=spy.MemoryType.upload,
                    label="skinny.storage.shared",
                )
                self.shared = True
            except Exception as exc:  # noqa: BLE001 — degrade to staged uploads
                _warn_shared_fallback(exc)
        if not self.shared:
            self.buffer = ctx.device.create_buffer(
                size=self.size,
                usage=_buffer_usage(spy, indirect=indirect),
                memory_type=spy.MemoryType.device_local,
                label="skinny.storage",
            )

    def _flush(self) -> None:
        self.buffer.copy_from_numpy(
            np.frombuffer(bytes(self._shadow), dtype=np.uint8).copy())

    def write_in_place(self, data: bytes, offset: int = 0) -> None:
        """Host-write ``data`` at ``offset`` so the next dispatch reads it.

        On a shared-mode buffer this is the UMA in-place path: one host memcpy
        into the shared heap, no staging upload. The caller owns hazard ordering
        — write only when no in-flight command buffer reads the buffer (the
        neural publisher calls this at the frame boundary, design D1). On a
        non-shared buffer it degenerates to the staged ``upload_range`` path
        (logged once) so the publisher still works."""
        if not self.shared:
            _warn_shared_fallback("write_in_place on a non-shared buffer")
        end = int(offset) + len(data)
        if end > self.size:
            raise ValueError(
                f"StorageBuffer write_in_place: {offset}+{len(data)}B "
                f"> buffer {self.size}B"
            )
        self._shadow[int(offset):end] = bytes(data)
        self._flush()

    def upload_sync(self, data: bytes) -> None:
        payload = bytes(data)
        if len(payload) > self.size:
            raise ValueError(
                f"StorageBuffer upload: payload {len(payload)}B > buffer {self.size}B"
            )
        self._shadow[: len(payload)] = payload
        self._flush()

    def upload_range(self, data: bytes, dst_offset: int) -> None:
        if not data:
            return
        end = int(dst_offset) + len(data)
        if end > self.size:
            raise ValueError(
                f"StorageBuffer upload_range: {dst_offset}+{len(data)}B "
                f"> buffer {self.size}B"
            )
        self._shadow[int(dst_offset):end] = bytes(data)
        # Partial GPU write via the encoder upload — a full-shadow
        # copy_from_numpy would re-upload the whole buffer per call, which the
        # per-frame record-drain counter reset cannot afford on a multi-MB
        # drain target (change metal-record-drain). Falls back to the full
        # flush on builds without the entry point.
        try:
            enc = self.ctx.device.create_command_encoder()
            enc.upload_buffer_data(
                self.buffer, int(dst_offset),
                np.frombuffer(bytes(data), dtype=np.uint8).copy())
            self.ctx.device.submit_command_buffer(enc.finish())
        except Exception:  # noqa: BLE001 — older slangpy: full flush still correct
            self._flush()

    def download_sync(self, byte_count: int) -> bytes:
        self.ctx.device.wait_for_idle()
        n = min(int(byte_count), self.size)
        if self.shared:
            # slang-rhi refuses `to_numpy()` on an upload-heap buffer — bounce
            # the bytes through a read_back staging copy so a shared buffer
            # reads back its true GPU-visible contents like any other.
            spy = self.ctx._spy
            staging = self.ctx.device.create_buffer(
                size=self.size,
                usage=spy.BufferUsage.copy_destination,
                memory_type=spy.MemoryType.read_back,
                label="skinny.storage.shared.readback",
            )
            enc = self.ctx.device.create_command_encoder()
            enc.copy_buffer(staging, 0, self.buffer, 0, self.size)
            self.ctx.device.submit_command_buffer(enc.finish())
            self.ctx.device.wait_for_idle()
            return staging.to_numpy().tobytes()[:n]
        return self.buffer.to_numpy().tobytes()[:n]

    def fill_zero_sync(self) -> None:
        self._shadow = bytearray(self.size)
        self._flush()

    def export_handle(self):  # signature parity; no external memory on Metal
        return None

    def destroy(self) -> None:  # SlangPy owns lifetime via refcount
        self.buffer = None


class HostStorageBuffer:
    """Host-visible storage buffer (SSBO) — sibling of
    ``vk_compute.HostStorageBuffer``. Used at binding 30 (toolBuffer). Backed by an
    ``upload``-heap buffer the CPU writes and the GPU reads."""

    def __init__(self, ctx, size_bytes: int) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.size = max(int(size_bytes), 16)
        self._shadow = bytearray(self.size)
        self.buffer = ctx.device.create_buffer(
            size=self.size,
            usage=_buffer_usage(spy),
            memory_type=spy.MemoryType.device_local,
            label="skinny.host_storage",
        )
        # Zero-initialise the GPU buffer. Device-local memory is uninitialised
        # garbage otherwise, and the megakernel reads `toolBuffer[0].x` every
        # frame as the tool mode (binding 30) — a stray non-zero value hijacks
        # `mainImage` into the BXDF/BSSRDF tool branch and skips the scene render.
        self.buffer.copy_from_numpy(
            np.frombuffer(bytes(self._shadow), dtype=np.uint8).copy())

    def write(self, data: bytes, offset: int = 0) -> None:
        end = int(offset) + len(data)
        if end > self.size:
            raise ValueError(
                f"HostStorageBuffer.write: {end}B > buffer {self.size}B"
            )
        self._shadow[int(offset):end] = bytes(data)
        self.buffer.copy_from_numpy(
            np.frombuffer(bytes(self._shadow), dtype=np.uint8).copy())

    def read(self, length: int, offset: int = 0) -> bytes:
        end = int(offset) + int(length)
        if end > self.size:
            raise ValueError(
                f"HostStorageBuffer.read: {end}B > buffer {self.size}B"
            )
        self.ctx.device.wait_for_idle()
        return self.buffer.to_numpy().tobytes()[int(offset):end]

    def destroy(self) -> None:
        self.buffer = None


class UniformBuffer:
    """Uniform-block holder — sibling of ``vk_compute.UniformBuffer(ctx, size_bytes)``.

    On Metal the megakernel uniform (`fc`) is **not** a separate GPU buffer: it is
    set on the pipeline each frame via ``ShaderCursor.set_data`` byte blobs (design
    D4). This wrapper keeps the renderer's ``uniform_buffer.upload(blob)`` call site
    unchanged by stashing the latest blob; the Metal dispatch reads :attr:`latest`."""

    def __init__(self, ctx, size_bytes: int) -> None:
        self.ctx = ctx
        self.size = int(size_bytes)
        self.latest: bytes = b"\x00" * self.size

    def upload(self, data: bytes) -> None:
        self.latest = bytes(data)

    def destroy(self) -> None:
        self.latest = b""


class StorageImage:
    """Device-local 2D storage image (``RWTexture2D``) — sibling of
    ``vk_compute.StorageImage(ctx, width, height, format=…, transfer_src=False)``.

    ``format`` is a ``slangpy.Format`` (default ``rgba32_float`` to match the Vulkan
    ``R32G32B32A32_SFLOAT`` accumulation/output image)."""

    def __init__(self, ctx, width: int, height: int, format=None,
                 transfer_src: bool = False) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.width = int(width)
        self.height = int(height)
        # The megakernel's storage images (`outputBuffer`, `accumBuffer`) are
        # `RWTexture2D<float4>` with read-write access. Metal only guarantees
        # read-write on a subset of formats; `rgba32_float` is always safe, so a
        # storage image is float32 regardless of the requested token (the Vulkan
        # blit-source `rgba8_unorm` choice is irrelevant on Metal — the display
        # readback converts float → RGBA8). Avoids an unsupported-access abort.
        self.format = spy.Format.rgba32_float
        usage = (
            spy.TextureUsage.unordered_access
            | spy.TextureUsage.shader_resource
            | spy.TextureUsage.copy_source
            | spy.TextureUsage.copy_destination
        )
        self.texture = ctx.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=self.format,
            width=self.width,
            height=self.height,
            usage=usage,
            memory_type=spy.MemoryType.device_local,
            label="skinny.storage_image",
        )

    def read_rgba(self) -> np.ndarray:
        """Drain and return the image as an ``(H, W, 4)`` array (native format)."""
        self.ctx.device.wait_for_idle()
        return self.texture.to_numpy()

    def destroy(self) -> None:
        self.texture = None


class SampledImage:
    """Device-local sampled 2D image + sampler — sibling of
    ``vk_compute.SampledImage``.

    On the Metal target the combined ``Sampler2D`` is split into a ``Texture2D``
    plus a discrete ``SamplerState`` (design D8): :attr:`texture` binds the texture
    global (e.g. ``envMap``) and :attr:`sampler` binds the paired sampler global
    (e.g. ``envMapSampler``)."""

    def __init__(self, ctx, width: int, height: int, format=None,
                 bytes_per_pixel: int = 16,
                 address_mode_u="repeat",
                 address_mode_v="clamp") -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.width = int(width)
        self.height = int(height)
        self.format = _resolve_format(spy, format)
        address_mode_u = _resolve_address_mode(address_mode_u, "repeat")
        address_mode_v = _resolve_address_mode(address_mode_v, "clamp")
        # Host pixel dtype/channels for upload reinterpretation (the renderer
        # passes raw bytes for RGBA8 maps and float arrays for HDR/env images).
        if self.format == spy.Format.rgba32_float:
            self._np_dtype, self._channels = np.float32, 4
        elif self.format == spy.Format.r8_unorm:
            self._np_dtype, self._channels = np.uint8, 1
        else:  # rgba8_unorm / rgba8_srgb
            self._np_dtype, self._channels = np.uint8, 4
        self.bytes_per_pixel = int(bytes_per_pixel)
        self._byte_count = self.width * self.height * self.bytes_per_pixel
        self.texture = ctx.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=self.format,
            width=self.width,
            height=self.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local,
            label="skinny.sampled_image",
        )
        self.sampler = _make_sampler(
            ctx, address_u=address_mode_u, address_v=address_mode_v
        )

    def upload_sync(self, data) -> None:
        """Copy host pixels into the image. Accepts raw bytes (RGBA8/R8 maps) or a
        numpy array (HDR/env float images); reinterpreted to the texture's pixel
        format. r8 images use an ``(H, W)`` array; others ``(H, W, C)``."""
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        arr = np.frombuffer(raw, dtype=self._np_dtype).copy()
        if self._channels == 1:
            arr = arr.reshape(self.height, self.width)
        else:
            arr = arr.reshape(self.height, self.width, self._channels)
        self.texture.copy_from_numpy(arr)

    def destroy(self) -> None:
        self.texture = None
        self.sampler = None


class SampledImage3D:
    """Device-local sampled 3D image + sampler — sibling of
    ``vk_compute.SampledImage3D`` (nanovdb-volume-rendering, design D3).

    R16F single-channel density field. Trilinear min/mag filtering, clamp-to-edge
    on all three axes (the shader additionally zeroes samples outside [0,1]³ so
    edge clamping cannot smear boundary density outward). Like the 2D
    ``SampledImage``, the Metal target splits the combined sampler:
    :attr:`texture` binds the ``Texture3D`` global (``volumeDensity``) and
    :attr:`sampler` the paired ``SamplerState`` (``volumeDensitySampler``)."""

    def __init__(self, ctx, width: int, height: int, depth: int) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)
        self.format = spy.Format.r16_float
        self.texture = ctx.device.create_texture(
            type=spy.TextureType.texture_3d,
            format=self.format,
            width=self.width,
            height=self.height,
            depth=self.depth,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local,
            label="skinny.sampled_image_3d",
        )
        self.sampler = _make_sampler(ctx, address_u="clamp", address_v="clamp")

    def upload_sync(self, voxels: np.ndarray) -> None:
        """Copy a ``(depth, height, width)`` float16/float32 array into the
        texture (float32 is converted to float16 on upload)."""
        arr = np.ascontiguousarray(voxels)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        expected = (self.depth, self.height, self.width)
        if arr.shape != expected:
            raise ValueError(
                f"volume upload: got shape {arr.shape}, expected (D,H,W)={expected}")
        self.texture.copy_from_numpy(arr)

    def destroy(self) -> None:
        self.texture = None
        self.sampler = None


class HudOverlay:
    """R8 HUD alpha overlay (binding 3, ``hudMask``) — sibling of
    ``vk_compute.HudOverlay``. The host writes the mask; the shader reads it. On
    Metal the upload writes straight into the texture (no staging command buffer);
    :meth:`record_copy` is a no-op the Metal render path never needs."""

    def __init__(self, ctx, width: int, height: int) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.width = int(width)
        self.height = int(height)
        self._byte_count = self.width * self.height
        self.texture = ctx.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.r8_unorm,
            width=self.width,
            height=self.height,
            usage=(spy.TextureUsage.unordered_access
                   | spy.TextureUsage.shader_resource
                   | spy.TextureUsage.copy_destination),
            memory_type=spy.MemoryType.device_local,
            label="skinny.hud_overlay",
        )

    def upload(self, data: bytes) -> None:
        if len(data) != self._byte_count:
            raise ValueError(
                f"HUD payload is {len(data)} bytes, expected {self._byte_count}"
            )
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(
            self.height, self.width).copy()
        self.texture.copy_from_numpy(arr)

    def record_copy(self, cmd=None) -> None:  # Vulkan-only seam; upload is immediate
        return None

    def destroy(self) -> None:
        self.texture = None


class ReadbackBuffer:
    """Image readback — sibling of ``vk_compute.ReadbackBuffer``. On Metal the
    offscreen image is read back directly via ``StorageImage.read_rgba`` in the
    Metal render path, so this is a thin holder kept for construction parity."""

    def __init__(self, ctx, width: int, height: int, bytes_per_pixel: int = 4):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self._src = None

    def record_copy_from(self, cmd, src_image) -> None:
        self._src = src_image

    def read(self) -> bytes:
        if self._src is None:
            return b"\x00" * (self.width * self.height * 4)
        arr = self._src.read_rgba()
        return _rgba_f32_to_rgba8(arr).tobytes()

    def destroy(self) -> None:
        self._src = None


def _rgba_f32_to_rgba8(arr: np.ndarray) -> np.ndarray:
    """Clamp an ``(H, W, 4)`` float image to RGBA8 (display pixels already in [0,1])."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class ComputePipeline:
    """Metal megakernel pipeline — compiles ``main_pass.slang`` (``mainImage``)
    in-process and dispatches it (design D2/D7).

    Mirrors the ``vk_compute.ComputePipeline(ctx, shader_dir, entry_module,
    entry_point, graph_fragments=None, *, compile_pipeline=True)`` signature. Runs
    :func:`emit_megakernel_sources` (so the generated material modules
    ``main_pass`` imports are present) then builds a SlangPy slang session with
    ``include_paths=[shaders, mtlx/genslang]`` and defines
    ``SKINNY_COMPUTE_PIPELINE=1`` + ``SKINNY_METAL=1`` (no ``-fvk-use-scalar-layout``
    — Metal uses MSL layout), links the entry, and creates the compute pipeline.
    Exposes ``graph_bindings`` / ``python_material_modules`` like the Vulkan one,
    plus :attr:`uniform_layout` (reflected MSL ``fc`` offsets) for the MSL packer."""

    def __init__(self, ctx, shader_dir, entry_module: str = "main_pass",
                 entry_point: str = "mainImage", graph_fragments=None, *,
                 compile_pipeline: bool = True, spectral: bool = False) -> None:
        self.ctx = ctx
        self._spy = ctx._spy
        self.shader_dir = Path(shader_dir)
        self.entry_module = entry_module
        self.entry_point = entry_point
        # Spectral megakernel variant — adds the `SKINNY_SPECTRAL=1` compile
        # define. Off ⇒ byte-identical RGB path (mirrors vk_compute).
        self.spectral = bool(spectral)
        self.graph_fragments = list(graph_fragments) if graph_fragments else []

        # Emit the generated material/dispatcher Slang both backends need before
        # linking main_pass (design D7).
        self.graph_bindings, self.python_material_modules = emit_megakernel_sources(
            self.shader_dir, self.graph_fragments
        )

        # A duck-typed "pipeline present" sentinel mirrors the Vulkan attribute the
        # renderer's `_backend_render_ready` checks. Wavefront/scene-bindings-only
        # builds keep it None.
        self.pipeline = None
        self.uniform_layout = {}
        self.uniform_size = 0
        # Reflected MSL layout of the `mtlxSkin` StructuredBuffer element
        # (binding 15). Maps field-name → (msl_offset, msl_size); `stride` is the
        # MSL element stride. Empty/0 until `_build` reflects them (or if the
        # buffer is dead-stripped). Drives the renderer's `_pack_mtlx_skin_array_msl`
        # repack — Slang pads each `float3` to 16 B on Metal, so the gen-reflected
        # scalar record (`pack_material_values`) must be relocated field-by-field.
        self.mtlx_skin_layout = {}
        self.mtlx_skin_stride = 0
        # Always empty since change combine-graph-param-buffers: graph params
        # live in one byte buffer (`graphParamsCombined`, binding 25) read via
        # `Load<GraphParams_X>`, which uses scalar layout identical on Metal and
        # SPIR-V — so the host packs one scalar blob and there is no per-field MSL
        # layout to reflect (kept for the duck-typed `_msl_layout_source` surface).
        self.graph_param_layouts: dict[str, tuple[dict[str, tuple[int, int]], int]] = {}
        # Reflected MSL layout of the `StructuredBuffer<StdSurfaceParams>` global
        # (`stdSurfaceParams`, binding 19). `{field: (msl_offset, msl_size)}` plus
        # the MSL element stride. Same float3→16 B MSL padding hazard as
        # `graph_param_layouts`/`mtlx_skin_layout`: the host packs the record in
        # scalar/std430 layout (`pack_std_surface_params`, float3 = 12 B), so the
        # renderer relocates each field into this MSL layout per slot
        # (`pack_std_surface_params_msl`). Empty/0 until `_build` reflects them.
        self.std_surface_layout: dict[str, tuple[int, int]] = {}
        self.std_surface_stride: int = 0
        self._default_tex = None
        self._kernel = None  # lazy compute-kernel for the generic dispatch_kernel path

        if not compile_pipeline:
            return

        self._build()

    @classmethod
    def scene_bindings_only(cls, ctx, shader_dir, graph_fragments=None):
        """Emit the scene plumbing without compiling/dispatching the megakernel —
        Metal sibling of ``vk_compute.ComputePipeline.scene_bindings_only``."""
        return cls(ctx, shader_dir, entry_module="main_pass", entry_point="mainImage",
                   graph_fragments=graph_fragments, compile_pipeline=False)

    # ── Compile + reflect ────────────────────────────────────────

    def _build(self) -> None:
        spy = self._spy
        dev = self.ctx.device
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        opts = spy.SlangCompilerOptions()
        opts.include_paths = [self.shader_dir, mtlx_genslang]
        # Build the full dict and assign ONCE: SlangCompilerOptions.defines is a
        # property whose getter returns a fresh copy, so `opts.defines[k] = v`
        # mutates a throwaway and is silently lost (the spectral define never
        # reached the compile → the megakernel ran the RGB variant).
        defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"}
        if self.spectral:
            defines["SKINNY_SPECTRAL"] = "1"
        opts.defines = defines
        # Match the Vulkan `slangc` path's matrix layout. slangc defaults to
        # column-major (HLSL/Slang default — the Vulkan flags never override it)
        # and both the `_pack_uniforms` camera matrices and the `Instance`
        # `float4x4` transforms (binding 12) are packed for that convention.
        # SlangPy defaults to row_major, which transposes every matrix on Metal:
        # the camera `mul`s drop the per-pixel projection term AND the instance
        # world↔local transform is wrong, so every primary ray misses the head and
        # the frame collapses to the flat environment. Column-major aligns Metal
        # with Vulkan so the same packed bytes are read identically.
        opts.matrix_layout = spy.SlangMatrixLayout.column_major
        session = dev.create_slang_session(compiler_options=opts)

        src_path = self.shader_dir / f"{self.entry_module}.slang"
        module = session.load_module_from_source(
            self.entry_module, src_path.read_text(encoding="utf-8"), str(src_path)
        )
        self.program = session.link_program(
            [module], [module.entry_point(self.entry_point)]
        )
        self.pipeline = dev.create_compute_pipeline(program=self.program)

        self._reflect_globals()
        self._reflect_uniform_layout()
        self._reflect_mtlx_skin_layout()
        self._reflect_graph_param_layouts()
        self._reflect_std_surface_layout()
        # Default 1×1 texture for unfilled bindless slots (Metal binds every slot).
        self._default_tex = dev.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local, label="skinny.bindless_default",
        )

    def _reflect_globals(self) -> None:
        """Record the set of reflected global parameter names so the renderer only
        binds slots that survived dead-stripping."""
        self.global_names = {p.name for p in self.program.layout.parameters}

    def _reflect_uniform_layout(self) -> None:
        """Reflect the MSL field offsets/size of the ``fc`` uniform block (design
        D3). ``uniform_layout`` maps a flattened field name → (offset, size); the
        embedded ``camera`` struct is flattened as ``camera.<field>``."""
        fc = next((p for p in self.program.layout.parameters if p.name == "fc"), None)
        if fc is None:
            return
        tl = fc.type_layout
        self.uniform_size = int(tl.size)
        layout: dict[str, tuple[int, int]] = {}

        def walk(type_layout, base: int, prefix: str) -> None:
            fields = getattr(type_layout, "fields", None)
            if not fields:
                return
            for f in fields:
                off = base + int(f.offset)
                ftl = f.type_layout
                name = f"{prefix}{f.name}"
                layout[name] = (off, int(getattr(ftl, "size", 0)))
                if getattr(ftl, "fields", None):
                    walk(ftl, off, f"{name}.")

        walk(tl, 0, "")
        self.uniform_layout = layout

    def _reflect_mtlx_skin_layout(self) -> None:
        """Reflect the MSL field offsets/size + element stride of the ``mtlxSkin``
        ``StructuredBuffer<MtlxSkinParams>`` (binding 15). ``mtlx_skin_layout`` maps
        each field name → (offset, size) inside one MSL record; ``mtlx_skin_stride``
        is the per-element stride. Empty/0 when the buffer is dead-stripped (no skin
        material in the scene). Unlike ``fc``, the float3 fields here sit at
        non-16-aligned scalar offsets, so the renderer cannot float4-wrap them — it
        repacks the gen-reflected scalar record into this MSL layout per frame
        (`_pack_mtlx_skin_array_msl`)."""
        p = next((p for p in self.program.layout.parameters
                  if p.name == "mtlxSkin"), None)
        if p is None:
            return
        # StructuredBuffer<T> → element type layout carries the struct fields.
        etl = getattr(p.type_layout, "element_type_layout", None)
        if etl is None or not getattr(etl, "fields", None):
            return
        layout: dict[str, tuple[int, int]] = {}
        for f in etl.fields:
            ftl = f.type_layout
            layout[f.name] = (int(f.offset), int(getattr(ftl, "size", 0)))
        self.mtlx_skin_layout = layout
        self.mtlx_skin_stride = int(getattr(etl, "stride", 0) or etl.size)

    def _reflect_graph_param_layouts(self) -> None:
        """No-op since change combine-graph-param-buffers: graph params now live
        in one byte-addressed ``ByteAddressBuffer graphParamsCombined`` read via
        ``Load<GraphParams_X>``, which uses scalar layout identical on Metal and
        SPIR-V — so the host packs one scalar blob and there is no per-field MSL
        layout to reflect (the former per-graph ``StructuredBuffer<GraphParams_*>``
        globals no longer exist). ``graph_param_layouts`` stays an empty dict for
        the duck-typed ``_msl_layout_source`` surface."""
        return

    def _reflect_std_surface_layout(self) -> None:
        """Reflect the MSL field offsets/size + element stride of the
        ``StructuredBuffer<StdSurfaceParams>`` global (``stdSurfaceParams``,
        binding 19). Stores ``std_surface_layout = {field: (offset, size)}`` and
        ``std_surface_stride``. Same rationale as
        :meth:`_reflect_graph_param_layouts`: ``StdSurfaceParams`` is a run of
        ``float3``+scalar fields the host packs in scalar layout
        (``pack_std_surface_params``), but Slang reads it MSL-padded on Metal
        (``float3``→16 B, 16-aligned), shifting every field after ``base_color``
        and growing the element stride past the scalar 256 B (≈400 B). The
        renderer relocates each scalar field into this layout per slot
        (``pack_std_surface_params_msl``). Empty/0 if the binding was
        dead-stripped (no std_surface material in the linked program)."""
        p = next((q for q in self.program.layout.parameters
                  if q.name == "stdSurfaceParams"), None)
        if p is None:
            return
        etl = getattr(p.type_layout, "element_type_layout", None)
        if etl is None or not getattr(etl, "fields", None):
            return
        self.std_surface_layout = {
            f.name: (int(f.offset), int(getattr(f.type_layout, "size", 0)))
            for f in etl.fields
        }
        self.std_surface_stride = int(getattr(etl, "stride", 0) or etl.size)

    # ── Dispatch ─────────────────────────────────────────────────

    def dispatch(self, width: int, height: int, *, uniform_blob: bytes,
                 binds: dict, bindless=None, bands: int = 1,
                 tile_origin_offset: int | None = None) -> None:
        """Bind ``fc`` (via ``set_data``), every resource in ``binds`` (name → native
        SlangPy ``Buffer``/``Texture``/``Sampler``), and the optional bindless texture
        array, then dispatch ``main_pass`` over ``width × height`` threads.

        ``bindless`` is ``(global_name, [native_texture | None, …])``; ``None`` slots
        bind the default 1×1 texture (Metal requires every array slot bound). Only
        names present in the reflected globals are bound (unused ones are
        dead-stripped). No per-field scalar cursor writes anywhere (design D4).

        ``bands`` (change metal-megakernel-watchdog-tiling): split the frame into
        this many screen-space row bands and commit ONE command buffer per band, so
        no single buffer covers the full frame. macOS cannot cancel another
        process's GPU work, so a full-frame BDPT dispatch over heavy (graph-material)
        pixels can exceed the GPU watchdog and wedge the device; tiling bounds each
        command buffer to ``width × bandHeight`` pixels. The band Y origin is patched
        into the ``tileOriginY`` u32 of the fc blob at ``tile_origin_offset`` (its
        reflected MSL byte offset; falls back to the final u32 when ``None``). The
        shader adds it to the dispatch thread's y under ``SKINNY_METAL``. The resource
        + bindless binds are set once on the shared root object and reused across
        bands. ``bands == 1`` reproduces the original single full-frame dispatch
        exactly (origin patched to 0)."""
        spy = self._spy
        dev = self.ctx.device
        ro = dev.create_root_shader_object(self.program)
        cur = spy.ShaderCursor(ro)

        blob = np.frombuffer(bytes(uniform_blob), dtype=np.uint8).copy()
        toff = len(blob) - 4 if tile_origin_offset is None else int(tile_origin_offset)

        for name, native in binds.items():
            if native is None or name not in self.global_names:
                continue
            cur[name] = native

        if bindless is not None:
            name, textures = bindless
            if name in self.global_names:
                slot_cur = cur[name]
                for i in range(BINDLESS_TEXTURE_CAPACITY):
                    tex = textures[i] if i < len(textures) and textures[i] is not None \
                        else self._default_tex
                    slot_cur[i] = tex

        n_bands = max(1, int(bands))
        band_h = (int(height) + n_bands - 1) // n_bands
        for y0 in range(0, int(height), band_h):
            h = min(band_h, int(height) - y0)
            # Patch only the tileOriginY u32 per band rather than re-pack the whole
            # blob; the shader offsets pixel.y by it (change
            # metal-megakernel-watchdog-tiling).
            blob[toff:toff + 4] = np.frombuffer(np.uint32(y0).tobytes(), dtype=np.uint8)
            cur["fc"].set_data(blob)
            enc = dev.create_command_encoder()
            cpass = enc.begin_compute_pass()
            cpass.bind_pipeline(self.pipeline, ro)
            cpass.dispatch([int(width), int(h), 1])
            cpass.end()
            dev.submit_command_buffer(enc.finish())
            dev.wait_for_idle()

    def dispatch_kernel(self, thread_count, *, buffers=None, vars=None) -> None:
        """Generic low-level dispatch for non-megakernel kernels (the foundation
        trivial path). Binds shader globals by name from :class:`StorageBuffer`s
        (their native buffer) plus any already-native ``vars``, and dispatches the
        kernel over ``thread_count`` (an int triple). Resource binding only — no
        per-field cursor writes (design D4). Separate from the megakernel
        :meth:`dispatch`, which binds the ``fc`` uniform block + descriptor map."""
        dev = self.ctx.device
        if self._kernel is None:
            self._kernel = dev.create_compute_kernel(self.program)
        bound = dict(vars or {})
        for name, res in (buffers or {}).items():
            bound[name] = getattr(res, "buffer", getattr(res, "texture", res))
        self._kernel.dispatch(thread_count=list(thread_count), vars=bound)
        dev.wait_for_idle()

    def dispatch_indirect(self, args_buffer, offset: int = 0, *, bindings=None) -> None:
        """Indirect compute dispatch (design D2) — Metal sibling of the Vulkan
        ``vkCmdDispatchIndirect`` the wavefront per-material shade uses.

        The thread-group count is a ``VkDispatchIndirectCommand`` triple
        ``(x, y, z)`` of ``uint32`` stored in ``args_buffer`` at byte ``offset`` —
        exactly the layout a ``build_args`` kernel writes. ``args_buffer`` is a
        :class:`StorageBuffer` (built with ``indirect=True``) or a native
        ``Buffer``; ``bindings`` is ``name → resource`` (wrapper or native
        ``Buffer``/``Texture``/``Sampler``), bound by name on the program's root
        object, skipping names absent from the reflected globals (dead-stripped).
        Resource binding only — no per-field scalar cursor writes (design D4).

        Two paths, selected by the context's one-time ``supports_indirect_dispatch``
        probe (design D2):

        * **native** — feed ``args_buffer`` straight to
          ``dispatch_compute_indirect``; the group count is read GPU-side, no host
          round-trip.
        * **CPU-readback fallback (task 1.3)** — when the device no-ops indirect
          dispatch (slang-rhi 0.42's Metal backend), drain the device, read the
          ``(x, y, z)`` triple back to the host, and issue an equivalent direct
          ``dispatch_compute`` over the same group count. Correct, but a per-call
          host↔GPU sync. The two paths issue the **same** group count by
          construction (same triple), which the task 1.6 unit test asserts.

        This standalone form opens, submits, and drains its own command encoder; the
        multi-pass loop (task 1.4) re-encodes the same dispatch into one shared
        frame encoder.
        """
        spy = self._spy
        dev = self.ctx.device
        native_args = getattr(args_buffer, "buffer", args_buffer)

        ro = dev.create_root_shader_object(self.program)
        cur = spy.ShaderCursor(ro)
        for name, res in (bindings or {}).items():
            if res is None or name not in self.global_names:
                continue
            cur[name] = getattr(res, "buffer", getattr(res, "texture", res))

        groups = None
        if not self.ctx.supports_indirect_dispatch:
            # Drain so a GPU-written args buffer (a prior build_args dispatch) is
            # visible, then read the (x, y, z) group-count triple to the host.
            dev.wait_for_idle()
            raw = native_args.to_numpy().tobytes()
            gx, gy, gz = struct.unpack_from("<III", raw, int(offset))
            groups = (int(gx), int(gy), int(gz))

        enc = dev.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.bind_pipeline(self.pipeline, ro)
        if groups is None:
            cpass.dispatch_compute_indirect(spy.BufferOffsetPair(native_args, int(offset)))
        else:
            cpass.dispatch_compute(spy.math.uint3(*groups))
        cpass.end()
        dev.submit_command_buffer(enc.finish())
        dev.wait_for_idle()

    def destroy(self) -> None:
        self.pipeline = None
        self._default_tex = None
        self._kernel = None


class PreviewPipelineMetal:
    """Metal material-preview pipeline (change metal-tool-dock-render P1, design
    D1) — the native-Metal sibling of ``vk_compute.PreviewPipeline``.

    Compiles ``preview_pass.slang`` (``previewMain``) to MSL through a SlangPy
    slang session configured **exactly** like the megakernel
    :class:`ComputePipeline` (same ``include_paths``, ``SKINNY_COMPUTE_PIPELINE``/
    ``SKINNY_METAL`` defines, and ``column_major`` matrix layout), linking the
    same emit-time ``generated_materials`` modules (already written to disk by
    :func:`emit_megakernel_sources` for the scene's graph set) so the preview
    shades identically to the main render. There are **no Vulkan descriptor
    sets**: :meth:`dispatch` binds the scene material resources + the preview
    output image by name on a root shader object, sets the ``fc`` uniform block
    and the ``pc`` push block via ``set_data`` (no per-field cursor writes,
    design D4), runs one compute pass over ``size × size``, submits, and waits
    idle (Metal auto-syncs the dispatch→readback, so no barriers).
    """

    def __init__(self, ctx, shader_dir, graph_fragments=None) -> None:
        self.ctx = ctx
        self._spy = ctx._spy
        self.shader_dir = Path(shader_dir)
        self.graph_fragments = list(graph_fragments) if graph_fragments else []
        # Reflected MSL layout of this program's `fc` uniform block (see
        # `_reflect_uniform_layout`) — the renderer packs the `fc` blob against
        # this so the preview never depends on the megakernel/wavefront layout
        # source existing yet. Duck-typed like `ComputePipeline` so
        # `_pack_uniforms_msl` can consume either.
        self.uniform_layout: dict[str, tuple[int, int]] = {}
        self.uniform_size = 0
        # Emit the generated material/dispatcher Slang preview_pass imports.
        # Idempotent: the scene `ComputePipeline` already wrote the same files
        # for this fragment set; re-emitting keeps this pipeline self-contained.
        emit_megakernel_sources(self.shader_dir, self.graph_fragments)
        self._build()

    def _build(self) -> None:
        spy = self._spy
        dev = self.ctx.device
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        opts = spy.SlangCompilerOptions()
        opts.include_paths = [self.shader_dir, mtlx_genslang]
        opts.defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"}
        # Column-major matches the megakernel path so the shared `fc` camera
        # matrices read identically (see ComputePipeline._build).
        opts.matrix_layout = spy.SlangMatrixLayout.column_major
        session = dev.create_slang_session(compiler_options=opts)

        src_path = self.shader_dir / "preview_pass.slang"
        module = session.load_module_from_source(
            "preview_pass", src_path.read_text(encoding="utf-8"), str(src_path)
        )
        self.program = session.link_program(
            [module], [module.entry_point("previewMain")]
        )
        self.pipeline = dev.create_compute_pipeline(program=self.program)
        # Only bind names the compiled module actually references (dead-stripped
        # ones are skipped) — same discipline as ComputePipeline.dispatch.
        self.global_names = {p.name for p in self.program.layout.parameters}
        # Reflect this program's OWN `fc` uniform layout so the preview packs the
        # `fc` blob against its own reflection — independent of whether a
        # megakernel / wavefront pass has compiled yet. In Metal wavefront mode
        # the scene bindings are `scene_bindings_only` (no reflection) and
        # `_msl_layout_source` may be None until a pass builds, so relying on it
        # here would crash. `fc` is the same struct in every program, so the
        # renderer can pack its scalar blob against this layout identically.
        self._reflect_uniform_layout()
        # Default 1×1 texture for unfilled bindless slots (Metal binds every slot).
        self._default_tex = dev.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local, label="skinny.preview_bindless_default",
        )

    def _reflect_uniform_layout(self) -> None:
        """Reflect the MSL field offsets/size of the ``fc`` uniform block from
        THIS program (mirrors ``ComputePipeline._reflect_uniform_layout``).
        ``uniform_layout`` maps a flattened field name → (offset, size); the
        embedded ``camera`` struct is flattened as ``camera.<field>``. Lets the
        renderer's ``_pack_uniforms_msl`` relocate the scalar ``fc`` blob into
        the preview program's MSL layout without a megakernel/wavefront source."""
        fc = next((p for p in self.program.layout.parameters if p.name == "fc"), None)
        if fc is None:
            return
        tl = fc.type_layout
        self.uniform_size = int(tl.size)
        layout: dict[str, tuple[int, int]] = {}

        def walk(type_layout, base: int, prefix: str) -> None:
            fields = getattr(type_layout, "fields", None)
            if not fields:
                return
            for f in fields:
                off = base + int(f.offset)
                ftl = f.type_layout
                name = f"{prefix}{f.name}"
                layout[name] = (off, int(getattr(ftl, "size", 0)))
                if getattr(ftl, "fields", None):
                    walk(ftl, off, f"{name}.")

        walk(tl, 0, "")
        self.uniform_layout = layout

    def dispatch(self, size: int, *, push_bytes: bytes, uniform_blob: bytes,
                 binds: dict, output_image, bindless=None) -> None:
        """Bind the scene material resources (``binds``: name → native SlangPy
        ``Buffer``/``Texture``/``Sampler``), the ``fc`` uniform block, the ``pc``
        push block, the optional bindless texture array, and the preview output
        image (``previewOutput``), then dispatch ``previewMain`` over
        ``size × size`` threads. Names absent from the reflected globals are
        skipped. ``bindless`` is ``(global_name, [native_texture | None, …])``;
        ``None`` slots bind the default 1×1 texture (Metal requires every slot
        bound)."""
        spy = self._spy
        dev = self.ctx.device
        ro = dev.create_root_shader_object(self.program)
        cur = spy.ShaderCursor(ro)

        if "fc" in self.global_names:
            cur["fc"].set_data(
                np.frombuffer(bytes(uniform_blob), dtype=np.uint8).copy())
        cur["pc"].set_data(
            np.frombuffer(bytes(push_bytes), dtype=np.uint8).copy())
        cur["previewOutput"] = output_image

        for name, native in binds.items():
            if native is None or name == "previewOutput" \
                    or name not in self.global_names:
                continue
            cur[name] = native

        if bindless is not None:
            name, textures = bindless
            if name in self.global_names:
                slot_cur = cur[name]
                for i in range(BINDLESS_TEXTURE_CAPACITY):
                    tex = textures[i] if i < len(textures) and textures[i] is not None \
                        else self._default_tex
                    slot_cur[i] = tex

        enc = dev.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.bind_pipeline(self.pipeline, ro)
        # Thread count (slang-rhi divides by numthreads(8,8,1)) — mirrors the
        # megakernel dispatch which passes pixel dims, not group counts.
        cpass.dispatch([int(size), int(size), 1])
        cpass.end()
        dev.submit_command_buffer(enc.finish())
        dev.wait_for_idle()

    def destroy(self) -> None:
        self.pipeline = None
        self.program = None
        self._default_tex = None


class DebugRasterMetal:
    """Metal Camera Debug software rasteriser host (change metal-tool-dock-render
    P2, task 3.1). The native Metal backend has no graphics pipeline, so the
    debug viewport's line vertex stream is scan-converted in a compute kernel.

    Phase-1 slice: a DDA line rasteriser with **no depth** yet (opaque
    last-writer-wins). Compiles ``debug_raster.slang`` and drives two compute
    kernels — ``clearImage`` (fill the packed-RGBA8 output with the background)
    then ``rasterLines`` (one thread per line). The output is a width*height
    ``uint`` buffer packed ``r | g<<8 | b<<16 | a<<24``; :meth:`render` returns it
    as a little-endian RGBA8 byte stream. Kernel math mirrors
    :mod:`skinny.debug_raster_ref` so it is host-checkable and GPU-diffable.

    Buffers are reused across frames (grown as needed) so a steady debug view
    does not churn allocations. Each dispatch is bounded (clear = one thread per
    pixel; lines = one thread per line, per-line pixel loop capped at
    ``max_steps``) so no single command buffer can exceed the macOS GPU watchdog.
    """

    _VERTEX_FLOATS = 7

    def __init__(self, ctx, shader_dir) -> None:
        self.ctx = ctx
        self._spy = ctx._spy
        self.shader_dir = Path(shader_dir)
        dev = ctx.device
        spy = self._spy
        opts = spy.SlangCompilerOptions()
        opts.include_paths = [self.shader_dir]
        opts.defines = {"SKINNY_METAL": "1"}
        session = dev.create_slang_session(compiler_options=opts)
        src = self.shader_dir / "debug_raster.slang"
        module = session.load_module_from_source(
            "debug_raster", src.read_text(encoding="utf-8"), str(src))

        def _kernel(entry):
            return dev.create_compute_kernel(
                session.link_program([module], [module.entry_point(entry)]))

        self._clear_color = _kernel("clearImage")
        self._clear_depth = _kernel("clearDepth")
        self._depth_lines = _kernel("depthLines")
        self._color_lines = _kernel("colorLines")
        self._blend_tris = _kernel("blendTris")

        self._color = None   # packed-RGBA8 output (uint per pixel)
        self._depth = None   # packed depth (uint per pixel)
        self._color_px = 0
        self._lverts = None  # line vertex StorageBuffer (float32)
        self._tverts = None  # triangle vertex StorageBuffer (float32)

    @staticmethod
    def _grow(buf, nbytes, ctx):
        nbytes = max(int(nbytes), 16)
        if buf is None or buf.size < nbytes:
            if buf is not None:
                buf.destroy()
            return StorageBuffer(ctx, nbytes)
        return buf

    def render(self, line_floats, tri_floats, view_proj, width: int, height: int,
               *, max_steps: int = 1 << 16) -> bytes:
        """Rasterise the line + triangle streams → RGBA8 bytes (``width*height*4``,
        row 0 = top). ``view_proj`` is the math-form 4x4 (row-major, clip =
        VP·[x,y,z,1]); ``line_floats`` / ``tri_floats`` are flat float sequences
        (``_VERTEX_FLOATS`` per vertex; 2/line, 3/triangle). Lines are opaque and
        depth-ordered; triangles alpha-blend over them (depth-tested, no depth
        write), mirroring :func:`skinny.debug_raster_ref.rasterise`."""
        from skinny.debug_raster_ref import CLEAR_RGBA8
        dev = self.ctx.device
        w, h = int(width), int(height)
        px = w * h
        self._color = self._grow(self._color, px * 4, self.ctx)
        self._depth = self._grow(self._depth, px * 4, self.ctx)
        self._color_px = px
        r, g, b, a = CLEAR_RGBA8
        clear_packed = int(r | (g << 8) | (b << 16) | (a << 24)) & 0xFFFFFFFF
        vp = np.ascontiguousarray(view_proj, dtype=np.float32).reshape(4, 4)
        vp_vars = {"gVP0": vp[0].tolist(), "gVP1": vp[1].tolist(),
                   "gVP2": vp[2].tolist(), "gVP3": vp[3].tolist()}

        # Clear colour + depth.
        self._clear_color.dispatch(
            thread_count=[w, h, 1],
            vars={"colorOut": self._color.buffer, "gWidth": w, "gHeight": h,
                  "gClearPacked": [clear_packed, 0, 0, 0]})
        self._clear_depth.dispatch(
            thread_count=[w, h, 1],
            vars={"depthOut": self._depth.buffer, "gWidth": w, "gHeight": h})
        dev.wait_for_idle()

        lverts = np.ascontiguousarray(line_floats, dtype=np.float32).ravel()
        n_lines = int(lverts.size // (self._VERTEX_FLOATS * 2))
        if n_lines > 0:
            self._lverts = self._grow(self._lverts, lverts.nbytes, self.ctx)
            self._lverts.upload_sync(lverts.tobytes())
            common = {"lineVerts": self._lverts.buffer,
                      "colorOut": self._color.buffer, "depthOut": self._depth.buffer,
                      "gWidth": w, "gHeight": h, "gCount": n_lines,
                      "gMaxSteps": int(max_steps), **vp_vars}
            # Pass 1 (depth) then pass 2 (colour) — depth must be complete before
            # the colour pass re-reads the winning depth.
            self._depth_lines.dispatch(thread_count=[n_lines, 1, 1], vars=common)
            dev.wait_for_idle()
            self._color_lines.dispatch(thread_count=[n_lines, 1, 1], vars=common)
            dev.wait_for_idle()

        tverts = np.ascontiguousarray(tri_floats, dtype=np.float32).ravel()
        n_tris = int(tverts.size // (self._VERTEX_FLOATS * 3))
        if n_tris > 0:
            self._tverts = self._grow(self._tverts, tverts.nbytes, self.ctx)
            self._tverts.upload_sync(tverts.tobytes())
            # One thread per (triangle, screen-row): each walks <= width pixels.
            self._blend_tris.dispatch(
                thread_count=[n_tris, h, 1],
                vars={"triVerts": self._tverts.buffer,
                      "colorOut": self._color.buffer, "depthOut": self._depth.buffer,
                      "gWidth": w, "gHeight": h, "gCount": n_tris, **vp_vars})
            dev.wait_for_idle()

        return self._color.download_sync(px * 4)

    def destroy(self) -> None:
        for name in ("_color", "_depth", "_lverts", "_tverts"):
            buf = getattr(self, name, None)
            if buf is not None:
                buf.destroy()
                setattr(self, name, None)
        self._clear_color = None
        self._clear_depth = None
        self._depth_lines = None
        self._color_lines = None
        self._blend_tris = None


class MetalFrameEncoder:
    """Single-frame, multi-pass compute command encoder (design D3) — the Metal
    sibling of recording the whole wavefront bounce loop into one Vulkan command
    buffer.

    Accumulates a sequence of compute dispatches into **one** slang-rhi command
    encoder, inserting a global compute-memory barrier between stages (so stage
    N+1 observes stage N's buffer writes — the analogue of the Vulkan
    ``COMPUTE→COMPUTE`` pipeline barrier), and submits + drains **once** at
    :meth:`submit`. This replaces the per-stage ``wait_for_idle`` of the
    single-shot :meth:`ComputePipeline.dispatch_kernel` so a 6-bounce loop is not
    serialized on the GPU idle each stage.

    Parameter binding stays ``set_data``-only for the uniform block (D4 fence-hang
    discipline); resources bind by name on each stage's root object. The Metal
    wavefront recorder (phase 3) drives the bounce loop through this surface;
    :meth:`flush` lets it submit + drain mid-frame when the indirect CPU-readback
    fallback (task 1.3) must read a GPU-written args buffer back to the host.
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self._spy = ctx._spy
        self._enc = ctx.device.create_command_encoder()
        self._submitted = False

    def _root(self, pipe: ComputePipeline, bindings, uniform_blob,
              uniforms=None, bindless=None):
        ro = self.ctx.device.create_root_shader_object(pipe.program)
        cur = self._spy.ShaderCursor(ro)
        if uniform_blob is not None:
            cur["fc"].set_data(
                np.frombuffer(bytes(uniform_blob), dtype=np.uint8).copy())
        # Additional uniform blocks (e.g. the wavefront per-tile constants that
        # are Vulkan push constants) — byte blobs via set_data only (design D4).
        # A `ConstantBuffer<T>` global cannot bind bytes directly; dereference
        # to its element first (plain `uniform T` cursors dereference to
        # themselves, so this is safe for both declaration styles).
        for name, blob in (uniforms or {}).items():
            if blob is None or name not in pipe.global_names:
                continue
            field = cur[name]
            deref = field.dereference()
            target = deref if deref.is_valid() else field
            target.set_data(np.frombuffer(bytes(blob), dtype=np.uint8).copy())
        for name, res in (bindings or {}).items():
            if res is None or name not in pipe.global_names:
                continue
            cur[name] = getattr(res, "buffer", getattr(res, "texture", res))
        # Bindless texture array: (global_name, [native_texture | None, …],
        # default_texture) — every slot must be bound on Metal, so None slots get
        # the default 1×1 texture (mirrors ComputePipeline.dispatch).
        if bindless is not None:
            name, textures, default_tex = bindless
            if name in pipe.global_names:
                slot_cur = cur[name]
                for i in range(BINDLESS_TEXTURE_CAPACITY):
                    tex = textures[i] if i < len(textures) and textures[i] is not None \
                        else default_tex
                    slot_cur[i] = tex
        return ro

    def dispatch(self, pipe: ComputePipeline, groups, *, bindings=None,
                 uniform_blob=None, uniforms=None, bindless=None) -> None:
        """Encode a direct dispatch of ``pipe`` over the ``(x, y, z)`` thread-group
        count ``groups`` into the shared encoder (no submit)."""
        ro = self._root(pipe, bindings, uniform_blob, uniforms, bindless)
        cpass = self._enc.begin_compute_pass()
        cpass.bind_pipeline(pipe.pipeline, ro)
        cpass.dispatch_compute(self._spy.math.uint3(*(int(g) for g in groups)))
        cpass.end()

    def dispatch_indirect(self, pipe: ComputePipeline, args_buffer, offset: int = 0,
                          *, bindings=None, uniform_blob=None, uniforms=None,
                          bindless=None) -> None:
        """Encode a native indirect dispatch of ``pipe`` into the shared encoder.

        Native path only — the CPU-readback fallback needs a host round-trip on the
        args buffer, which breaks single-encoder accumulation; the recorder reads
        the count via :meth:`flush` and falls back to :meth:`dispatch` itself when
        ``ctx.supports_indirect_dispatch`` is false (task 1.3)."""
        ro = self._root(pipe, bindings, uniform_blob, uniforms, bindless)
        native_args = getattr(args_buffer, "buffer", args_buffer)
        cpass = self._enc.begin_compute_pass()
        cpass.bind_pipeline(pipe.pipeline, ro)
        cpass.dispatch_compute_indirect(
            self._spy.BufferOffsetPair(native_args, int(offset)))
        cpass.end()

    def clear(self, buffer) -> None:
        """Encode a full-buffer zero-fill (the Metal analogue of
        ``vkCmdFillBuffer(.., 0)`` the wavefront count/cursor clears use)."""
        self._enc.clear_buffer(getattr(buffer, "buffer", buffer))

    def barrier(self) -> None:
        """Global compute-memory barrier between stages (design D3) — makes every
        prior stage's writes visible to subsequent reads in this encoder."""
        self._enc.global_barrier()

    def flush(self) -> None:
        """Submit + drain the accumulated work and reopen a fresh encoder, without
        ending the frame. Used so a GPU-written args buffer is visible to a host
        readback mid-frame (the indirect fallback)."""
        self.ctx.device.submit_command_buffer(self._enc.finish())
        self.ctx.device.wait_for_idle()
        self._enc = self.ctx.device.create_command_encoder()

    def submit(self) -> None:
        """Submit the whole frame's encoder once and drain the device."""
        if self._submitted:
            return
        self.ctx.device.submit_command_buffer(self._enc.finish())
        self.ctx.device.wait_for_idle()
        self._submitted = True
