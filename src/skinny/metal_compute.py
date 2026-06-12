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
# branch (`Texture2D<float4> flatMaterialTextures[120]`).
BINDLESS_TEXTURE_CAPACITY = 120

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

    ``external`` is accepted for signature parity but is a no-op on Metal (the
    shared-storage/MTLSharedEvent handoff is a later phase; capability flags are
    ``False``). A host shadow backs ``upload_range`` so partial writes compose."""

    def __init__(self, ctx, size_bytes: int, *, indirect: bool = False,
                 external: bool = False) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.size = max(int(size_bytes), 16)  # GPU drivers dislike zero-sized buffers
        self.external = False  # no Metal shared-storage export yet (P1 D5)
        self._shadow = bytearray(self.size)
        self.buffer = ctx.device.create_buffer(
            size=self.size,
            usage=_buffer_usage(spy, indirect=indirect),
            memory_type=spy.MemoryType.device_local,
            label="skinny.storage",
        )

    def _flush(self) -> None:
        self.buffer.copy_from_numpy(
            np.frombuffer(bytes(self._shadow), dtype=np.uint8).copy())

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
        self._flush()

    def download_sync(self, byte_count: int) -> bytes:
        self.ctx.device.wait_for_idle()
        n = min(int(byte_count), self.size)
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
                 compile_pipeline: bool = True) -> None:
        self.ctx = ctx
        self._spy = ctx._spy
        self.shader_dir = Path(shader_dir)
        self.entry_module = entry_module
        self.entry_point = entry_point
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
        # Reflected MSL layout of each per-graph `StructuredBuffer<GraphParams_*>`
        # (bindings GRAPH_BINDING_BASE+i). Maps graph target_name →
        # ({field: (msl_offset, msl_size)}, msl_stride). Same float3→16 B MSL
        # padding hazard as `mtlx_skin_layout`: the generated GraphParams structs
        # carry raw `float3` fields at scalar offsets, so the renderer relocates
        # the scalar-packed record into this layout per slot
        # (`_upload_graph_param_buffers`). Empty until `_build` reflects them.
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
        opts.defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"}
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
        """Reflect the MSL field offsets/size + element stride of each per-graph
        ``StructuredBuffer<GraphParams_*>`` (global ``graphParams_<sanitized>``).
        Stores ``graph_param_layouts[target_name] = ({field: (offset, size)},
        stride)``. Same rationale as :meth:`_reflect_mtlx_skin_layout`: the
        generated GraphParams structs sit at scalar offsets with raw ``float3``
        fields, which Slang pads to 16 B on Metal — so the renderer repacks the
        scalar record into this MSL layout per slot
        (``_upload_graph_param_buffers``). Graphs whose buffer was dead-stripped
        (e.g. an empty-graph fallback) are simply absent from the map."""
        params = {p.name: p for p in self.program.layout.parameters}
        for gf in self.graph_fragments:
            pname = f"graphParams_{gf.sanitized_name}"
            p = params.get(pname)
            if p is None:
                continue
            etl = getattr(p.type_layout, "element_type_layout", None)
            if etl is None or not getattr(etl, "fields", None):
                continue
            layout = {
                f.name: (int(f.offset), int(getattr(f.type_layout, "size", 0)))
                for f in etl.fields
            }
            stride = int(getattr(etl, "stride", 0) or etl.size)
            self.graph_param_layouts[gf.target_name] = (layout, stride)

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
                 binds: dict, bindless=None) -> None:
        """Bind ``fc`` (via ``set_data``), every resource in ``binds`` (name → native
        SlangPy ``Buffer``/``Texture``/``Sampler``), and the optional bindless texture
        array, then dispatch ``main_pass`` over ``width × height`` threads.

        ``bindless`` is ``(global_name, [native_texture | None, …])``; ``None`` slots
        bind the default 1×1 texture (Metal requires every array slot bound). Only
        names present in the reflected globals are bound (unused ones are
        dead-stripped). No per-field scalar cursor writes anywhere (design D4)."""
        spy = self._spy
        dev = self.ctx.device
        ro = dev.create_root_shader_object(self.program)
        cur = spy.ShaderCursor(ro)

        blob = np.frombuffer(bytes(uniform_blob), dtype=np.uint8).copy()
        cur["fc"].set_data(blob)

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

        enc = dev.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.bind_pipeline(self.pipeline, ro)
        cpass.dispatch([int(width), int(height), 1])
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
