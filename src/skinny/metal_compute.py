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
    "rgba8_srgb": "rgba8_srgb",
    "r8_unorm": "r8_unorm",
    "r32_float": "r32_float",
}
_VKFORMAT_INTS = {  # VkFormat enum values → token
    9: "r8_unorm", 37: "rgba8_unorm", 43: "rgba8_srgb", 109: "rgba32_float",
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
        self._default_tex = None

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

    def destroy(self) -> None:
        self.pipeline = None
        self._default_tex = None
