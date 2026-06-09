"""Minimal Metal compute resource wrappers (foundation phase).

The Metal siblings of the :mod:`skinny.vk_compute` classes the foundation test
uses — :class:`StorageBuffer`, :class:`StorageImage`, :class:`ComputePipeline` —
each wrapping a SlangPy resource and matching the Vulkan constructor signatures.
Scoped to what proves the device foundation (a trivial compute dispatch); the
full resource set (uniforms, bindless textures, samplers, indirect dispatch,
external memory) is staged in later changes alongside the renderer port.

Pipeline-parameter discipline (design D4): parameters are bound either as
resources through ``dispatch(vars=…)`` or, for uniform blocks, via ``set_data``
byte blobs through a :class:`slangpy.ShaderCursor` — **never** per-field cursor
writes. On the Metal path a scalar cursor write around an open encoder can leave
the GPU fence un-signalled (the hang the original ``metal`` branch hit), so this
module never does one. P1's trivial kernel has no uniform block; the byte-blob
helper (:meth:`ComputePipeline.set_param_blob`) is here for the later MSL-correct
uniform packing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StorageBuffer:
    """Device-local storage buffer wrapping a SlangPy ``Buffer``.

    Mirrors ``vk_compute.StorageBuffer(ctx, size_bytes, *, indirect=False,
    external=False)``. ``external`` is accepted for signature parity but is a
    no-op in P1 (Metal shared-storage export lands in a later phase; see design
    D5).
    """

    def __init__(self, ctx, size_bytes: int, *, indirect: bool = False,
                 external: bool = False) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.size = max(int(size_bytes), 16)  # avoid zero-sized GPU buffers
        self.external = False  # P1: no Metal shared-storage export yet
        usage = (
            spy.BufferUsage.unordered_access
            | spy.BufferUsage.shader_resource
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        if indirect:
            usage |= spy.BufferUsage.indirect_argument
        self.buffer = ctx.device.create_buffer(
            size=self.size,
            usage=usage,
            memory_type=spy.MemoryType.device_local,
            label="skinny.storage",
        )

    def upload(self, data: bytes) -> None:
        """Copy ``data`` (bytes) into the buffer."""
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
        self.buffer.copy_from_numpy(arr)

    def download_sync(self) -> bytes:
        """Drain the device and return the buffer's first ``size`` bytes."""
        self.ctx.device.wait_for_idle()
        return self.buffer.to_numpy().tobytes()[: self.size]


class StorageImage:
    """Device-local 2D storage image wrapping a SlangPy ``Texture``.

    Mirrors ``vk_compute.StorageImage(ctx, width, height, format=…,
    transfer_src=False)``. ``format`` is a ``slangpy.Format`` (default
    ``rgba32_float`` to match the Vulkan ``R32G32B32A32_SFLOAT`` accumulation
    image); pass ``None`` for the default.
    """

    def __init__(self, ctx, width: int, height: int, format=None,
                 transfer_src: bool = False) -> None:
        self.ctx = ctx
        spy = ctx._spy
        self.width = int(width)
        self.height = int(height)
        self.format = format if format is not None else spy.Format.rgba32_float
        usage = spy.TextureUsage.unordered_access | spy.TextureUsage.copy_destination
        if transfer_src:
            usage |= spy.TextureUsage.copy_source
        self.texture = ctx.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=self.format,
            width=self.width,
            height=self.height,
            usage=usage,
            memory_type=spy.MemoryType.device_local,
            label="skinny.storage_image",
        )


class ComputePipeline:
    """A single Metal compute pipeline compiled from a Slang entry point.

    Mirrors the ``vk_compute.ComputePipeline(ctx, shader_dir, entry_module,
    entry_point, …)`` signature for the trivial foundation kernel. Loads
    ``{entry_module}.slang`` from ``shader_dir`` and compiles + links it
    in-process via slang-rhi (resolves design O1 — no ``slangc`` shell-out).

    The entry point must not be named ``main``: Slang's Metal target reserves it
    and renames ``main``→``main_0``, which breaks compute-pipeline creation. Use
    e.g. ``computeMain``.
    """

    def __init__(self, ctx, shader_dir, entry_module: str, entry_point: str,
                 graph_fragments=None, *, compile_pipeline: bool = True) -> None:
        self.ctx = ctx
        self.entry_module = entry_module
        self.entry_point = entry_point
        spy = ctx._spy
        path = Path(shader_dir) / f"{entry_module}.slang"
        source = path.read_text(encoding="utf-8")
        module = ctx.device.load_module_from_source(
            entry_module, source, path=str(path)
        )
        program = ctx.device.link_program(
            [module], [module.entry_point(entry_point)]
        )
        self.kernel = ctx.device.create_compute_kernel(program)
        self._spy = spy

    def dispatch(self, thread_count, *, buffers=None, vars=None) -> None:
        """Dispatch the kernel over ``thread_count`` (an int triple).

        ``buffers`` maps a shader global name → a :class:`StorageBuffer` (its
        underlying SlangPy buffer is bound); ``vars`` passes any already-native
        values straight through. Resource binding only — no per-field cursor
        writes (design D4).
        """
        bound = dict(vars or {})
        for name, res in (buffers or {}).items():
            bound[name] = getattr(res, "buffer", getattr(res, "texture", res))
        self.kernel.dispatch(thread_count=list(thread_count), vars=bound)

    def set_param_blob(self, name: str, blob: bytes) -> None:
        """Bind a uniform block ``name`` from a packed byte ``blob`` via
        ``set_data`` (never per-field cursor writes — design D4). Unused by P1's
        trivial kernel; here for the later MSL-correct uniform packing."""
        cursor = self._spy.ShaderCursor(self.kernel.root_object)
        cursor[name].set_data(np.frombuffer(bytes(blob), dtype=np.uint8))
