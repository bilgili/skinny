"""Recording GPU adapter — the third sibling of ``vk_compute`` / ``metal_compute``.

It **records, it does not simulate** (change ``gpu-backend-adapter``, design D4).
Every allocation, upload, binding and dispatch appends one :class:`Call` to the
context's :class:`Recorder`; readbacks return zero-filled data. It never
attempts to produce pixels, so it can verify **ordering and binding coverage** —
exactly what a 40-branch backend sprawl endangers — and can never be mistaken
for a radiometric check, which the parity matrix owns.

Because it needs no device, it turns questions that used to require a
dual-device host and a guarded runner into plain hostless assertions::

    from skinny import recording_compute as rec

    ctx = rec.RecordingContext(64, 64)
    buf = rec.StorageBuffer(ctx, 4096)
    pipe = rec.ComputePipeline(ctx, shader_dir, "main_pass", "mainImage")
    pipe.reflect_globals({"outputBuffer", "sceneBuffer"})
    pipe.dispatch(64, 64, binds={"outputBuffer": buf})

    assert ctx.recorder.dispatch_entries() == ["mainImage"]
    assert ctx.recorder.missing_bindings() == [("mainImage", "sceneBuffer")]

The public surface mirrors the two device adapters, modulo
``gpu_backend.ONE_SIDED_MEMBERS`` — the conformance test in
``tests/test_gpu_backend.py`` fails if it drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skinny.gpu_backend import RECORDING_CAPABILITIES

__all__ = [
    "BINDLESS_TEXTURE_CAPACITY",
    "Call",
    "Recorder",
    "RecordingContext",
    "ComputePipeline",
    "HostStorageBuffer",
    "HudOverlay",
    "PreviewPipeline",
    "ReadbackBuffer",
    "SampledImage",
    "SampledImage3D",
    "StorageBuffer",
    "StorageImage",
    "UniformBuffer",
]

#: Mirrors the capability record, which is the single owner of this number.
BINDLESS_TEXTURE_CAPACITY = RECORDING_CAPABILITIES.bindless_texture_capacity


@dataclass(frozen=True)
class Call:
    """One recorded operation.

    ``op`` is the method that ran (``"alloc"``, ``"upload"``, ``"dispatch"``,
    ``"read"``, ``"destroy"``, …), ``target`` names the resource or pipeline,
    and ``detail`` carries the arguments worth asserting on (sizes, group
    counts, binding names).
    """

    op: str
    target: str
    detail: dict = field(default_factory=dict)


class Recorder:
    """The ordered log of everything a recording context was asked to do."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    def record(self, op: str, target: str, **detail) -> None:
        self.calls.append(Call(op, target, detail))

    # ── queries ────────────────────────────────────────────────────────
    def ops(self, op: str | None = None) -> list[Call]:
        """Every call, or every call whose ``op`` matches."""
        return [c for c in self.calls if op is None or c.op == op]

    def dispatch_entries(self) -> list[str]:
        """Dispatched entry points, in dispatch order — the pass sequence."""
        return [c.detail.get("entry", c.target) for c in self.ops("dispatch")]

    def allocations(self) -> list[tuple[str, int]]:
        """``(kind, size_bytes)`` for every allocation, in allocation order."""
        return [(c.target, int(c.detail.get("size", 0))) for c in self.ops("alloc")]

    def missing_bindings(self) -> list[tuple[str, str]]:
        """``(entry, global_name)`` for every shader global a recorded dispatch
        left unbound (task 4.3).

        A dispatch is checked only when its pipeline declared its reflected
        globals via :meth:`ComputePipeline.reflect_globals`; an undeclared
        pipeline reports nothing rather than reporting everything.
        """
        gaps: list[tuple[str, str]] = []
        for c in self.ops("dispatch"):
            reflected = c.detail.get("reflected")
            if not reflected:
                continue
            bound = set(c.detail.get("bound") or ())
            gaps += [(c.detail.get("entry", c.target), name)
                     for name in sorted(set(reflected) - bound)]
        return gaps

    def clear(self) -> None:
        self.calls.clear()

    def __len__(self) -> int:
        return len(self.calls)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return "\n".join(f"{c.op:9} {c.target:28} {c.detail}" for c in self.calls)


class RecordingContext:
    """A device-free context. Duck-types the two real contexts closely enough
    for the resource constructors, and carries the :class:`Recorder`."""

    backend_name = "recording"
    is_metal = False
    #: The real contexts expose this; `MetalContext` sets it to `None` rather
    #: than omitting it, which is the whole reason `hasattr` was never a
    #: backend test. Kept here so the mistake stays impossible to repeat.
    compute_queue = None
    supports_shared_memory = False
    supports_indirect_dispatch = True

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.width = int(width)
        self.height = int(height)
        self.recorder = Recorder()

    def wait_idle(self) -> None:
        self.recorder.record("wait_idle", "context")

    def destroy(self) -> None:
        self.recorder.record("destroy", "context")


class _Resource:
    """Shared bookkeeping: record the allocation, record the teardown."""

    def __init__(self, ctx, **detail) -> None:
        self.ctx = ctx
        self._rec = ctx.recorder
        self._rec.record("alloc", type(self).__name__, **detail)

    def destroy(self) -> None:
        self._rec.record("destroy", type(self).__name__)


class StorageBuffer(_Resource):
    def __init__(self, ctx, size_bytes: int, *, indirect: bool = False,
                 external: bool = False, shared: bool = False) -> None:
        self.size = int(size_bytes)
        self.buffer = self
        super().__init__(ctx, size=self.size, indirect=indirect,
                         external=external, shared=shared)

    def upload_sync(self, data: bytes) -> None:
        self._rec.record("upload", "StorageBuffer", bytes=len(bytes(data)))

    def upload_range(self, data: bytes, dst_offset: int = 0) -> None:
        self._rec.record("upload", "StorageBuffer",
                         bytes=len(bytes(data)), offset=int(dst_offset))

    def download_sync(self, byte_count: int | None = None) -> bytes:
        n = self.size if byte_count is None else int(byte_count)
        self._rec.record("read", "StorageBuffer", bytes=n)
        return b"\x00" * n

    def fill_zero_sync(self) -> None:
        self._rec.record("fill_zero", "StorageBuffer", bytes=self.size)

    def export_handle(self) -> int:
        self._rec.record("export_handle", "StorageBuffer")
        return 0


class HostStorageBuffer(_Resource):
    def __init__(self, ctx, size_bytes: int) -> None:
        self.size = int(size_bytes)
        self.buffer = self
        super().__init__(ctx, size=self.size)

    def write(self, data: bytes, offset: int = 0) -> None:
        self._rec.record("upload", "HostStorageBuffer",
                         bytes=len(bytes(data)), offset=int(offset))

    def read(self, length: int | None = None, offset: int = 0) -> bytes:
        n = self.size if length is None else int(length)
        self._rec.record("read", "HostStorageBuffer", bytes=n, offset=int(offset))
        return b"\x00" * n


class UniformBuffer(_Resource):
    def __init__(self, ctx, size_bytes: int) -> None:
        self.size = int(size_bytes)
        self.buffer = self
        super().__init__(ctx, size=self.size)

    def upload(self, data: bytes) -> None:
        self._rec.record("upload", "UniformBuffer", bytes=len(bytes(data)))


class StorageImage(_Resource):
    def __init__(self, ctx, width: int, height: int, format="rgba32_float",
                 transfer_src: bool = False) -> None:
        self.width, self.height = int(width), int(height)
        self.format = format
        self.image = self.view = self
        super().__init__(ctx, width=self.width, height=self.height,
                         format=format, transfer_src=transfer_src)

    def read_rgba(self) -> np.ndarray:
        self._rec.record("read", "StorageImage",
                         width=self.width, height=self.height)
        return np.zeros((self.height, self.width, 4), dtype=np.float32)


class SampledImage(_Resource):
    def __init__(self, ctx, width: int, height: int, format="rgba32_float",
                 bytes_per_pixel: int = 16, address_mode_u="repeat",
                 address_mode_v="clamp") -> None:
        self.width, self.height = int(width), int(height)
        self.format = format
        self.image = self.view = self.sampler = self.texture = self
        super().__init__(ctx, width=self.width, height=self.height,
                         format=format, bytes_per_pixel=int(bytes_per_pixel),
                         address_mode_u=address_mode_u,
                         address_mode_v=address_mode_v)

    def upload_sync(self, data) -> None:
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        self._rec.record("upload", "SampledImage", bytes=len(raw))


class SampledImage3D(_Resource):
    def __init__(self, ctx, width: int, height: int, depth: int) -> None:
        self.width, self.height, self.depth = int(width), int(height), int(depth)
        self.image = self.view = self.sampler = self.texture = self
        super().__init__(ctx, width=self.width, height=self.height,
                         depth=self.depth)

    def upload_sync(self, voxels) -> None:
        arr = np.asarray(voxels)
        self._rec.record("upload", "SampledImage3D", shape=tuple(arr.shape))


class ReadbackBuffer(_Resource):
    def __init__(self, ctx, width: int, height: int,
                 bytes_per_pixel: int = 4) -> None:
        self.width, self.height = int(width), int(height)
        self._bpp = int(bytes_per_pixel)
        super().__init__(ctx, width=self.width, height=self.height,
                         bytes_per_pixel=self._bpp)

    def record_copy_from(self, cmd, src_image) -> None:
        self._rec.record("copy", "ReadbackBuffer")

    def read(self) -> bytes:
        n = self.width * self.height * self._bpp
        self._rec.record("read", "ReadbackBuffer", bytes=n)
        return b"\x00" * n


class HudOverlay(_Resource):
    def __init__(self, ctx, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        self.image = self.view = self
        super().__init__(ctx, width=self.width, height=self.height)

    def upload(self, data: bytes) -> None:
        self._rec.record("upload", "HudOverlay", bytes=len(bytes(data)))

    def record_copy(self, cmd) -> None:
        self._rec.record("copy", "HudOverlay")


class _RecordingPipeline(_Resource):
    """Shared dispatch recording for the two pipeline classes."""

    def __init__(self, ctx, entry_point: str, **detail) -> None:
        self.entry_point = entry_point
        self._reflected: set[str] = set()
        super().__init__(ctx, entry=entry_point, **detail)

    def reflect_globals(self, names) -> None:
        """Declare the shader globals this pipeline reflects.

        A test supplies them (from the real reflection, or by hand) so
        :meth:`Recorder.missing_bindings` can report a dispatch that leaves one
        unbound — the failure that otherwise surfaces only as a black image on a
        device.
        """
        self._reflected = set(names)

    def _record_dispatch(self, groups, binds) -> None:
        self._rec.record(
            "dispatch", type(self).__name__,
            entry=self.entry_point, groups=tuple(int(g) for g in groups),
            bound=sorted((binds or {}).keys()),
            reflected=sorted(self._reflected),
        )


class ComputePipeline(_RecordingPipeline):
    def __init__(self, ctx, shader_dir, entry_module: str, entry_point: str,
                 graph_fragments=None, *, compile_pipeline: bool = True,
                 spectral: bool = False) -> None:
        self.shader_dir = shader_dir
        self.entry_module = entry_module
        self.spectral = bool(spectral)
        super().__init__(ctx, entry_point, module=entry_module,
                         compile_pipeline=bool(compile_pipeline),
                         spectral=self.spectral)

    @classmethod
    def scene_bindings_only(cls, ctx, shader_dir, graph_fragments=None, *,
                            spectral: bool = False):
        return cls(ctx, shader_dir, "main_pass", "mainImage",
                   graph_fragments, compile_pipeline=False, spectral=spectral)

    def dispatch(self, width: int, height: int, uniform_blob=None, binds=None,
                 bindless=None, bands: int = 0, tile_origin_offset: int = 0):
        self._record_dispatch((width, height, 1), binds)

    def dispatch_indirect(self, args_buffer, offset: int = 0, bindings=None):
        self._record_dispatch((0, 0, 0), bindings)

    def dispatch_kernel(self, thread_count, buffers=None, vars=None):
        groups = thread_count if isinstance(thread_count, (tuple, list)) \
            else (thread_count, 1, 1)
        self._record_dispatch(groups, buffers)


class PreviewPipeline(_RecordingPipeline):
    _PUSH_FMT = "<IIIIffff"  # matches vk_compute.PreviewPipeline._PUSH_FMT

    def __init__(self, ctx, shader_dir, graph_fragments=None) -> None:
        self.shader_dir = shader_dir
        super().__init__(ctx, "previewMain")

    @staticmethod
    def pack_push(matId: int, graphId: int, primKind: int, size: int,
                  yaw: float, pitch: float, distance: float,
                  fovTan: float) -> bytes:
        import struct
        return struct.pack(
            PreviewPipeline._PUSH_FMT,
            int(matId), int(graphId), int(primKind), int(size),
            float(yaw), float(pitch), float(distance), float(fovTan),
        )

    def dispatch(self, size: int, push_bytes: bytes, uniform_blob=None,
                 binds=None, output_image=None, bindless=None):
        self._record_dispatch((size, size, 1), binds)
