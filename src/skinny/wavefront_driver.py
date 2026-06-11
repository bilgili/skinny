"""Backend-neutral wavefront stage-loop driver (design D1, change metal-wavefront-parity).

The staged wavefront path tracer's *stage order* — the tiled, counting-sorted
bounce loop — is identical on every backend; only the GPU command-recording
primitives differ (Vulkan records into a ``VkCommandBuffer`` with
``vkCmdDispatch``/``vkCmdDispatchIndirect``/``vkCmdPipelineBarrier``; Metal
encodes compute passes into a slang-rhi ``CommandEncoder`` with
``global_barrier``). This module holds the loop **once** and drives it through
the :class:`WavefrontRecorder` protocol, so each backend supplies only a thin
adapter that implements the primitive operations.

The Vulkan adapter lives in :mod:`skinny.vk_wavefront` (``_VkPathRecorder``) and
reproduces the prior inline ``WavefrontPathPass.record_dispatch`` byte-for-byte;
the Metal adapter (a later phase) implements the same protocol on slang-rhi.

This module imports no GPU backend — it is pure control flow over the protocol,
which keeps the stage order backend-agnostic and unit-testable with a recording
stub.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WavefrontRecorder(Protocol):
    """Primitive GPU operations the path loop sequences, one per backend.

    A recorder owns the per-frame recording target (a command buffer / encoder)
    plus the bound scene state, and exposes the pass's pipelines + queue buffers
    through these named operations. ``stream_size`` (the per-tile lane count) is
    a property so the loop can advance the tile base without threading it through
    every call.
    """

    @property
    def stream_size(self) -> int:
        """Lanes per tile — the path-state buffer's slot count."""
        ...

    @property
    def has_neural(self) -> bool:
        """Whether a neural-proposal pre-pass is attached (run every bounce)."""
        ...

    @property
    def has_restir(self) -> bool:
        """Whether a ReSTIR DI reuse plugin is attached (run at bounce 0)."""
        ...

    def barrier(self) -> None:
        """Compute→compute memory barrier (+ indirect-args read visibility)."""
        ...

    def clear_counts(self) -> None:
        """Zero the per-slot count + cursor buffers, with a barrier so the next
        compute stage sees the cleared values."""
        ...

    def push_tile(self, stream_base: int) -> None:
        """Set the per-tile constants ``{streamBase, shadeSlot=0, streamSize}``."""
        ...

    def dispatch_full(self, entry: str) -> None:
        """Bind ``entry`` and dispatch one thread per lane over the whole stream."""
        ...

    def dispatch_one(self, entry: str) -> None:
        """Bind ``entry`` and dispatch a single workgroup (the counts→args kernel)."""
        ...

    def shade(self, slot: int, entry: str) -> None:
        """Set ``shadeSlot=slot``, bind ``entry``, and dispatch indirectly over
        that slot's compacted queue (the per-material shade)."""
        ...

    def neural_prepass(self) -> None:
        """Run the attached neural-proposal pre-pass for the current bounce."""
        ...

    def restir_primary_direct(self) -> None:
        """Run the attached ReSTIR DI primary-direct pass (bounce 0 only)."""
        ...


def record_path_loop(
    rec: WavefrontRecorder,
    *,
    num_pixels: int,
    stream_size: int,
    max_bounces: int,
    build_catchall: bool,
) -> None:
    """Record the tiled, counting-sorted wavefront path-tracing bounce loop.

    Per tile: ``generate`` → for each bounce ``{ intersect (trace + classify +
    count) → build_args (counts → offsets + indirect args) → scatter (lanes →
    per-slot queues) → [neural pre-pass] → [ReSTIR primary-direct at bounce 0] →
    per-material shade dispatched indirectly over each slot's queue }`` →
    ``resolve``. The shade dispatches cover only their slot's lanes (coherence);
    path-state VRAM stays bounded by ``stream_size``.

    This is the single source of truth for the stage order shared by every
    backend; the recorder supplies the primitives. It must stay behaviourally
    identical to the historical inline Vulkan ``record_dispatch``.
    """
    stream_base = 0
    first = True
    while stream_base < num_pixels:
        if not first:
            rec.barrier()  # prior tile's resolve before reusing the buffers
        first = False
        rec.push_tile(stream_base)
        rec.dispatch_full("wfPathGenerate")
        for bounce in range(max_bounces):
            rec.clear_counts()
            rec.barrier()
            rec.dispatch_full("wfPathIntersect")  # trace + classify + count
            rec.barrier()
            rec.dispatch_one("wfBuildArgs")        # counts → offsets + args
            rec.barrier()
            rec.dispatch_full("wfScatter")         # lanes → per-slot queues
            rec.barrier()
            # Neural-proposal pre-pass: forward-sample every live lane into the
            # neural buffer the flat shade reads. Binds its own pipeline layout,
            # so the per-tile constants are restored afterwards.
            if rec.has_neural:
                rec.neural_prepass()
                rec.barrier()
                rec.push_tile(stream_base)
            # ReSTIR DI reuse hook: at the primary vertex, compute primary direct
            # (into the path-state radiance) before shade, whose depth-0
            # reuseDirect is gated to zero. Binds a different pipeline layout, so
            # restore the per-tile constants after.
            if bounce == 0 and rec.has_restir:
                rec.restir_primary_direct()
                rec.barrier()
                rec.push_tile(stream_base)
            rec.shade(0, "wfPathShadeFlat")        # slot 0 (flat)
            if build_catchall:
                rec.barrier()
                rec.shade(1, "wfPathShade")        # slot 1 (non-flat catch-all)
        rec.barrier()
        rec.dispatch_full("wfPathResolve")
        stream_base += stream_size
