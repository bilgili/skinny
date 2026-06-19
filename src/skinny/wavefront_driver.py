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


def record_bdpt_loop(
    rec: WavefrontRecorder,
    *,
    num_pixels: int,
    stream_size: int,
    walk_mode: str,
    eye_bounces: int,
    light_bounces: int,
    slot_nee: int = 0,
    slot_full: int = 1,
) -> None:
    """Record the tiled, fully-staged wavefront BDPT loop (phase 4).

    Per tile: ``build_subpaths`` (per ``walk_mode``: the fused single-kernel
    walk, a staged eye walk + fused light tail, or fully staged eye + light
    walks + standalone splat — each staged bounce is its own counting-sorted
    compaction + indirect dispatch over only the live lanes) → connect
    counting sort (``classify → build_args → scatter``) → indirect connect
    over the NEE then FULL queues → ``resolve``. The eye/light/aux + queue
    buffers are bounded by ``stream_size``, not the pixel count; the
    counting-sort scratch is shared across all the compactions.

    Reuses the :class:`WavefrontRecorder` protocol — ``shade(slot, entry)`` is
    the generic "set slot constant + indirect dispatch over that slot's
    compacted queue" primitive (the BDPT bounce + connect kernels), and the
    neural/ReSTIR hooks are simply never invoked. It must stay behaviourally
    identical to the historical inline Vulkan
    ``WavefrontBdptPass.record_dispatch``.
    """
    if walk_mode not in ("fused", "eye", "eye_light"):
        raise ValueError(f"unknown bdpt walk_mode {walk_mode!r}")

    def compact(classify_entry: str) -> None:
        # clear counts → classify (count) → build_args → scatter, leaving the
        # live lanes gathered into their slot queues for an indirect dispatch.
        rec.clear_counts()
        rec.dispatch_full(classify_entry)
        rec.barrier()
        rec.dispatch_one("wfBdptBuildArgs")
        rec.barrier()
        rec.dispatch_full("wfBdptScatter")
        rec.barrier()

    def build_subpaths() -> None:
        """Dispatch the subpath-construction kernels for the active walk_mode,
        leaving each lane's aux (eyeLen/lightLen/escaped/rngState) ready for
        the shared connect+resolve tail."""
        if walk_mode == "fused":
            rec.dispatch_full("wfBdptWalk")       # eye+light+splat in one kernel
            rec.barrier()
            return
        # staged eye walk (eye + eye_light modes)
        rec.dispatch_full("wfBdptGenEye")         # eye[0..1] + first ray
        rec.barrier()
        for _ in range(eye_bounces):
            compact("wfBdptWalkClassify")         # gather live eye lanes → slot 0
            rec.shade(slot_nee, "wfBdptBounceEye")   # extend one eye vertex
            rec.barrier()
        if walk_mode == "eye":
            rec.dispatch_full("wfBdptLightTail")  # fused light walk + splat
            rec.barrier()
            return
        # eye_light: staged light walk + standalone splat
        rec.dispatch_full("wfBdptGenLight")       # light[0] + first light ray
        rec.barrier()
        for _ in range(light_bounces):
            compact("wfBdptWalkClassify")         # gather live light lanes → slot 0
            rec.shade(slot_nee, "wfBdptBounceLight")  # extend one light vertex
            rec.barrier()
        rec.dispatch_full("wfBdptSplat")          # s=1 light-tracer splat
        rec.barrier()

    stream_base = 0
    first = True
    while stream_base < num_pixels:
        if not first:
            rec.barrier()  # prior tile's resolve before reusing the buffers
        first = False
        rec.push_tile(stream_base)
        build_subpaths()
        compact("wfBdptClassify")                 # route lanes NEE / FULL / dead
        rec.shade(slot_nee, "wfBdptConnectNee")
        rec.barrier()
        rec.shade(slot_full, "wfBdptConnectFull")
        rec.barrier()
        rec.dispatch_full("wfBdptResolve")
        stream_base += stream_size


def record_sppm_loop(
    rec,
    *,
    num_pixels: int,
    stream_size: int,
    num_cells: int,
    photons: int,
    first_frame: bool,
) -> None:
    """Record one SPPM pass (== one progressive-accumulation frame).

    The mandated split ordering (an adversarial-review requirement): the grid +
    photon stages are GLOBAL over every visible point, so they must run AFTER all
    eye tiles and BEFORE any update tile — never interleaved per tile. ``tiles ==
    1`` (num_pixels <= stream_size) is just the degenerate case of this order.

        [frame 0 only] clear the persistent visible-point buffer
        phase 1: all eye tiles            (write every pixel's visible point)
        phase 2: grid build               (clear -> count -> scan -> scatter)
        phase 3: photon pass              (clear accumulator -> emit/trace/deposit)
        phase 4: all update tiles         (reduce + resolve + composite)

    The recorder must supply, beyond the path-loop primitives (``stream_size``,
    ``barrier``, ``push_tile``, ``dispatch_full``, ``dispatch_one``): ``dispatch_count(
    entry, count, group_size)`` (dispatch ceil(count/group_size) workgroups over a
    host-known count — grid/photon stages have no indirect dispatch),
    ``clear_visible_points()``, ``clear_grid()`` (zero gridCount + gridCursor), and
    ``clear_accum()`` (zero the per-pass SppmAccum region).
    """
    # frame 0: zero the persistent visible-point buffer so the n==0
    # first-activation radius init in wfSppmEye is reliable.
    if first_frame:
        rec.clear_visible_points()
        rec.barrier()

    # phase 1 — all eye tiles.
    stream_base = 0
    first = True
    while stream_base < num_pixels:
        if not first:
            rec.barrier()
        first = False
        rec.push_tile(stream_base)
        rec.dispatch_full("wfSppmEye")
        stream_base += stream_size
    rec.barrier()

    # phase 2 — single global grid build (counting sort).
    rec.clear_grid()
    rec.barrier()
    rec.dispatch_count("wfSppmGridCount", num_pixels, 64)
    rec.barrier()
    rec.dispatch_count("wfSppmGridScanBlock", num_cells, 256)
    rec.barrier()
    rec.dispatch_one("wfSppmGridScanBlockSums")
    rec.barrier()
    rec.dispatch_count("wfSppmGridScanAdd", num_cells, 256)
    rec.barrier()
    rec.dispatch_count("wfSppmGridScatter", num_pixels, 64)
    rec.barrier()

    # phase 3 — single global photon pass.
    rec.clear_accum()
    rec.barrier()
    rec.dispatch_count("wfSppmPhotonTrace", photons, 64)
    rec.barrier()

    # phase 4 — all update tiles.
    stream_base = 0
    first = True
    while stream_base < num_pixels:
        if not first:
            rec.barrier()
        first = False
        rec.push_tile(stream_base)
        rec.dispatch_full("wfSppmUpdate")
        stream_base += stream_size
    rec.barrier()
