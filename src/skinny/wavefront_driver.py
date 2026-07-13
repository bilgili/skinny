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

    def flush_heavy_eye(self) -> None:
        """Bound the heavy per-tile eye submit (change
        wavefront-nonflat-tiled-fallback). When the scene has a non-terminal
        non-flat material (VOLUME / PYTHON), whose non-flat first-hit path
        fallback runs the full multi-bounce ``PathTracer.estimateRadiance`` in the
        eye kernel, the Metal backend submits + drains the accumulated command
        buffer here so no single command buffer runs the fallback over more than
        one ``stream_size`` tile (the macOS GPU watchdog bound, metal
        row-band discipline). A no-op on Vulkan (no watchdog) and whenever the
        scene has no non-terminal non-flat material (byte-identical single
        submit)."""
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
        # Bound the heavy per-tile eye submit: the non-flat first-hit path
        # fallback in wfBdptWalk / wfBdptGenEye runs a full multi-bounce path for
        # VOLUME / PYTHON, so on Metal commit this tile before the next so no
        # single command buffer exceeds the GPU watchdog (no-op otherwise).
        rec.flush_heavy_eye()


def record_sppm_loop(
    rec,
    *,
    num_pixels: int,
    stream_size: int,
    num_cells: int,
    photons: int,
    first_frame: bool,
    photon_batch: int = 0,
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
        # Bound the heavy per-tile eye submit (see record_bdpt_loop): wfSppmEye's
        # non-flat first-hit path fallback runs a full multi-bounce path for
        # VOLUME / PYTHON. Phase 1 is otherwise all dispatch_full with no indirect
        # shade (no implicit flush), so without this every eye tile would
        # accumulate into one command buffer. No-op off Metal / on terminal-only
        # scenes.
        rec.flush_heavy_eye()
    rec.barrier()
    # Bound the SPPM command buffers under the macOS GPU watchdog (Metal only;
    # no-op on Vulkan): without a flush at each phase boundary the whole pass —
    # every eye tile + grid + the entire photon pass + updates — commits as one
    # command buffer, which wedges the GPU on a heavy (caustic / spectral) scene
    # (change spectral-wavefront). Isolate each phase into its own submission.
    rec.flush()

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
    rec.flush()

    # phase 3 — global photon pass. The heaviest work: one thread per photon,
    # each depositing into every visible point within radius (spectral recolor
    # per λ). A caustic scene clusters visible points into the focus cell, so a
    # single command buffer of all `photons` would run photons × VPs-in-cell and
    # wedge the macOS GPU watchdog. Tile the dispatch by breadth into flushed
    # sub-batches (change sppm-photon-dispatch-tiling): each command buffer traces
    # `photon_batch` photons at base `[0, batch, 2·batch, …]` (the shader reads
    # `pid = streamBase + tid.x`). `clear_accum` runs ONCE before the loop — the
    # deposits are additive atomics, so batching is bit-exact vs one dispatch and
    # never starves the photon budget. `photon_batch <= 0` (Vulkan / no watchdog)
    # is the degenerate single full-photon dispatch, base 0.
    rec.clear_accum()
    rec.barrier()
    # `dispatch_count` rounds the launch up to a multiple of the 64-wide
    # threadgroup, and the photon kernel is bounded only by the GLOBAL guard
    # `pid >= sppmPhotonsEmitted`. So a NON-final batch whose count is not
    # 64-aligned would over-launch threads with `pid ∈ [base+n, base+ceil64(n))`
    # that are all < photons (hence unmasked) and ALSO belong to the next batch —
    # double-depositing those photons (energy bias). Align the batch to 64 so
    # every non-final batch is exactly `batch` photons (no round-up); only the
    # final batch's tail rounds up, and that tail satisfies `pid >= photons` and
    # is masked. `photon_batch <= 0` = single full dispatch (round-up masked).
    if photon_batch and photon_batch > 0:
        batch = max(64, (int(photon_batch) // 64) * 64)
    else:
        batch = int(photons)
    base = 0
    while base < photons:
        n = min(batch, photons - base)
        rec.push_tile(base)
        rec.dispatch_count("wfSppmPhotonTrace", n, 64)
        rec.barrier()
        rec.flush()
        base += n

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
