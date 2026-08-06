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

# ── kernel entry-point names — the single owner (change choice-table-wavefront-owners) ──
#
# Every wavefront compute kernel's Slang entry-point name is declared ONCE here.
# The driver dispatches through these constants; both backend pass modules import
# them for their pipeline `entries` lists (`from skinny.wavefront_driver import
# WF_…`). A rename edits one line and is an import-time failure in every consumer,
# never a runtime dispatch failure on one backend. `tests/test_wavefront_kernel_names.py`
# pins each constant to its historical string and fails if any kernel-name literal
# reappears in the driver or either backend. These are ENTRY-POINT names only — the
# Metal bind-by-name resource names (`wfEye`, `wfSlotCount`, …) are a different
# namespace and are not owned here.

# Path tracer (wavefront/wavefront_path.slang)
WF_PATH_GENERATE = "wfPathGenerate"
WF_PATH_INTERSECT = "wfPathIntersect"
WF_BUILD_ARGS = "wfBuildArgs"
WF_SCATTER = "wfScatter"
WF_PATH_SHADE_FLAT = "wfPathShadeFlat"
WF_PATH_SHADE = "wfPathShade"
WF_PATH_RESOLVE = "wfPathResolve"

# BDPT (wavefront/wavefront_bdpt.slang)
WF_BDPT_BUILD_ARGS = "wfBdptBuildArgs"
WF_BDPT_SCATTER = "wfBdptScatter"
WF_BDPT_WALK = "wfBdptWalk"
WF_BDPT_GEN_EYE = "wfBdptGenEye"
WF_BDPT_WALK_CLASSIFY = "wfBdptWalkClassify"
WF_BDPT_BOUNCE_EYE = "wfBdptBounceEye"
WF_BDPT_LIGHT_TAIL = "wfBdptLightTail"
WF_BDPT_GEN_LIGHT = "wfBdptGenLight"
WF_BDPT_BOUNCE_LIGHT = "wfBdptBounceLight"
WF_BDPT_SPLAT = "wfBdptSplat"
WF_BDPT_CLASSIFY = "wfBdptClassify"
WF_BDPT_CONNECT_NEE = "wfBdptConnectNee"
WF_BDPT_CONNECT_FULL = "wfBdptConnectFull"
WF_BDPT_RESOLVE = "wfBdptResolve"

# SPPM (wavefront/wavefront_sppm.slang)
WF_SPPM_EYE = "wfSppmEye"
WF_SPPM_GRID_COUNT = "wfSppmGridCount"
WF_SPPM_GRID_SCAN_BLOCK = "wfSppmGridScanBlock"
WF_SPPM_GRID_SCAN_BLOCK_SUMS = "wfSppmGridScanBlockSums"
WF_SPPM_GRID_SCAN_ADD = "wfSppmGridScanAdd"
WF_SPPM_GRID_SCATTER = "wfSppmGridScatter"
WF_SPPM_PHOTON_TRACE = "wfSppmPhotonTrace"
WF_SPPM_UPDATE = "wfSppmUpdate"

# MLT (wavefront/wavefront_mlt.slang)
WF_MLT_BOOTSTRAP = "wfMltBootstrap"
WF_MLT_INIT = "wfMltInit"
WF_MLT_MUTATE = "wfMltMutate"
WF_MLT_RESOLVE = "wfMltResolve"

# Neural directional-proposal pre-pass (both backends).
WF_NEURAL_PROPOSAL = "wfNeuralProposal"
# Indirect-args paint (Vulkan only; Metal uses the CPU readback fallback).
WF_INDIRECT_PAINT = "wfIndirectPaint"

#: Every owned kernel entry-point name — the set the source gate forbids as a
#: literal outside this owner.
KERNEL_ENTRY_NAMES: frozenset[str] = frozenset({
    WF_PATH_GENERATE, WF_PATH_INTERSECT, WF_BUILD_ARGS, WF_SCATTER,
    WF_PATH_SHADE_FLAT, WF_PATH_SHADE, WF_PATH_RESOLVE,
    WF_BDPT_BUILD_ARGS, WF_BDPT_SCATTER, WF_BDPT_WALK, WF_BDPT_GEN_EYE,
    WF_BDPT_WALK_CLASSIFY, WF_BDPT_BOUNCE_EYE, WF_BDPT_LIGHT_TAIL,
    WF_BDPT_GEN_LIGHT, WF_BDPT_BOUNCE_LIGHT, WF_BDPT_SPLAT, WF_BDPT_CLASSIFY,
    WF_BDPT_CONNECT_NEE, WF_BDPT_CONNECT_FULL, WF_BDPT_RESOLVE,
    WF_SPPM_EYE, WF_SPPM_GRID_COUNT, WF_SPPM_GRID_SCAN_BLOCK,
    WF_SPPM_GRID_SCAN_BLOCK_SUMS, WF_SPPM_GRID_SCAN_ADD, WF_SPPM_GRID_SCATTER,
    WF_SPPM_PHOTON_TRACE, WF_SPPM_UPDATE,
    WF_MLT_BOOTSTRAP, WF_MLT_INIT, WF_MLT_MUTATE, WF_MLT_RESOLVE,
    WF_NEURAL_PROPOSAL, WF_INDIRECT_PAINT,
})


# ── shared wavefront pass constants (must be equal across backends) ──
#
# The values every backend must agree on: bounce counts, per-stream lane caps,
# the slot count, the walk modes, and the ReSTIR default config. Both backend
# pass modules derive their class attributes from these, so a change lands on
# both at once. NOT here (legitimately per-backend, pinned with a reason in
# tests/test_wavefront_kernel_names.py): the vertex/aux/reservoir strides — a
# real stride on Vulkan but a reflection fallback on Metal (the MSL stride is
# authoritative) — and the record-stack sizing formula, which differs by design.
WF_MAX_BOUNCES = 6              # lockstep with WF_MAX_BOUNCES in the shader
# The path and BDPT counting-sort slot domains are DISTINCT — path routes
# flat(0)/non-flat(1) (wf_shade_common.slang WF_NUM_SLOTS), BDPT routes
# nee(0)/full(1) (wavefront_bdpt.slang WF_BDPT_NUM_SLOTS). Each host constant
# mirrors ONE shader constant; they are independently 2 today, so they get two
# owners, not one (forcing them equal would couple two unrelated shader facts).
WF_NUM_SLOTS = 2               # path: 0 = flat, 1 = non-flat catch-all
WF_BDPT_NUM_SLOTS = 2          # bdpt: 0 = nee, 1 = full
WF_STREAM_CAP_PATH = 1 << 20    # path/sppm max lanes per stream (~68 MB path-state)
WF_STREAM_CAP_BDPT = 1 << 18    # smaller: each lane owns 2×BDPT_MAX_VERTS vertices
BDPT_MAX_VERTS = 7             # lockstep with bdpt.slang BDPT_MAX_VERTS
WF_EYE_BOUNCES = BDPT_MAX_VERTS - 2
WF_LIGHT_BOUNCES = BDPT_MAX_VERTS - 1
#: BDPT subpath-build walk modes — the same tuple the CLI advertises as
#: ``cli_common.WALK_CHOICES`` (pinned equal by the test; kept here so the
#: low-level driver does not import the high-level CLI module).
WALK_MODES = ("fused", "eye", "eye_light")
#: Default ReSTIR DI config (mirrors restir_primary.slang RestirPC). flags bit0
#: spatial, bit1 temporal. Renderer overrides via RestirDiReuse.
RESTIR_DEFAULT_CONFIG = dict(flags=0x3, mLight=8, spatialK=5, spatialRadius=16.0,
                             normalThresh=0.9, depthThresh=0.1, mCap=20, mBsdf=1)


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
        """Mark a tile boundary after a potentially heavy eye dispatch (change
        wavefront-nonflat-tiled-fallback).

        The driver states *where* the boundary is; the recorder decides what to
        do there. A backend whose
        :attr:`~skinny.gpu_backend.BackendCapabilities.needs_watchdog_tiling`
        is set submits + drains the accumulated command buffer, so no single
        command buffer runs the non-flat first-hit fallback — the full
        multi-bounce ``PathTracer.estimateRadiance`` that VOLUME / PYTHON
        materials take in the eye kernel — over more than one ``stream_size``
        tile. Every other backend records a no-op and the frame stays one
        byte-identical submit, as does any scene with no non-terminal non-flat
        material.

        Reassessed under change gpu-backend-adapter (task 5.3) and deliberately
        **kept in this protocol**: the capability record already owns *whether*
        the boundary costs anything, and hoisting the decision into the driver
        (``if caps.needs_watchdog_tiling: rec.flush()``) would both put a
        watchdog test in the backend-neutral loop and drop the per-scene
        ``bound_heavy_eye`` condition the recorder folds in. Like
        :meth:`barrier`, this is a point in the recording that each backend
        interprets."""
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
        rec.dispatch_full(WF_PATH_GENERATE)
        for bounce in range(max_bounces):
            rec.clear_counts()
            rec.barrier()
            rec.dispatch_full(WF_PATH_INTERSECT)  # trace + classify + count
            rec.barrier()
            rec.dispatch_one(WF_BUILD_ARGS)        # counts → offsets + args
            rec.barrier()
            rec.dispatch_full(WF_SCATTER)         # lanes → per-slot queues
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
            rec.shade(0, WF_PATH_SHADE_FLAT)        # slot 0 (flat)
            if build_catchall:
                rec.barrier()
                rec.shade(1, WF_PATH_SHADE)        # slot 1 (non-flat catch-all)
        rec.barrier()
        rec.dispatch_full(WF_PATH_RESOLVE)
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
        rec.dispatch_one(WF_BDPT_BUILD_ARGS)
        rec.barrier()
        rec.dispatch_full(WF_BDPT_SCATTER)
        rec.barrier()

    def build_subpaths() -> None:
        """Dispatch the subpath-construction kernels for the active walk_mode,
        leaving each lane's aux (eyeLen/lightLen/escaped/rngState) ready for
        the shared connect+resolve tail."""
        if walk_mode == "fused":
            rec.dispatch_full(WF_BDPT_WALK)       # eye+light+splat in one kernel
            rec.barrier()
            return
        # staged eye walk (eye + eye_light modes)
        rec.dispatch_full(WF_BDPT_GEN_EYE)         # eye[0..1] + first ray
        rec.barrier()
        for _ in range(eye_bounces):
            compact(WF_BDPT_WALK_CLASSIFY)         # gather live eye lanes → slot 0
            rec.shade(slot_nee, WF_BDPT_BOUNCE_EYE)   # extend one eye vertex
            rec.barrier()
        if walk_mode == "eye":
            rec.dispatch_full(WF_BDPT_LIGHT_TAIL)  # fused light walk + splat
            rec.barrier()
            return
        # eye_light: staged light walk + standalone splat
        rec.dispatch_full(WF_BDPT_GEN_LIGHT)       # light[0] + first light ray
        rec.barrier()
        for _ in range(light_bounces):
            compact(WF_BDPT_WALK_CLASSIFY)         # gather live light lanes → slot 0
            rec.shade(slot_nee, WF_BDPT_BOUNCE_LIGHT)  # extend one light vertex
            rec.barrier()
        rec.dispatch_full(WF_BDPT_SPLAT)          # s=1 light-tracer splat
        rec.barrier()

    stream_base = 0
    first = True
    while stream_base < num_pixels:
        if not first:
            rec.barrier()  # prior tile's resolve before reusing the buffers
        first = False
        rec.push_tile(stream_base)
        build_subpaths()
        compact(WF_BDPT_CLASSIFY)                 # route lanes NEE / FULL / dead
        rec.shade(slot_nee, WF_BDPT_CONNECT_NEE)
        rec.barrier()
        rec.shade(slot_full, WF_BDPT_CONNECT_FULL)
        rec.barrier()
        rec.dispatch_full(WF_BDPT_RESOLVE)
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
        rec.dispatch_full(WF_SPPM_EYE)
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
    rec.dispatch_count(WF_SPPM_GRID_COUNT, num_pixels, 64)
    rec.barrier()
    rec.dispatch_count(WF_SPPM_GRID_SCAN_BLOCK, num_cells, 256)
    rec.barrier()
    rec.dispatch_one(WF_SPPM_GRID_SCAN_BLOCK_SUMS)
    rec.barrier()
    rec.dispatch_count(WF_SPPM_GRID_SCAN_ADD, num_cells, 256)
    rec.barrier()
    rec.dispatch_count(WF_SPPM_GRID_SCATTER, num_pixels, 64)
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
    # Hard per-dispatch ceiling, every backend (change sppm-env-photon-budget):
    # Vulkan only guarantees maxComputeWorkGroupCount[0] >= 65535, so a single
    # dispatch may carry at most 65535 × 64 = 4,194,240 photons — beyond that a
    # driver may silently clamp groupCountX, dropping photons while the update
    # stage still divides by the full emitted count (dim bias). 64-aligned by
    # construction, so the non-final-batch double-deposit hazard above stays
    # closed. The env-aware budget (×8 on env-lit scenes) crosses this at
    # ~724² renders; the flat budget crossed it latently at ~4.2 Mpx.
    batch = min(batch, 65535 * 64)
    base = 0
    while base < photons:
        n = min(batch, photons - base)
        rec.push_tile(base)
        rec.dispatch_count(WF_SPPM_PHOTON_TRACE, n, 64)
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
        rec.dispatch_full(WF_SPPM_UPDATE)
        stream_base += stream_size
    rec.barrier()


def _mlt_chain_batch(num_chains: int, chain_batch: int = 0) -> int:
    """Per-dispatch chain window: 64-aligned (dispatch rounds up to 64-wide
    groups; the window guard masks the tail), never above the portable
    65535 x 64 workgroup-count ceiling, never above ``num_chains``."""
    if chain_batch and chain_batch > 0:
        batch = max(64, (int(chain_batch) // 64) * 64)
    else:
        batch = int(num_chains)
    return min(batch, num_chains, 65535 * 64)


def record_mlt_bootstrap(rec, *, bootstrap_samples: int, num_chains: int,
                         chain_batch: int = 0) -> None:
    """Record the MLT bootstrap phase (change mlt-integrator).

    One thread = one bootstrap sample writing its scalar contribution; the host
    reads the weights back afterwards (CDF, b, chain-seed resample) — that host
    step is NOT part of this recording. Dispatches are breadth-tiled to at most
    ``num_chains`` in-flight slots because the primary-sample buffer doubles as
    bootstrap scratch (each slot owns one X slice), and flushed per sub-batch
    (Metal watchdog; no-op on Vulkan). The recorder must supply ``push_window(
    base, size)`` (the MLT tile push: streamBase + exact streamSize) plus the
    shared ``dispatch_count`` / ``barrier`` / ``flush`` primitives."""
    batch = _mlt_chain_batch(num_chains, chain_batch)
    base = 0
    while base < bootstrap_samples:
        n = min(batch, bootstrap_samples - base)
        rec.push_window(base, n)
        rec.dispatch_count(WF_MLT_BOOTSTRAP, n, 64)
        rec.barrier()
        rec.flush()
        base += n


def record_mlt_init(rec, *, num_chains: int, chain_batch: int = 0) -> None:
    """Record the MLT chain-init phase: replay each chain's resampled bootstrap
    seed (the host uploaded ``mltChainSeeds`` after the bootstrap readback) and
    store its current state. Breadth-tiled + flushed like the bootstrap."""
    batch = _mlt_chain_batch(num_chains, chain_batch)
    base = 0
    while base < num_chains:
        n = min(batch, num_chains - base)
        rec.push_window(base, n)
        rec.dispatch_count(WF_MLT_INIT, n, 64)
        rec.barrier()
        rec.flush()
        base += n


def record_mlt_frame(rec, *, num_pixels: int, num_chains: int, iterations: int,
                     chain_batch: int = 0) -> None:
    """Record one MLT accumulation frame: ``iterations`` Metropolis steps over
    every chain (each step = one proposal + dual splat + accept/reject), then
    the b-normalized resolve folding the frame's splat buffer into the
    accumulation image and clearing it.

    Every mutate dispatch is breadth-tiled + flushed (Metal watchdog; no-op on
    Vulkan); a barrier separates steps so iteration i+1 reads iteration i's
    chain state, and separates the last step from the resolve. ``mpp_actual =
    iterations * num_chains / num_pixels`` is packed by the host into
    ``fc.mltMppActual`` — the resolve divides by the ACTUAL executed budget,
    never the requested target (design D4)."""
    batch = _mlt_chain_batch(num_chains, chain_batch)
    for _ in range(max(1, int(iterations))):
        base = 0
        while base < num_chains:
            n = min(batch, num_chains - base)
            rec.push_window(base, n)
            rec.dispatch_count(WF_MLT_MUTATE, n, 64)
            rec.barrier()
            rec.flush()
            base += n
    rec.push_window(0, num_pixels)
    rec.dispatch_count(WF_MLT_RESOLVE, num_pixels, 64)
    rec.barrier()
