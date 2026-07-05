"""Metal wavefront execution backend — the staged path + bdpt tracers on slang-rhi.

Metal sibling of :mod:`skinny.vk_wavefront`'s ``WavefrontPathPass`` /
``WavefrontBdptPass`` (change metal-wavefront-parity, design D1/D3). The loop
*stage orders* live in :func:`skinny.wavefront_driver.record_path_loop` /
:func:`~skinny.wavefront_driver.record_bdpt_loop` — shared with Vulkan — so
this module supplies only the Metal primitives:

* per-entry **in-process Slang→Metal** pipelines (one slang session, one module
  load, one linked program + compute pipeline per kernel entry — no ``slangc``,
  no SPIR-V), compiled with ``SKINNY_METAL=1`` like the megakernel;
* the path-state / hit / counting-sort queue buffers, **sized from the
  reflected MSL strides** (design D4 / task 3.3 — Slang pads ``float3`` to 16 B
  on Metal, so the Vulkan scalar strides would undersize them; the reflected
  ``wfState`` stride is asserted equal to the GPU-free
  ``wavefront_layout.path_state_size(msl=True)`` mirror from task 1.5);
* :class:`_MetalWavefrontRecorder`, the :class:`skinny.wavefront_driver.
  WavefrontRecorder` adapter that encodes every stage into **one**
  :class:`skinny.metal_compute.MetalFrameEncoder` per frame with a global
  compute barrier between stages (design D3 — no per-stage ``wait_for_idle``).

Per-material shade dispatches use native indirect dispatch when the device
supports it. On slang-rhi 0.42's Metal backend — which silently no-ops indirect
dispatch, so the `metal_context` probe resolves **False** — the recorder
flushes the frame encoder, reads the GPU-written ``(x, y, z)`` group triple
back to the host, and issues an equal-count direct dispatch (design D2's
CPU-readback fallback, task 1.3). Uniform and per-tile constants bind via
``set_data`` byte blobs only (design D4 fence-hang discipline); resources bind
by name on each stage's root object, filtered to the entry's reflected globals
(dead-stripped names are skipped).
"""

from __future__ import annotations

import struct
from pathlib import Path

from skinny.metal_compute import MetalFrameEncoder, StorageBuffer
from skinny.wavefront_driver import record_bdpt_loop, record_path_loop
from skinny.wavefront_layout import path_state_size, rec_vertex_size


def _defines_dict(tokens: tuple[str, ...]) -> dict[str, str]:
    """Convert the slangc-style ``("-D", "K=V", …)`` neural-config tokens into
    the ``{name: value}`` dict a SlangPy session takes. Default config → empty
    tuple → empty dict → byte-identical compile options."""
    out: dict[str, str] = {}
    it = iter(tokens)
    for tok in it:
        if tok != "-D":
            continue
        kv = next(it, None)
        if kv is None:
            break
        name, _, value = kv.partition("=")
        out[name] = value if value else "1"
    return out


def _reflect_uniform_layout(program) -> tuple[dict[str, tuple[int, int]], int]:
    """Reflect the MSL field offsets/size of the ``fc`` uniform block (the same
    walk as ``metal_compute.ComputePipeline._reflect_uniform_layout``), so the
    renderer's ``_pack_uniforms_msl`` relocator works against a wavefront
    program when no megakernel pipeline is compiled (wavefront mode builds the
    scene bindings without ``main_pass``)."""
    fc = next((p for p in program.layout.parameters if p.name == "fc"), None)
    if fc is None:
        return {}, 0
    layout: dict[str, tuple[int, int]] = {}

    def walk(type_layout, base: int, prefix: str) -> None:
        for f in getattr(type_layout, "fields", None) or []:
            off = base + int(f.offset)
            ftl = f.type_layout
            name = f"{prefix}{f.name}"
            layout[name] = (off, int(getattr(ftl, "size", 0)))
            if getattr(ftl, "fields", None):
                walk(ftl, off, f"{name}.")

    walk(fc.type_layout, 0, "")
    return layout, int(fc.type_layout.size)


def _reflect_element(program, name: str):
    """Reflect a ``StructuredBuffer<T>`` global's element layout: returns
    ``({field: (offset, size)}, stride)`` or ``None`` when the global is absent
    (dead-stripped) from this program."""
    p = next((q for q in program.layout.parameters if q.name == name), None)
    if p is None:
        return None
    etl = getattr(p.type_layout, "element_type_layout", None)
    if etl is None:
        return None
    fields = {
        f.name: (int(f.offset), int(getattr(f.type_layout, "size", 0)))
        for f in (getattr(etl, "fields", None) or [])
    }
    stride = int(getattr(etl, "stride", 0) or getattr(etl, "size", 0))
    return fields, stride


def _metal_slang_session(ctx, shader_dir: Path, extra_defines: dict | None = None):
    """In-process Slang→Metal session — identical compiler surface to the
    Metal megakernel (`metal_compute.ComputePipeline._build`): MSL layout
    (no scalar-layout flag), column-major matrices so the Vulkan-packed
    camera/instance matrices read identically, and the same include + define
    set as the Vulkan ``_compile_full_spv`` wavefront kernels."""
    spy = ctx._spy
    mtlx_genslang = shader_dir.parent / "mtlx" / "genslang"
    opts = spy.SlangCompilerOptions()
    opts.include_paths = [shader_dir, mtlx_genslang]
    # SKINNY_WAVEFRONT mirrors the Vulkan wavefront compile (vk_wavefront.py): it
    # selects the wavefront-only 3D interior subsurface walk in shared shader code
    # (path.slang). The Metal megakernel (metal_compute.py) omits it → 1D slab.
    defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1",
               "SKINNY_WAVEFRONT": "1"}
    defines.update(extra_defines or {})
    opts.defines = defines
    opts.matrix_layout = spy.SlangMatrixLayout.column_major
    return ctx.device.create_slang_session(compiler_options=opts)


class _EntryPipeline:
    """One linked wavefront kernel entry: program + compute pipeline + the
    reflected global-parameter names (the bind-by-name filter). Duck-types the
    surface :class:`skinny.metal_compute.MetalFrameEncoder` dispatches against
    (``program`` / ``pipeline`` / ``global_names``)."""

    def __init__(self, ctx, session, module, entry: str) -> None:
        self.entry = entry
        self.program = session.link_program(
            [module], [module.entry_point(entry)])
        try:
            self.pipeline = ctx.device.create_compute_pipeline(program=self.program)
        except Exception as exc:
            # Metal caps a kernel's buffer argument table at 31 slots; an
            # overflow surfaces here as an opaque SLANG_FAIL from pipeline
            # creation. Name the kernel and the program's global count so the
            # failure is actionable (change metal-record-drain, design D2):
            # the fix is compiling wavefront-dead globals out of this build
            # flavor (see bindings.slang / path_record_common.slang gates).
            n_globals = len(list(self.program.layout.parameters))
            raise RuntimeError(
                f"Metal compute pipeline for wavefront kernel {entry!r} failed "
                f"to build ({exc}). The program declares {n_globals} globals — "
                f"if this flavor exceeds Metal's 31-buffer-slot argument table, "
                f"compile out wavefront-dead buffer globals for it "
                f"(see the SKINNY_METAL_RECORDS gates, change metal-record-drain)."
            ) from exc
        self.global_names = {p.name for p in self.program.layout.parameters}


class _MetalWavefrontRecorder:
    """Metal adapter for the :class:`skinny.wavefront_driver.WavefrontRecorder`
    protocol — drives both :func:`~skinny.wavefront_driver.record_path_loop`
    (over a :class:`MetalWavefrontPathPass`) and
    :func:`~skinny.wavefront_driver.record_bdpt_loop` (over a
    :class:`MetalWavefrontBdptPass`); the two passes expose the same
    ``stream_size`` / ``_GROUP`` / ``_entries`` / ``buffers`` surface.

    Encodes each primitive into the pass's per-frame
    :class:`~skinny.metal_compute.MetalFrameEncoder`. The Vulkan push-constant
    triple ``{streamBase, shadeSlot, streamSize}`` is carried host-side and
    baked into every stage dispatch as the ``wfTile`` uniform blob (each
    dispatch owns a fresh root shader object, so per-stage values are
    naturally scoped — no GPU push-constant state to restore)."""

    def __init__(self, p, enc: MetalFrameEncoder,
                 binds: dict, fc_blob: bytes, bindless) -> None:
        self._p = p
        self._enc = enc
        self._binds = binds
        self._fc = fc_blob
        self._bindless = bindless
        self._tile = (0, 0, p.stream_size)

    @property
    def stream_size(self) -> int:
        return self._p.stream_size

    @property
    def has_neural(self) -> bool:
        return self._p._neural is not None

    @property
    def has_restir(self) -> bool:
        return self._p._restir is not None

    def _tile_blob(self) -> bytes:
        return struct.pack("3I", *(int(v) for v in self._tile))

    def _groups_full(self) -> int:
        return (self._p.stream_size + self._p._GROUP - 1) // self._p._GROUP

    def _dispatch(self, entry: str, groups) -> None:
        self._enc.dispatch(
            self._p._entries[entry], groups,
            bindings=self._binds, uniform_blob=self._fc,
            uniforms={"wfTile": self._tile_blob()}, bindless=self._bindless)

    def barrier(self) -> None:
        self._enc.barrier()

    def flush_heavy_eye(self) -> None:
        # Bound the heavy per-tile eye submit (change
        # wavefront-nonflat-tiled-fallback): when the scene has a non-terminal
        # non-flat material (VOLUME / PYTHON), the non-flat first-hit path
        # fallback runs a full multi-bounce path in the eye kernel, so commit +
        # drain this tile's command buffer before the next to stay within the
        # macOS GPU watchdog. No-op otherwise (single submit, byte-identical).
        if getattr(self._p, "bound_heavy_eye", False):
            self._enc.flush()

    def clear_counts(self) -> None:
        # WAR guard: the previous bounce's shade reads slot_count/queue; order
        # those reads before the clears, then make the zeros visible to the
        # next compute stage (the Vulkan path uses TRANSFER barriers around
        # vkCmdFillBuffer; global_barrier covers both directions here).
        self._enc.barrier()
        self._enc.clear(self._p.buffers["slot_count"])
        self._enc.clear(self._p.buffers["slot_cursor"])
        self._enc.barrier()

    def push_tile(self, stream_base: int) -> None:
        self._tile = (int(stream_base), 0, self._p.stream_size)

    def dispatch_full(self, entry: str) -> None:
        self._dispatch(entry, (self._groups_full(), 1, 1))

    def dispatch_one(self, entry: str) -> None:
        self._dispatch(entry, (1, 1, 1))

    def shade(self, slot: int, entry: str) -> None:
        self._tile = (self._tile[0], int(slot), self._tile[2])
        if self._p.ctx.supports_indirect_dispatch:
            self._enc.dispatch_indirect(
                self._p._entries[entry], self._p.buffers["indirect"],
                int(slot) * 12, bindings=self._binds, uniform_blob=self._fc,
                uniforms={"wfTile": self._tile_blob()}, bindless=self._bindless)
            return
        # CPU-readback fallback (design D2 / task 1.3, the live path on
        # slang-rhi 0.42): drain the encoded work so the GPU-written args are
        # visible, read the (x, y, z) triple, and direct-dispatch that count.
        self._enc.flush()
        raw = self._p.buffers["indirect"].buffer.to_numpy().tobytes()
        gx, gy, gz = struct.unpack_from("<III", raw, int(slot) * 12)
        if gx == 0 or gy == 0 or gz == 0:
            return  # empty queue — nothing to shade this bounce
        self._dispatch(entry, (gx, gy, gz))

    def neural_prepass(self) -> None:
        """Encode the neural-proposal forward pass over the whole stream — the
        Metal analogue of ``vk_wavefront.WavefrontNeuralProposalPass.record``.
        The Vulkan 16 B ``npc`` push constant rides the same ``set_data``
        uniform-blob mechanism as ``wfTile``/``rpc`` (design D4); the
        npState/npHits/npOut binds are already merged into ``self._binds`` by
        ``dispatch_frame``."""
        np_ = self._p._neural
        groups = ((np_.stream_size + np_._GROUP - 1) // np_._GROUP, 1, 1)
        self._enc.dispatch(
            np_._entries["wfNeuralProposal"], groups, bindings=self._binds,
            uniform_blob=self._fc, uniforms={"npc": np_.npc_blob()},
            bindless=self._bindless)

    def restir_primary_direct(self) -> None:
        """Encode the ReSTIR DI primary-direct pass set (fill → spatial →
        resolve) at bounce 0 — the Metal analogue of
        ``vk_wavefront.RestirDiPass.record_primary_direct``. The Vulkan
        36 B ``rpc`` push constant rides the same ``set_data`` uniform-blob
        mechanism as ``wfTile`` (design D4); the reservoir/G-buffer binds are
        already merged into ``self._binds`` by ``dispatch_frame``."""
        rp = self._p._restir
        groups = ((rp.stream_size + rp._GROUP - 1) // rp._GROUP, 1, 1)
        blob = rp.rpc_blob()
        for k, entry in enumerate(("restirFill", "restirSpatial", "restirResolve")):
            if k > 0:
                self._enc.barrier()
            self._enc.dispatch(
                rp._entries[entry], groups, bindings=self._binds,
                uniform_blob=self._fc, uniforms={"rpc": blob},
                bindless=self._bindless)


class MetalRestirDiPass:
    """ReSTIR DI wavefront pass set on a :class:`~skinny.metal_context.
    MetalContext` — the Metal sibling of ``vk_wavefront.RestirDiPass`` (change
    metal-wavefront-parity, phase 5; the host-side selector is
    ``sampling.reuse.RestirDiReuse``).

    Compiles the three ``restir/restir_primary.slang`` entries (fill →
    spatial → resolve) through the same in-process Slang→Metal session as the
    wavefront kernels, and owns the per-pixel ping-pong reservoir + G-buffer
    ``StorageBuffer``s, sized from the **reflected MSL** element strides
    (Slang pads ``float3`` to 16 B on Metal, so the Vulkan scalar-headroom
    strides could undersize them). The buffers persist across frames — the
    temporal pass reads last frame's ``wfReservoirB`` — and reset only when
    the renderer rebuilds the pass on a reuse-config-hash change (the
    pass-structural contract of the scene-sampling reuse hook).

    The shared ``wfState``/``wfHits`` buffers and the scene globals bind by
    name at dispatch (``dispatch_frame`` merges :attr:`bind_map` over the
    scene binds); the ``rpc`` config block is a ``set_data`` byte blob per
    dispatch (design D4), packed scalar — uints/floats only, so the MSL
    constant-buffer layout matches the Vulkan 36 B push constant.
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] in restir_primary.slang
    RESERVOIR_STRIDE = 32  # reflection fallback only — MSL stride is authoritative
    GBUF_STRIDE = 32       # reflection fallback only
    # Default ReSTIR config (mirrors restir_primary.slang RestirPC and the
    # Vulkan pass's DEFAULT_CONFIG — lockstep). flags bit0 spatial, bit1
    # temporal. Renderer overrides via RestirDiReuse.
    DEFAULT_CONFIG = dict(flags=0x3, mLight=8, spatialK=5, spatialRadius=16.0,
                          normalThresh=0.9, depthThresh=0.1, mCap=20, mBsdf=1)

    def __init__(self, ctx, shader_dir: Path, stream_size: int,
                 config: dict | None = None) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # No neural defines: parity with the Vulkan `_compile_full_spv`
        # restir kernels, which compile with the plain define set.
        session = _metal_slang_session(ctx, Path(shader_dir))
        src_path = Path(shader_dir) / "restir" / "restir_primary.slang"
        module = session.load_module_from_source(
            "restir_primary", src_path.read_text(encoding="utf-8"), str(src_path))
        self._entries = {e: _EntryPipeline(ctx, session, module, e)
                         for e in ("restirFill", "restirSpatial", "restirResolve")}

        fill = self._entries["restirFill"].program
        self.reservoir_stride = ((_reflect_element(fill, "wfReservoirA")
                                  or (None, 0))[1] or self.RESERVOIR_STRIDE)
        self.gbuf_stride = ((_reflect_element(fill, "wfGBuffer")
                             or (None, 0))[1] or self.GBUF_STRIDE)

        # Ping-pong reservoirs (A/B) + G-buffer (pos+normal) for the spatial
        # domain check. Zero-filled once: device-local memory is otherwise
        # uninitialised, and spatial reads wfReservoirB (the temporal history)
        # before the first resolve writes it.
        n = self.stream_size
        self._resA = StorageBuffer(ctx, n * self.reservoir_stride)
        self._resB = StorageBuffer(ctx, n * self.reservoir_stride)
        self._gbuffer = StorageBuffer(ctx, n * self.gbuf_stride)
        for buf in (self._resA, self._resB, self._gbuffer):
            buf.fill_zero_sync()
        # Slang global name → buffer, merged into the scene binds per dispatch.
        self.bind_map = {
            "wfReservoirA": self._resA,
            "wfReservoirB": self._resB,
            "wfGBuffer": self._gbuffer,
        }

    def rpc_blob(self) -> bytes:
        """The RestirPC config block: streamSize, flags, mLight, spatialK,
        spatialRadius, normalThresh, depthThresh, mCap, mBsdf (scalar, 36 B —
        identical to the Vulkan push-constant pack)."""
        c = self.config
        return struct.pack(
            "IIIIfffII", self.stream_size, int(c["flags"]), int(c["mLight"]),
            int(c["spatialK"]), float(c["spatialRadius"]),
            float(c["normalThresh"]), float(c["depthThresh"]), int(c["mCap"]),
            int(c.get("mBsdf", 1)))

    def destroy(self) -> None:  # SlangPy owns lifetimes via refcount
        for buf in (self._resA, self._resB, self._gbuffer):
            buf.destroy()
        self.bind_map = {}
        self._entries = {}


class MetalNeuralProposalPass:
    """Neural directional-proposal pre-pass on a :class:`~skinny.metal_context.
    MetalContext` — the Metal sibling of ``vk_wavefront.
    WavefrontNeuralProposalPass`` (change metal-wavefront-parity, phase 6).

    Compiles the single ``wavefront/neural_proposal_pass.slang::
    wfNeuralProposal`` entry through the same in-process Slang→Metal session
    as the wavefront kernels, with ``SKINNY_METAL_NEURAL=1`` so the frozen
    weight buffers (33/34/35) are real declarations rather than the slot-cap
    stubs, plus the size/precision defines of the active ``NeuralBuildConfig``
    (the host must match — ``renderer._effective_neural_config()`` degrades
    fp16 to fp32 on devices whose probe lacks it, design D6).

    Owns no buffers: ``npState``/``npHits``/``npOut`` alias the path pass's
    state/hit/neural ``StorageBuffer``s via :attr:`bind_map` (merged into the
    scene binds per dispatch), and the weight buffers are the renderer's
    backend-neutral 33/34/35 allocations, bound by name from
    ``_build_metal_binds``. The Vulkan 16 B ``npc`` push constant
    ``{streamSize, networkVersion, pad, pad}`` is a ``set_data`` byte blob
    (design D4).
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] in neural_proposal_pass.slang

    def __init__(self, ctx, shader_dir: Path, path_pass: "MetalWavefrontPathPass",
                 stream_size: int, network_version: int = 0,
                 neural_config=None) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.network_version = int(network_version)

        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        defines = _defines_dict(neural_config.slang_defines())
        defines["SKINNY_METAL_NEURAL"] = "1"
        session = _metal_slang_session(ctx, Path(shader_dir), defines)
        src_path = Path(shader_dir) / "wavefront" / "neural_proposal_pass.slang"
        module = session.load_module_from_source(
            "neural_proposal_pass", src_path.read_text(encoding="utf-8"),
            str(src_path))
        self._entries = {"wfNeuralProposal": _EntryPipeline(
            ctx, session, module, "wfNeuralProposal")}

        # Slang global name → buffer, merged into the scene binds per dispatch
        # (the Vulkan pass's set-1 contents: state 0, hit 1, neural-out 2).
        self.bind_map = {
            "npState": path_pass.buffers["state"],
            "npHits": path_pass.buffers["hit"],
            "npOut": path_pass.buffers["neural"],
        }

    def npc_blob(self) -> bytes:
        """The NeuralPC block: {streamSize, networkVersion, _pad0, _pad1}
        (16 B — identical to the Vulkan push-constant pack)."""
        return struct.pack("4I", self.stream_size, self.network_version, 0, 0)

    def destroy(self) -> None:  # SlangPy owns lifetimes via refcount
        self.bind_map = {}
        self._entries = {}


class MetalWavefrontPathPass:
    """Staged wavefront path tracer on a :class:`~skinny.metal_context.
    MetalContext` — the Metal sibling of ``vk_wavefront.WavefrontPathPass``.

    Owns the per-entry Slang→Metal pipelines plus the path-state / hit /
    counting-sort / neural / record buffers (allocated through the
    backend-neutral ``metal_compute.StorageBuffer`` wrappers at the **reflected
    MSL** strides — task 2.4's allocation migration plus task 3.3's layout
    resolution in one place). ``dispatch_frame`` records the shared
    ``record_path_loop`` into one frame encoder and submits once.

    Exposes the same reflected-layout surface as the Metal megakernel
    ``ComputePipeline`` (``uniform_layout`` / ``graph_param_layouts`` /
    ``std_surface_layout`` / ``mtlx_skin_layout``) so the renderer's MSL
    relocators work in wavefront mode, where no megakernel is compiled.
    """

    _GROUP = 64  # matches [numthreads(64, 1, 1)] in wavefront_path.slang
    MAX_BOUNCES = 6  # lockstep with WF_MAX_BOUNCES in the shader
    STREAM_CAP = 1 << 20  # max lanes per stream — bounds path-state VRAM
    NUM_SLOTS = 2  # lockstep with WF_NUM_SLOTS (0 = flat, 1 = catch-all)

    def __init__(self, ctx, shader_dir: Path, stream_size: int, num_pixels: int,
                 build_catchall: bool = True, record_capacity: int = 0,
                 graph_fragments=None, neural_config=None,
                 neural_active: bool = False, records_active: bool = False) -> None:
        self.ctx = ctx
        self.shader_dir = Path(shader_dir)
        self.stream_size = int(stream_size)
        self.num_pixels = int(num_pixels)
        self.build_catchall = bool(build_catchall)
        self.record_capacity = int(record_capacity)
        self.graph_fragments = list(graph_fragments) if graph_fragments else []
        self._restir = None  # reuse plugin seam (phase 5)
        self._neural = None  # neural pre-pass seam (phase 6)
        self.neural_active = bool(neural_active)
        self.records_active = bool(records_active)

        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        self._neural_config = neural_config

        spy = ctx._spy
        dev = ctx.device

        # SKINNY_METAL_NEURAL un-stubs the frozen weight buffers + the inline
        # inverse pdf in the shade kernels (phase 6). Compiled in only when the
        # neural proposal is selected, so the default build stays under Metal's
        # 31-buffer-slot argument-table cap with the same headroom as phase 3.
        # SKINNY_METAL_RECORDS (change metal-record-drain) un-stubs the
        # wf_records emitters + the binding-36/37 record append, compiled in
        # only while online training is armed — the default render stays
        # byte-identical and keeps its slot headroom.
        defines = _defines_dict(neural_config.slang_defines())
        if self.neural_active:
            defines["SKINNY_METAL_NEURAL"] = "1"
        if self.records_active:
            defines["SKINNY_METAL_RECORDS"] = "1"
        session = _metal_slang_session(ctx, self.shader_dir, defines)

        src_path = self.shader_dir / "wavefront" / "wavefront_path.slang"
        module = session.load_module_from_source(
            "wavefront_path", src_path.read_text(encoding="utf-8"), str(src_path))

        entries = ["wfPathGenerate", "wfPathIntersect", "wfBuildArgs",
                   "wfScatter", "wfPathShadeFlat", "wfPathResolve"]
        if self.build_catchall:
            entries.append("wfPathShade")
        self._entries = {e: _EntryPipeline(ctx, session, module, e)
                         for e in entries}

        # ── Reflected MSL strides (task 3.3) ─────────────────────────
        # The record structs round-trip through VRAM GPU-side only, so the host
        # needs just the stride to size the buffers (design B, task 1.5). The
        # wfState stride is locked to the GPU-free mirror; the others come
        # straight from reflection (HitInfo has no GPU-free mirror).
        gen = self._entries["wfPathGenerate"].program
        isect = self._entries["wfPathIntersect"].program
        flat = self._entries["wfPathShadeFlat"].program
        state_stride = (_reflect_element(gen, "wfState") or (None, 0))[1]
        expected = path_state_size(msl=True)
        if state_stride and state_stride != expected:
            raise RuntimeError(
                f"reflected Metal WavefrontPathState stride {state_stride}B != "
                f"wavefront_layout.path_state_size(msl=True) {expected}B — "
                f"update the GPU-free mirror (task 1.5)")
        self.state_stride = state_stride or expected
        self.hit_stride = (_reflect_element(isect, "wfHits") or (None, 0))[1] or 128
        self.neural_stride = (_reflect_element(flat, "wfNeural") or (None, 0))[1] or 48
        rec_ref = (_reflect_element(gen, "wfRecStack")
                   or _reflect_element(flat, "wfRecStack"))
        self.rec_vertex_stride = (rec_ref or (None, 0))[1] or rec_vertex_size(msl=True)

        # ── Buffers (task 2.4: backend-neutral wrappers, MSL sizing) ─
        # Mirrors the Vulkan pass's set-1 contents; bound by the Slang global
        # names instead of binding numbers. Zero-filled once: device-local
        # memory is otherwise uninitialised and resolve/flat-shade read
        # wfState/wfNeural before the first full write cycle.
        n = self.stream_size
        # Records build: element 0 of each lane's region is the count header
        # (wf_records.slang WF_REC_LANE = REC_MAX_BOUNCES + 1 — the per-lane
        # count buffer is folded into the stack to save an argument slot).
        rec_lane = self.MAX_BOUNCES + (1 if self.records_active else 0)
        rec_stack_elems = max(1, self.record_capacity * rec_lane)
        rec_count_elems = max(1, self.record_capacity)
        self.buffers: dict[str, StorageBuffer] = {
            "state": StorageBuffer(ctx, n * self.state_stride),
            "hit": StorageBuffer(ctx, n * self.hit_stride),
            "lane_slot": StorageBuffer(ctx, n * 4),
            "slot_count": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "slot_offset": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "slot_queue": StorageBuffer(ctx, n * 4),
            "slot_cursor": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "indirect": StorageBuffer(ctx, self.NUM_SLOTS * 12, indirect=True),
            "neural": StorageBuffer(ctx, n * self.neural_stride),
            "rec_stack": StorageBuffer(ctx, rec_stack_elems * self.rec_vertex_stride),
            "rec_count": StorageBuffer(ctx, rec_count_elems * 4),
        }
        for buf in self.buffers.values():
            buf.fill_zero_sync()
        self.neural_buf = self.buffers["neural"]
        # Slang global name → buffer, merged into the scene binds per dispatch.
        self._bind_map = {
            "wfState": self.buffers["state"],
            "wfHits": self.buffers["hit"],
            "wfLaneSlot": self.buffers["lane_slot"],
            "wfSlotCount": self.buffers["slot_count"],
            "wfSlotOffset": self.buffers["slot_offset"],
            "wfSlotQueue": self.buffers["slot_queue"],
            "wfSlotCursor": self.buffers["slot_cursor"],
            "wfIndirectArgs": self.buffers["indirect"],
            "wfNeural": self.buffers["neural"],
            "wfRecStack": self.buffers["rec_stack"],
            "wfRecCount": self.buffers["rec_count"],
        }

        # ── Reflected layouts for the renderer's MSL relocators ──────
        # (duck-types the megakernel ComputePipeline's reflection surface; in
        # wavefront mode no megakernel is compiled, so this pass is the
        # `_msl_layout_source`.) fc comes from the generate program; the
        # material-param layouts from the shade program(s) that reference them.
        self.uniform_layout, self.uniform_size = _reflect_uniform_layout(gen)
        catch = self._entries.get("wfPathShade")
        self.mtlx_skin_layout: dict = {}
        self.mtlx_skin_stride = 0
        if catch is not None:
            ref = _reflect_element(catch.program, "mtlxSkin")
            if ref is not None:
                self.mtlx_skin_layout, self.mtlx_skin_stride = ref
        self.std_surface_layout: dict = {}
        self.std_surface_stride = 0
        ref = _reflect_element(flat, "stdSurfaceParams")
        if ref is not None:
            self.std_surface_layout, self.std_surface_stride = ref
        # Empty since change combine-graph-param-buffers: graph params are read
        # from the combined `ByteAddressBuffer graphParamsCombined` via `Load<T>`
        # (scalar layout, identical Metal/SPIR-V), so there is no per-graph
        # StructuredBuffer element layout to reflect.
        self.graph_param_layouts: dict = {}

        # Default 1×1 texture for unfilled bindless slots (Metal binds every
        # slot; mirrors the megakernel pipeline's `_default_tex`).
        self.default_tex = dev.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local,
            label="skinny.wf_bindless_default",
        )

    def set_restir(self, restir) -> None:
        """Hook the ReSTIR DI reuse plugin (a :class:`MetalRestirDiPass`, or
        ``None`` for identity reuse). The recorder schedules its fill →
        spatial → resolve set at bounce 0 through the shared
        ``record_path_loop`` reuse hook."""
        self._restir = restir

    def set_neural(self, neural) -> None:
        """Hook the neural-proposal pre-pass (a :class:`MetalNeuralProposalPass`,
        or ``None``). The recorder schedules its forward dispatch every bounce
        between scatter and shade through the shared ``record_path_loop`` hook.
        Requires the pass to have been built with ``neural_active=True`` — the
        shade kernels' inline inverse pdf is the slot-cap stub otherwise."""
        if neural is not None and not self.neural_active:
            raise RuntimeError(
                "MetalWavefrontPathPass was compiled without "
                "SKINNY_METAL_NEURAL (neural_active=False); rebuild the pass "
                "with neural_active=True before attaching a neural pre-pass")
        self._neural = neural

    def dispatch_frame(self, *, binds: dict, uniform_blob: bytes,
                       bindless_textures=None) -> None:
        """Record + submit one wavefront frame: the shared
        ``record_path_loop`` stage order over one ``MetalFrameEncoder``.

        ``binds`` is the renderer's scene global → native resource map (the
        megakernel ``_build_metal_binds`` dict); the pass's own queue buffers
        are merged on top. ``uniform_blob`` is the MSL-packed ``fc`` block;
        ``bindless_textures`` the flat-material texture-pool slot list."""
        all_binds = dict(binds)
        all_binds.update(self._bind_map)
        if self._restir is not None:
            all_binds.update(self._restir.bind_map)
        if self._neural is not None:
            all_binds.update(self._neural.bind_map)
        bindless = None
        if bindless_textures is not None:
            bindless = ("flatMaterialTextures", bindless_textures, self.default_tex)
        enc = MetalFrameEncoder(self.ctx)
        rec = _MetalWavefrontRecorder(self, enc, all_binds, uniform_blob, bindless)
        record_path_loop(
            rec,
            num_pixels=self.num_pixels,
            stream_size=self.stream_size,
            max_bounces=self.MAX_BOUNCES,
            build_catchall=self.build_catchall,
        )
        enc.submit()

    def destroy(self) -> None:  # SlangPy owns lifetimes via refcount
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
        self._bind_map = {}
        self._entries = {}
        self.default_tex = None


class _MetalSppmRecorder:
    """Metal adapter for :func:`skinny.wavefront_driver.record_sppm_loop`.

    Encodes the SPPM stage primitives into one :class:`MetalFrameEncoder`. SPPM
    has no indirect dispatch (all counts are host-known), so every stage is a
    plain ``enc.dispatch``; the grid / accumulator / visible-point clears are
    whole-buffer ``enc.clear`` (the grid sub-ranges not touched by a clear are
    fully overwritten by the scan / scatter stages, so a whole-buffer clear is
    equivalent + simpler than the Vulkan sub-range fills)."""

    def __init__(self, p, enc: MetalFrameEncoder, binds: dict, fc_blob: bytes, bindless) -> None:
        self._p = p
        self._enc = enc
        self._binds = binds
        self._fc = fc_blob
        self._bindless = bindless
        self._tile = (0, 0, p.stream_size)

    @property
    def stream_size(self) -> int:
        return self._p.stream_size

    def _tile_blob(self) -> bytes:
        return struct.pack("3I", *(int(v) for v in self._tile))

    def _dispatch(self, entry: str, groups) -> None:
        self._enc.dispatch(
            self._p._entries[entry], groups,
            bindings=self._binds, uniform_blob=self._fc,
            uniforms={"sppmTile": self._tile_blob()}, bindless=self._bindless)

    def barrier(self) -> None:
        self._enc.barrier()

    def push_tile(self, stream_base: int) -> None:
        self._tile = (int(stream_base), 0, self._p.stream_size)

    def dispatch_full(self, entry: str) -> None:
        groups = (self._p.stream_size + self._p._GROUP - 1) // self._p._GROUP
        self._dispatch(entry, (max(groups, 1), 1, 1))

    def dispatch_one(self, entry: str) -> None:
        self._dispatch(entry, (1, 1, 1))

    def dispatch_count(self, entry: str, count: int, group_size: int) -> None:
        groups = (int(count) + group_size - 1) // group_size
        self._dispatch(entry, (max(groups, 1), 1, 1))

    def flush_heavy_eye(self) -> None:
        # Bound the heavy per-tile SPPM eye submit (change
        # wavefront-nonflat-tiled-fallback): wfSppmEye's non-flat first-hit path
        # fallback runs a full multi-bounce path for VOLUME / PYTHON. Phase 1 is
        # all dispatch_full (no implicit shade-flush), so commit + drain each eye
        # tile to stay within the macOS GPU watchdog. No-op otherwise.
        if getattr(self._p, "bound_heavy_eye", False):
            self._enc.flush()

    def clear_visible_points(self) -> None:
        self._enc.barrier()
        self._enc.clear(self._p.buffers["visible_points"])
        self._enc.barrier()

    def clear_grid(self) -> None:
        self._enc.barrier()
        self._enc.clear(self._p.buffers["grid"])
        self._enc.barrier()

    def clear_accum(self) -> None:
        self._enc.barrier()
        self._enc.clear(self._p.buffers["accum"])
        self._enc.barrier()


class MetalWavefrontSppmPass:
    """Native-Metal staged SPPM pass (change photon-mapping-sppm) — the Metal
    sibling of :class:`skinny.vk_wavefront.WavefrontSppmPass`. Compiles the eight
    ``integrators/wavefront_sppm.slang`` entries via SlangPy and owns four
    backend-neutral ``StorageBuffer``s bound by Slang global name. SPPM never
    compiles the neural weights, so no ``SKINNY_METAL_SPPM`` gate is needed (the
    kernels sit well under Metal's 31-buffer-slot cap)."""

    _GROUP = 64
    STREAM_CAP = 1 << 20

    _ENTRIES = ["wfSppmEye", "wfSppmGridCount", "wfSppmGridScanBlock",
                "wfSppmGridScanBlockSums", "wfSppmGridScanAdd", "wfSppmGridScatter",
                "wfSppmPhotonTrace", "wfSppmUpdate"]

    def __init__(self, ctx, shader_dir: Path, stream_size: int, num_pixels: int,
                 graph_fragments=None, neural_config=None) -> None:
        from skinny.wavefront_layout import (
            SPPM_ACCUM_STRIDE,
            VISIBLE_POINT_STRIDE_MSL,
            sppm_grid_buffer_sizes,
            sppm_grid_cell_count,
        )
        self.ctx = ctx
        self.shader_dir = Path(shader_dir)
        self.num_pixels = int(num_pixels)
        self.stream_size = int(min(stream_size, self.STREAM_CAP))
        self.num_cells = sppm_grid_cell_count(self.num_pixels)
        self.graph_fragments = list(graph_fragments) if graph_fragments else []

        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        defines = _defines_dict(neural_config.slang_defines())
        session = _metal_slang_session(ctx, self.shader_dir, defines)
        src_path = self.shader_dir / "integrators" / "wavefront_sppm.slang"
        module = session.load_module_from_source(
            "wavefront_sppm", src_path.read_text(encoding="utf-8"), str(src_path))
        self._entries = {e: _EntryPipeline(ctx, session, module, e) for e in self._ENTRIES}

        # Reflected VisiblePoint stride must match the host MSL mirror.
        eye = self._entries["wfSppmEye"].program
        vp_stride = (_reflect_element(eye, "sppmVisiblePoints") or (None, 0))[1]
        if vp_stride and vp_stride != VISIBLE_POINT_STRIDE_MSL:
            raise RuntimeError(
                f"reflected Metal VisiblePoint stride {vp_stride}B != "
                f"wavefront_layout.VISIBLE_POINT_STRIDE_MSL {VISIBLE_POINT_STRIDE_MSL}B")
        self.vp_stride = vp_stride or VISIBLE_POINT_STRIDE_MSL

        # MSL reflection surface for the renderer's relocators (the pass is the
        # `_msl_layout_source` in wavefront mode — no megakernel is compiled). fc
        # + the std_surface material params come from the eye program (it pulls
        # the full flat-material tree). SPPM has no skin/catch-all kernel and
        # reads graph params from the combined ByteAddressBuffer, so those layouts
        # are empty.
        self.uniform_layout, self.uniform_size = _reflect_uniform_layout(eye)
        self.mtlx_skin_layout: dict = {}
        self.mtlx_skin_stride = 0
        self.std_surface_layout: dict = {}
        self.std_surface_stride = 0
        _ss = _reflect_element(eye, "stdSurfaceParams")
        if _ss is not None:
            self.std_surface_layout, self.std_surface_stride = _ss
        self.graph_param_layouts: dict = {}

        grid_sizes = sppm_grid_buffer_sizes(self.num_pixels)
        self.buffers: dict[str, StorageBuffer] = {
            "visible_points": StorageBuffer(ctx, self.num_pixels * self.vp_stride),
            "accum": StorageBuffer(ctx, self.num_pixels * SPPM_ACCUM_STRIDE),
            "grid": StorageBuffer(ctx, grid_sizes["grid_combined"]),
            "scan": StorageBuffer(ctx, grid_sizes["scan_scratch"]),
        }
        for buf in self.buffers.values():
            buf.fill_zero_sync()
        self._bind_map = {
            "sppmVisiblePoints": self.buffers["visible_points"],
            "sppmAccum": self.buffers["accum"],
            "sppmGrid": self.buffers["grid"],
            "sppmScanScratch": self.buffers["scan"],
        }

        spy = ctx._spy
        self.default_tex = ctx.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local, label="skinny.sppm_bindless_default")

    def dispatch_frame(self, *, binds: dict, uniform_blob: bytes,
                       bindless_textures=None, photons: int, first_frame: bool) -> None:
        """Record + submit one SPPM pass over a fresh ``MetalFrameEncoder``."""
        from skinny.wavefront_driver import record_sppm_loop
        all_binds = dict(binds)
        all_binds.update(self._bind_map)
        bindless = None
        if bindless_textures is not None:
            bindless = ("flatMaterialTextures", bindless_textures, self.default_tex)
        enc = MetalFrameEncoder(self.ctx)
        rec = _MetalSppmRecorder(self, enc, all_binds, uniform_blob, bindless)
        record_sppm_loop(
            rec, num_pixels=self.num_pixels, stream_size=self.stream_size,
            num_cells=self.num_cells, photons=int(photons), first_frame=bool(first_frame))
        enc.submit()

    def destroy(self) -> None:
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
        self._bind_map = {}
        self._entries = {}
        self.default_tex = None


class MetalWavefrontBdptPass:
    """Staged wavefront bidirectional path tracer on a :class:`~skinny.
    metal_context.MetalContext` — the Metal sibling of
    ``vk_wavefront.WavefrontBdptPass`` (change metal-wavefront-parity,
    phase 4).

    The loop *stage order* lives in :func:`skinny.wavefront_driver.
    record_bdpt_loop` (shared with Vulkan); this pass supplies the per-entry
    Slang→Metal pipelines for the active ``walk_mode`` plus the eye/light
    subpath-vertex, aux, and connect counting-sort buffers — all sized from
    the **reflected MSL** element strides (design D4: Slang pads ``float3``
    to 16 B on Metal, so the Vulkan scalar strides would undersize them).
    The strategy-split connect and staged bounce-extend kernels dispatch
    indirectly over their compacted slot queues through the same
    :class:`_MetalWavefrontRecorder` primitives as the path pass (native
    indirect when probed, else the CPU-readback fallback).

    Exposes the same reflected-layout surface as the Metal megakernel /
    wavefront-path pipelines (``uniform_layout`` / ``graph_param_layouts`` /
    ``std_surface_layout`` / ``mtlx_skin_layout``) so the renderer's MSL
    relocators work when this pass is the only compiled Metal program.
    """

    _GROUP = 64           # matches [numthreads(64, 1, 1)] in wavefront_bdpt.slang
    BDPT_MAX_VERTS = 7    # lockstep with bdpt.slang BDPT_MAX_VERTS
    VERTEX_STRIDE = 128   # reflection fallback only — MSL stride is authoritative
    AUX_STRIDE = 128      # reflection fallback only
    NUM_SLOTS = 2         # lockstep with WF_BDPT_NUM_SLOTS in the shader
    SLOT_NEE = 0
    SLOT_FULL = 1
    # Eye-walk extend bounces: gen-eye seeds eye[0..1], the loop extends eye[2..].
    EYE_BOUNCES = BDPT_MAX_VERTS - 2
    # Light-walk extend bounces: gen-light seeds light[0], the loop extends light[1..].
    LIGHT_BOUNCES = BDPT_MAX_VERTS - 1
    # Smaller cap than the path tracer: each lane owns 2×BDPT_MAX_VERTS vertices.
    STREAM_CAP = 1 << 18

    WALK_MODES = ("fused", "eye", "eye_light")

    def __init__(self, ctx, shader_dir: Path, stream_size: int, num_pixels: int,
                 walk_mode: str = "fused", graph_fragments=None) -> None:
        self.ctx = ctx
        self.shader_dir = Path(shader_dir)
        self.stream_size = int(stream_size)
        self.num_pixels = int(num_pixels)
        if walk_mode not in self.WALK_MODES:
            raise ValueError(
                f"unknown bdpt walk_mode {walk_mode!r} (expected {self.WALK_MODES})")
        self.walk_mode = walk_mode
        self.graph_fragments = list(graph_fragments) if graph_fragments else []
        self._restir = None  # recorder protocol stubs — bdpt has no reuse hook
        self._neural = None  # nor a neural pre-pass

        spy = ctx._spy
        dev = ctx.device
        # No neural defines: parity with the Vulkan `_compile_full_spv` bdpt
        # kernels, which compile with the plain define set.
        session = _metal_slang_session(ctx, self.shader_dir)
        src_path = self.shader_dir / "wavefront" / "wavefront_bdpt.slang"
        module = session.load_module_from_source(
            "wavefront_bdpt", src_path.read_text(encoding="utf-8"), str(src_path))

        # Entry set per walk mode — mirrors the Vulkan pass: the connect
        # counting sort + split connect + resolve are shared by all modes; only
        # the subpath-build kernels differ. Only the active mode's kernels are
        # compiled (no wasted Metal pipeline builds).
        shared = ["wfBdptClassify", "wfBdptBuildArgs", "wfBdptScatter",
                  "wfBdptConnectNee", "wfBdptConnectFull", "wfBdptResolve"]
        staged_eye = ["wfBdptGenEye", "wfBdptWalkClassify", "wfBdptBounceEye"]
        if walk_mode == "fused":
            entries = ["wfBdptWalk"] + shared
        elif walk_mode == "eye":
            entries = staged_eye + ["wfBdptLightTail"] + shared
        else:  # eye_light
            entries = staged_eye + ["wfBdptGenLight", "wfBdptBounceLight",
                                    "wfBdptSplat"] + shared
        self._entries = {e: _EntryPipeline(ctx, session, module, e)
                         for e in entries}

        # ── Reflected MSL strides (design D4) ────────────────────────
        # BDPTVertex / WfBdptAux round-trip through VRAM GPU-side only, so the
        # host needs just the stride to size the buffers. Reflect from any
        # program that references the globals (the subpath-build kernels do).
        progs = [ep.program for ep in self._entries.values()]

        def stride_of(name: str, fallback: int) -> int:
            for prog in progs:
                ref = _reflect_element(prog, name)
                if ref is not None and ref[1]:
                    return ref[1]
            return fallback

        self.vertex_stride = stride_of("wfEye", self.VERTEX_STRIDE)
        self.aux_stride = stride_of("wfAux", self.AUX_STRIDE)

        # ── Buffers (backend-neutral wrappers, MSL sizing) ───────────
        # Mirrors the Vulkan pass's set-1 contents (eye/light/aux + the 6
        # counting-sort buffers); bound by Slang global name. Zero-filled once:
        # device-local memory is otherwise uninitialised and connect/resolve
        # read wfAux before the first full write cycle.
        n = self.stream_size
        vert_bytes = n * self.BDPT_MAX_VERTS * self.vertex_stride
        self.buffers: dict[str, StorageBuffer] = {
            "eye": StorageBuffer(ctx, vert_bytes),
            "light": StorageBuffer(ctx, vert_bytes),
            "aux": StorageBuffer(ctx, n * self.aux_stride),
            "lane_slot": StorageBuffer(ctx, n * 4),
            "slot_count": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "slot_offset": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "slot_queue": StorageBuffer(ctx, n * 4),
            "slot_cursor": StorageBuffer(ctx, self.NUM_SLOTS * 4),
            "indirect": StorageBuffer(ctx, self.NUM_SLOTS * 12, indirect=True),
        }
        for buf in self.buffers.values():
            buf.fill_zero_sync()
        # Slang global name → buffer, merged into the scene binds per dispatch.
        self._bind_map = {
            "wfEye": self.buffers["eye"],
            "wfLight": self.buffers["light"],
            "wfAux": self.buffers["aux"],
            "wfLaneSlot": self.buffers["lane_slot"],
            "wfSlotCount": self.buffers["slot_count"],
            "wfSlotOffset": self.buffers["slot_offset"],
            "wfSlotQueue": self.buffers["slot_queue"],
            "wfSlotCursor": self.buffers["slot_cursor"],
            "wfIndirectArgs": self.buffers["indirect"],
        }

        # ── Reflected layouts for the renderer's MSL relocators ──────
        # (duck-types the megakernel/wavefront-path reflection surface — in
        # bdpt wavefront mode this pass is the `_msl_layout_source`.)
        self.uniform_layout: dict = {}
        self.uniform_size = 0
        for prog in progs:
            layout, size = _reflect_uniform_layout(prog)
            if layout:
                self.uniform_layout, self.uniform_size = layout, size
                break
        self.mtlx_skin_layout: dict = {}
        self.mtlx_skin_stride = 0
        self.std_surface_layout: dict = {}
        self.std_surface_stride = 0
        for prog in progs:
            ref = _reflect_element(prog, "stdSurfaceParams")
            if ref is not None and ref[0]:
                self.std_surface_layout, self.std_surface_stride = ref
                break
        # Empty since change combine-graph-param-buffers (see the path-pass
        # ctor): the combined byte buffer is read via `Load<T>` in scalar layout,
        # so there is no per-graph element layout to reflect.
        self.graph_param_layouts: dict = {}

        # Default 1×1 texture for unfilled bindless slots (Metal binds every
        # slot; mirrors the megakernel pipeline's `_default_tex`).
        self.default_tex = dev.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=1, height=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local,
            label="skinny.wf_bdpt_bindless_default",
        )

    def dispatch_frame(self, *, binds: dict, uniform_blob: bytes,
                       bindless_textures=None) -> None:
        """Record + submit one wavefront bdpt frame: the shared
        ``record_bdpt_loop`` stage order over one ``MetalFrameEncoder``.

        Same surface as ``MetalWavefrontPathPass.dispatch_frame``: ``binds``
        is the renderer's scene global → native resource map; the pass's own
        subpath/queue buffers are merged on top."""
        all_binds = dict(binds)
        all_binds.update(self._bind_map)
        bindless = None
        if bindless_textures is not None:
            bindless = ("flatMaterialTextures", bindless_textures, self.default_tex)
        enc = MetalFrameEncoder(self.ctx)
        rec = _MetalWavefrontRecorder(self, enc, all_binds, uniform_blob, bindless)
        record_bdpt_loop(
            rec,
            num_pixels=self.num_pixels,
            stream_size=self.stream_size,
            walk_mode=self.walk_mode,
            eye_bounces=self.EYE_BOUNCES,
            light_bounces=self.LIGHT_BOUNCES,
            slot_nee=self.SLOT_NEE,
            slot_full=self.SLOT_FULL,
        )
        enc.submit()

    def destroy(self) -> None:  # SlangPy owns lifetimes via refcount
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
        self._bind_map = {}
        self._entries = {}
        self.default_tex = None
