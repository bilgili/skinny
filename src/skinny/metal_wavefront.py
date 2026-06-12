"""Metal wavefront execution backend — the staged path tracer on slang-rhi.

Metal sibling of :mod:`skinny.vk_wavefront`'s ``WavefrontPathPass`` (change
metal-wavefront-parity, design D1/D3). The bounce-loop *stage order* lives in
:func:`skinny.wavefront_driver.record_path_loop` — shared with Vulkan — so this
module supplies only the Metal primitives:

* per-entry **in-process Slang→Metal** pipelines (one slang session, one module
  load, one linked program + compute pipeline per kernel entry — no ``slangc``,
  no SPIR-V), compiled with ``SKINNY_METAL=1`` like the megakernel;
* the path-state / hit / counting-sort queue buffers, **sized from the
  reflected MSL strides** (design D4 / task 3.3 — Slang pads ``float3`` to 16 B
  on Metal, so the Vulkan scalar strides would undersize them; the reflected
  ``wfState`` stride is asserted equal to the GPU-free
  ``wavefront_layout.path_state_size(msl=True)`` mirror from task 1.5);
* :class:`_MetalPathRecorder`, the :class:`skinny.wavefront_driver.
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
from skinny.wavefront_driver import record_path_loop
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


class _EntryPipeline:
    """One linked wavefront kernel entry: program + compute pipeline + the
    reflected global-parameter names (the bind-by-name filter). Duck-types the
    surface :class:`skinny.metal_compute.MetalFrameEncoder` dispatches against
    (``program`` / ``pipeline`` / ``global_names``)."""

    def __init__(self, ctx, session, module, entry: str) -> None:
        self.entry = entry
        self.program = session.link_program(
            [module], [module.entry_point(entry)])
        self.pipeline = ctx.device.create_compute_pipeline(program=self.program)
        self.global_names = {p.name for p in self.program.layout.parameters}


class _MetalPathRecorder:
    """Metal adapter for :func:`skinny.wavefront_driver.record_path_loop`.

    Encodes each primitive into the pass's per-frame
    :class:`~skinny.metal_compute.MetalFrameEncoder`. The Vulkan push-constant
    triple ``{streamBase, shadeSlot, streamSize}`` is carried host-side and
    baked into every stage dispatch as the ``wfTile`` uniform blob (each
    dispatch owns a fresh root shader object, so per-stage values are
    naturally scoped — no GPU push-constant state to restore)."""

    def __init__(self, p: "MetalWavefrontPathPass", enc: MetalFrameEncoder,
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
        raise NotImplementedError(
            "neural directional proposal on Metal lands in phase 6")

    def restir_primary_direct(self) -> None:
        raise NotImplementedError("ReSTIR DI on Metal lands in phase 5")


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
                 graph_fragments=None, neural_config=None) -> None:
        self.ctx = ctx
        self.shader_dir = Path(shader_dir)
        self.stream_size = int(stream_size)
        self.num_pixels = int(num_pixels)
        self.build_catchall = bool(build_catchall)
        self.record_capacity = int(record_capacity)
        self.graph_fragments = list(graph_fragments) if graph_fragments else []
        self._restir = None  # reuse plugin seam (phase 5)
        self._neural = None  # neural pre-pass seam (phase 6)

        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        self._neural_config = neural_config

        spy = ctx._spy
        dev = ctx.device

        # In-process Slang→Metal session — identical compiler surface to the
        # Metal megakernel (`metal_compute.ComputePipeline._build`): MSL layout
        # (no scalar-layout flag), column-major matrices so the Vulkan-packed
        # camera/instance matrices read identically, and the same include +
        # define set as the Vulkan `_compile_full_spv` wavefront kernels.
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        opts = spy.SlangCompilerOptions()
        opts.include_paths = [self.shader_dir, mtlx_genslang]
        defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"}
        defines.update(_defines_dict(neural_config.slang_defines()))
        opts.defines = defines
        opts.matrix_layout = spy.SlangMatrixLayout.column_major
        session = dev.create_slang_session(compiler_options=opts)

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
        rec_stack_elems = max(1, self.record_capacity * self.MAX_BOUNCES)
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
        self.graph_param_layouts: dict = {}
        for gf in self.graph_fragments:
            for prog in ([flat] + ([catch.program] if catch else [])):
                ref = _reflect_element(prog, f"graphParams_{gf.sanitized_name}")
                if ref is not None and ref[0]:
                    self.graph_param_layouts[gf.target_name] = ref
                    break

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
        """Reuse-plugin seam (phase 5). Only identity reuse is supported on
        Metal until the ReSTIR pass set is ported."""
        if restir is not None:
            raise NotImplementedError("ReSTIR DI on Metal lands in phase 5")
        self._restir = None

    def set_neural(self, neural) -> None:
        """Neural-proposal seam (phase 6)."""
        if neural is not None:
            raise NotImplementedError(
                "neural directional proposal on Metal lands in phase 6")
        self._neural = None

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
        bindless = None
        if bindless_textures is not None:
            bindless = ("flatMaterialTextures", bindless_textures, self.default_tex)
        enc = MetalFrameEncoder(self.ctx)
        rec = _MetalPathRecorder(self, enc, all_binds, uniform_blob, bindless)
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
