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

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from skinny.gpu_backend import RECORDING_CAPABILITIES
from skinny.shader_variants import Family, ShaderVariantKey, Target

__all__ = [
    "BINDLESS_TEXTURE_CAPACITY",
    "Call",
    "RECORDABLE_EXCLUSIONS",
    "RECORDABLE_PASSES",
    "RecordablePass",
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
    "compute_entry_points",
    "shader_dir",
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
    supports_external_memory = False
    supports_external_semaphore = False

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
    def __init__(self, ctx, width: int, height: int, format=None,
                 transfer_src: bool = False) -> None:
        self.width, self.height = int(width), int(height)
        self.format = format
        # `texture` is the Metal handle name and `image`/`view` the Vulkan
        # ones: the binding step reads whichever its adapter uses, and a
        # recording image that carried only one of them could not stand in for
        # the pass a coverage gate drives.
        self.image = self.view = self.texture = self
        super().__init__(ctx, width=self.width, height=self.height,
                         format=format, transfer_src=transfer_src)

    def read_rgba(self) -> np.ndarray:
        self._rec.record("read", "StorageImage",
                         width=self.width, height=self.height)
        return np.zeros((self.height, self.width, 4), dtype=np.float32)


class SampledImage(_Resource):
    def __init__(self, ctx, width: int, height: int, format=None,
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
        self.image = self.view = self.texture = self
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

    def _record_dispatch(self, groups, binds, *, uniform_blob=None,
                         push_bytes=None, bindless=None, output_image=None,
                         vars=None) -> None:
        """Record one dispatch and what it actually binds.

        "Bound" mirrors the real adapters rather than the shape of the ``binds``
        dict (codex pre-merge review, MEDIUM 2). Two rules, both taken from
        `metal_compute`:

        - a key whose value is ``None`` is **skipped**, not bound — so recording
          a key with no resource must not mask a genuinely missing binding;
        - a resource does not have to arrive through ``binds``. The uniform blob
          binds ``fc``, a push block binds ``pc``, the preview output binds
          ``previewOutput``, and ``bindless`` binds the global it names. Ignoring
          those channels reported legitimate bindings as absent.
        """
        bound = {name for name, res in (binds or {}).items() if res is not None}
        if uniform_blob is not None:
            bound.add("fc")
        if push_bytes is not None:
            bound.add("pc")
        if output_image is not None:
            bound.add("previewOutput")
        if bindless is not None:
            # (global_name, textures) — the pool binds under its own global.
            bound.add(bindless[0] if isinstance(bindless, (tuple, list))
                      else str(bindless))
        bound.update(name for name, v in (vars or {}).items() if v is not None)
        self._rec.record(
            "dispatch", type(self).__name__,
            entry=self.entry_point, groups=tuple(int(g) for g in groups),
            bound=sorted(bound),
            reflected=sorted(self._reflected),
        )


class ComputePipeline(_RecordingPipeline):
    def __init__(self, ctx, shader_dir, entry_module: str, entry_point: str,
                 graph_fragments=None, *, compile_pipeline: bool = True,
                 spectral: bool = False) -> None:
        self.shader_dir = shader_dir
        self.entry_module = entry_module
        self.spectral = bool(spectral)
        ctx.recorder.record("pipeline", entry_point, module=entry_module)
        super().__init__(ctx, entry_point, module=entry_module,
                         compile_pipeline=bool(compile_pipeline),
                         spectral=self.spectral)

    @classmethod
    def scene_bindings_only(cls, ctx, shader_dir, graph_fragments=None, *,
                            spectral: bool = False):
        return cls(ctx, shader_dir, "main_pass", "mainImage",
                   graph_fragments, compile_pipeline=False, spectral=spectral)

    def dispatch(self, width: int, height: int, uniform_blob=None, binds=None,
                 bindless=None, bands: int = 0, tile_origin_offset=None):
        self._record_dispatch((width, height, 1), binds,
                              uniform_blob=uniform_blob, bindless=bindless)

    def dispatch_indirect(self, args_buffer, offset: int = 0, bindings=None):
        self._record_dispatch((0, 0, 0), bindings)

    def dispatch_kernel(self, thread_count, buffers=None, vars=None):
        groups = thread_count if isinstance(thread_count, (tuple, list)) \
            else (thread_count, 1, 1)
        self._record_dispatch(groups, buffers, vars=vars)


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
        self._record_dispatch((size, size, 1), binds,
                              uniform_blob=uniform_blob, push_bytes=push_bytes,
                              bindless=bindless, output_image=output_image)


# ══════════════════════════════════════════════════════════════════════════
# Compute entry-point scan (change `recording-adapter-live-bindings`)
# ══════════════════════════════════════════════════════════════════════════
#
# The DECLARED globals of a pass are no longer parsed here. A hand parser could
# not tell a file-scope resource global from a function parameter of resource
# type without full scope tracking, and codex pre-merge review kept finding valid
# Slang spellings it under-reported — the exact fail-open the coverage gate
# exists to prevent. So the declared side now comes from the COMPILER's own
# reflection, generated offline into `tests/fixtures/recording_pass_globals.json`
# by `tests/fixtures/gen_recording_pass_globals.py` and trusted by the hostless
# gate exactly as the checked-in `.spv` is trusted. This module keeps only the
# entry-point SCAN the registry meta-test needs (a far simpler grammar than
# declarations): which `[shader("compute")]` entries exist, so an unregistered
# pass fails the build.


def shader_dir() -> Path:
    """The shader tree — the same directory the compile sites pass as ``-I``."""
    return Path(__file__).resolve().parent / "shaders"


#: Whitespace-tolerant, because Slang accepts `shader ("compute")` and
#: `[ shader ( "compute" ) ]`; a tight pattern would miss those and let a real
#: entry escape the registered-or-excluded check.
_ENTRY_RE = re.compile(
    r'\[\s*shader\s*\(\s*"compute"\s*\)\s*\][^{]*?\bvoid\s+([A-Za-z_]\w*)\s*\(',
    re.S)


def _strip_comments(text: str) -> str:
    """Remove `//` and `/* */` comments with a single left-to-right lexer.

    NOT two ordered regex passes: whichever comment form is stripped first, its
    delimiter can appear as literal text inside the other form and be
    mis-parsed. Concretely (codex review), a `/*` inside one `//` line and a
    `*/` inside a later `//` line made a block-first regex erase every line —
    and a compute entry — between them, silently failing the registry meta-test
    open. A lexer that recognises which context it is in cannot make that
    mistake: inside a line comment, `/*` is just text; inside a string literal,
    both are. String literals are tracked so `shader("compute")` survives (a
    `//` or `/*` inside a string would otherwise open a phantom comment).

    A removed comment becomes a single space, never nothing: an inline block
    comment is a legal token separator, so ``void/* */name`` must lex to
    ``void name`` — deleting it outright yields ``voidname`` and the entry
    vanishes before `_ENTRY_RE` (codex review). Newlines inside a block are kept
    on top of that space so line structure is preserved.
    """
    out: list[str] = []
    i, n, state = 0, len(text), "code"
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                out.append(" "); state = "line"; i += 2
            elif c == "/" and nxt == "*":
                out.append(" "); state = "block"; i += 2
            elif c == '"':
                out.append(c); state = "str"; i += 1
            else:
                out.append(c); i += 1
        elif state == "line":
            if c == "\n":
                out.append(c); state = "code"
            i += 1
        elif state == "block":
            if c == "*" and nxt == "/":
                state = "code"; i += 2
            else:
                if c == "\n":
                    out.append(c)   # preserve line structure
                i += 1
        else:  # "str"
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt); i += 2
            else:
                if c == '"':
                    state = "code"
                i += 1
    return "".join(out)


def _source_slang(base: Path) -> list[Path]:
    """The SOURCE ``.slang`` files under ``base`` — the ones a human wrote and
    committed, never the renderer's per-scene generated output.

    The renderer emits generated shaders INTO the tree at run time:
    ``generated_materials.slang`` and, per scene material, a
    ``wavefront/shade_<Material>.slang`` module — each carrying a
    ``[shader("compute")]`` entry. A plain ``rglob("*.slang")`` picks those up,
    so on any checkout that has ever rendered, the registry meta-test sees
    entries that are not (and must not be) registered and fails. Those files are
    never ``git add``ed, so the source set is exactly the **git-tracked**
    ``.slang`` — tracked-ness does not depend on ``.gitignore``, so this reads
    the same on the primary checkout and in a worktree whose ``.gitignore`` is
    ``*``. A directory that is not a git work tree (a tmp test root) falls back
    to a full rglob.
    """
    try:
        inside = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True)
        if inside.returncode == 0 and inside.stdout.strip() == "true":
            listed = subprocess.run(
                ["git", "-C", str(base), "ls-files", "-z", "--", "*.slang"],
                capture_output=True, text=True, check=True)
            return [base / rel for rel in listed.stdout.split("\0") if rel]
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(base.rglob("*.slang"))


def compute_entry_points(root=None) -> frozenset[tuple[str, str]]:
    """``(module, entry_point)`` for every SOURCE compute entry the shader tree
    declares, gates ignored (generated per-scene shaders excluded — see
    :func:`_source_slang`).

    Keyed by module AND name, never by name alone: four modules declare an
    entry called ``computeMain``, so a name-keyed map silently collapses three
    of them — an unregistered pass would then read as covered, which is the
    exact defect class this capability exists to remove.

    The registry's meta-test consumes this: an entry that is neither registered
    nor excluded fails the build, which is what makes coverage enforced rather
    than aspirational.
    """
    base = Path(root) if root is not None else shader_dir()
    found: set[tuple[str, str]] = set()
    for path in sorted(_source_slang(base)):
        # `".".join(parts)`, not `.replace("/", ".")`: the latter is POSIX-only
        # and leaves a `\` separator on Windows, so a nested module's key would
        # not match its dotted registry/exclusion entry (codex review).
        module = ".".join(path.relative_to(base).with_suffix("").parts)
        for name in _ENTRY_RE.findall(_strip_comments(path.read_text())):
            found.add((module, name))
    return frozenset(found)


# ══════════════════════════════════════════════════════════════════════════
# The host's real bind map, and the registry of recordable passes
# ══════════════════════════════════════════════════════════════════════════
#
# ── Survey: how a pass's bind map is built (task 2.1) ─────────────────────
#
# Answers the design's second open question. Every Metal dispatch in
# `renderer.py` binds through ONE builder — `Renderer._build_metal_binds`,
# which is `SceneResourceSet.metal_binds()` plus the two globals the renderer
# owns rather than the resource set (the shared bindless sampler and, when a
# MaterialX graph is live, the combined graph-param buffer). No pass
# constructs a binding inside its own dispatch call; the bindless pool arrives
# as the separate `bindless=` argument, which the recorder already counts.
#
# So the provider shape is a plain callable taking the recording context and
# returning `(binds, bindless)` — and the shared provider below reaches the
# real builder by constructing a `SceneResourceSet` against THIS adapter. No
# device, no `Renderer`, and no second copy of the bind map.

#: Stands in for the renderer-owned shared bindless sampler (binding 38). The
#: name is what coverage compares; the value only has to be non-``None``.
_COMMON_SAMPLER = object()


class _RecordingTexturePool:
    """The bindless pool, with no slots filled — enough for
    :meth:`SceneResourceSet.metal_bindless`, which reports the pool under its
    own global name."""

    def __init__(self, ctx, gpu) -> None:
        self._slots = [None] * BINDLESS_TEXTURE_CAPACITY

    def filled_slots(self):
        return []

    def destroy(self) -> None:
        pass


def _recording_sizes(**overrides):
    """A :class:`~skinny.gpu_resources.ResourceSizes` with every input at its
    smallest legal value.

    Sizes are irrelevant here — coverage compares binding NAMES — so this
    deliberately does not copy the renderer's size arithmetic, which
    `tests/test_gpu_resources.py` owns against a captured golden.
    """
    import dataclasses

    from skinny.gpu_resources import ResourceSizes

    kwargs = {}
    for f in dataclasses.fields(ResourceSizes):
        if f.default is not dataclasses.MISSING:
            continue
        kwargs[f.name] = False if f.type == "bool" else 1
    kwargs.update(overrides)
    return ResourceSizes(**kwargs)


def scene_binds(ctx, *, spectral: bool = False) -> tuple[dict, tuple]:
    """The bind map the host builds for a scene pass, allocated against the
    recording adapter — the production ``metal_binds`` path with no device.

    Returns ``(binds, bindless)`` in the shape ``ComputePipeline.dispatch``
    takes.
    """
    from skinny.gpu_resources import SceneResourceSet

    rset = SceneResourceSet(ctx, sys.modules[__name__], _recording_sizes(),
                            spectral=spectral,
                            texture_pool_factory=_RecordingTexturePool)
    return (rset.metal_binds(common_sampler=_COMMON_SAMPLER),
            rset.metal_bindless())


@dataclass(frozen=True)
class RecordablePass:
    """One pass the coverage gate drives: what it compiles, and how the host
    builds its bind map."""

    name: str
    entry_module: str
    entry_point: str
    key: ShaderVariantKey
    binds: Callable[[RecordingContext], tuple[dict, tuple]]

    @property
    def entry_key(self) -> tuple[str, str]:
        """``(module, entry_point)`` — the same key
        :func:`compute_entry_points` reports."""
        return (self.entry_module, self.entry_point)

    # The declared globals are NOT derived here: they come from the compiler's
    # reflection, checked in as `tests/fixtures/recording_pass_globals.json`
    # (regenerated by `gen_recording_pass_globals.py` with this pass's `key`).
    # `key` is retained because that generator compiles each pass under its
    # variant defines.


def _megakernel_key() -> ShaderVariantKey:
    return ShaderVariantKey(target=Target.METAL, family=Family.MEGAKERNEL)


#: Every pass whose binding coverage is asserted. The bind map is the Metal
#: name table, because that is the backend that binds by NAME — a Vulkan
#: dispatch binds through a descriptor set, where a missing write is a
#: validation error rather than the silent zero this gate exists to catch.
#:
#: One pass today: the megakernel. The denoise auxiliary/display passes register
#: here too (change ``denoise-pipeline``), but they are OUT OF SCOPE for this
#: change — coverage is scoped per registered ``(pass, ShaderVariantKey)`` and
#: the denoise shaders do not exist on this branch. They join when denoise lands.
RECORDABLE_PASSES: tuple[RecordablePass, ...] = (
    RecordablePass(
        name="megakernel",
        entry_module="main_pass",
        entry_point="mainImage",
        key=_megakernel_key(),
        binds=lambda ctx: scene_binds(ctx),
    ),
)


@dataclass(frozen=True)
class RecordableExclusion:
    """A compute entry point deliberately outside the coverage gate.

    Keyed by ``(module, entry_point)`` like :func:`compute_entry_points`, so
    the four ``computeMain`` entries stay four distinct facts.
    """

    module: str
    entry_point: str
    reason: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.module, self.entry_point)


_WAVEFRONT_REASON = (
    "wavefront pass: its bind map is assembled by the staged pass object "
    "(vk_wavefront / metal_wavefront), which owns the per-stream buffers in "
    "descriptor set 1 — SceneResourceSet supplies only set 0. Registering it "
    "needs a provider that drives that object; a follow-up")
_STANDALONE_REASON = (
    "standalone pipeline with its own descriptor set, not the scene bind map")

#: Entry points the gate does not drive, each with its reason. The meta-test
#: fails the build when an entry is neither registered above nor listed here,
#: and fails again when an entry here no longer exists in the tree — a stale
#: exclusion silently re-admits the gap it was meant to bound.
RECORDABLE_EXCLUSIONS: tuple[RecordableExclusion, ...] = tuple(
    RecordableExclusion(module, entry, reason)
    for module, entries, reason in (
        ("wavefront.wavefront_path",
         ("wfPathGenerate", "wfPathIntersect", "wfPathShade", "wfPathShadeFlat",
          "wfPathResolve", "wfScatter", "wfBuildArgs"), _WAVEFRONT_REASON),
        ("wavefront.compaction", ("compactAlive",), _WAVEFRONT_REASON),
        ("wavefront.scatter", ("scatterByMaterial",), _WAVEFRONT_REASON),
        ("wavefront.build_args", ("buildArgs",), _WAVEFRONT_REASON),
        ("wavefront.indirect_paint", ("wfIndirectPaint",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_visibility",
         ("wavefrontVisibility",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_normal", ("wavefrontNormal",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_material",
         ("wavefrontMaterial",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_env", ("wavefrontEnv",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_diffuse",
         ("wavefrontDiffuse",), _WAVEFRONT_REASON),
        ("wavefront.wavefront_bdpt",
         ("wfBdptGenEye", "wfBdptGenLight", "wfBdptWalk", "wfBdptWalkClassify",
          "wfBdptBounceEye", "wfBdptBounceLight", "wfBdptConnectNee",
          "wfBdptConnectFull", "wfBdptSplat", "wfBdptResolve",
          "wfBdptClassify", "wfBdptScatter", "wfBdptBuildArgs",
          "wfBdptLightTail"), _WAVEFRONT_REASON),
        ("integrators.wavefront_sppm",
         ("wfSppmEye", "wfSppmPhotonTrace", "wfSppmUpdate", "wfSppmGridCount",
          "wfSppmGridScatter", "wfSppmGridScanBlock",
          "wfSppmGridScanBlockSums", "wfSppmGridScanAdd"), _WAVEFRONT_REASON),
        ("wavefront.wavefront_mlt",
         ("wfMltBootstrap", "wfMltInit", "wfMltMutate", "wfMltResolve"),
         _WAVEFRONT_REASON),
        ("wavefront.neural_proposal_pass",
         ("wfNeuralProposal",), _WAVEFRONT_REASON),
        ("restir.restir_primary",
         ("restirFill", "restirSpatial", "restirResolve"), _WAVEFRONT_REASON),
        ("main_pass", ("mainImageRecord",),
         "the megakernel's record-dump entry: same module and same bind map as "
         "`mainImage`, which IS registered"),
        ("preview_pass", ("previewMain",),
         "the material-graph dock preview: PreviewPipeline binds an output "
         "image and a push block the scene bind map does not carry"),
        ("tool_probe", ("computeMain",),
         "the BXDF/pick tool probe: dispatched with the tool buffer alone, "
         "not the scene bind map"),
        ("foundation_trivial", ("computeMain", "addBias"),
         "device-probe kernels: no scene resources at all"),
        ("skin", ("skinMain",), _STANDALONE_REASON + " (vk_skinning)"),
        ("bvh_refit", ("refitMain",), _STANDALONE_REASON + " (vk_skinning)"),
        ("debug_raster",
         ("clearImage", "clearDepth", "depthLines", "colorLines", "blendTris"),
         _STANDALONE_REASON + " (DebugRasterMetal, uniform globals only)"),
    )
    for entry in entries
)
