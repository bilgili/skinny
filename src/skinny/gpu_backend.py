"""The declared interface at the Vulkan/Metal seam (change ``gpu-backend-adapter``).

``backend_select.resource_module(ctx)`` hands every construction site one of two
sibling adapter modules — :mod:`skinny.vk_compute` or :mod:`skinny.metal_compute`
— and a third, :mod:`skinny.recording_compute`, that records calls instead of
executing them. This module declares what those three have to agree on:

- :class:`BackendCapabilities` — the facts consumers used to rediscover with a
  vendor branch. **Every field replaces at least one pre-existing live branch**
  (design D1); no speculative capabilities. :func:`capabilities` derives the
  record from a context, folding the two runtime device probes
  (``supports_shared_memory``, ``supports_indirect_dispatch``) in, so a consumer
  reads one named fact instead of a backend test plus a probe.
- :data:`ONE_SIDED_MEMBERS` — the members that genuinely exist on one adapter
  only (design D5). Declared, never discovered: adding one is a deliberate edit,
  exactly like ``METAL_ONLY_DEFINES`` in :mod:`skinny.shader_variants`.
- :func:`adapter_surface` — an AST reader for an adapter's public surface. AST,
  not ``import``, because the conformance test must run on a host with neither
  the ``vulkan`` extension nor a Metal device.

Two probes are gone from every consumer and MUST NOT come back:

``hasattr(ctx, "compute_queue")``
    Used as "is this Vulkan?" at 7 sites, but ``MetalContext.compute_queue`` is
    ``None`` rather than absent, so the attribute exists and the test was
    **unconditionally true** (design D3). Replaced by
    :attr:`BackendCapabilities.has_descriptor_sets` — the Vulkan pass factories
    need descriptor sets, which is the actual reason they are Vulkan-only.

``descriptor_sets is None``
    Used as an "is Metal" sentinel at 13 gates, 5 of them compounded with
    ``is_metal`` in the same expression (the same fact stated twice). Replaced
    by the same named capability read (design D2).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "BackendCapabilities",
    "VULKAN_CAPABILITIES",
    "METAL_CAPABILITIES",
    "RECORDING_CAPABILITIES",
    "ADAPTER_MODULES",
    "ONE_SIDED_MEMBERS",
    "capabilities",
    "adapter_surface",
]


@dataclass(frozen=True)
class BackendCapabilities:
    """What the active GPU backend can do, by name rather than by vendor.

    Each field is the *reason* a consumer used to branch on ``is_metal``. Read
    it through :func:`capabilities`; never reconstruct one from ``is_metal``.
    """

    #: ``"vulkan"`` / ``"metal"`` / ``"recording"``. For display and for the
    #: adapter dispatch that genuinely has two implementations — never a
    #: substitute for one of the capability fields below.
    name: str

    #: Vulkan writes resource references into descriptor sets ahead of the
    #: dispatch; Metal re-reads every reference by name at dispatch time. Gates
    #: every descriptor write, the Vulkan-only pass factories, and the
    #: megakernel readiness question.
    has_descriptor_sets: bool

    #: Vulkan needs per-frame semaphores + in-flight fences. The Metal
    #: megakernel dispatch is submit-and-wait, so it builds none.
    has_frame_sync_objects: bool

    #: ``VK_KHR_external_memory`` for the CUDA neural-weight handoff.
    has_external_memory: bool

    #: Unified memory the interop publisher can overwrite in place at the frame
    #: boundary (Metal, and only when the device probe agrees).
    has_shared_in_place_write: bool

    #: ``vkCmdDispatchIndirect`` / a Metal encoder that honours indirect args.
    #: False falls the wavefront slot counts back to a CPU readback.
    has_indirect_dispatch: bool

    #: ``vk_skinning.py``'s UsdSkel compute path. No MSL counterpart exists, so
    #: Metal falls back to CPU skinning (design non-goal: no MSL kernel here).
    has_gpu_skinning: bool

    #: The megakernel ``mainImageRecord`` record source for online training.
    #: Metal has only the wavefront-native source.
    has_megakernel_record_source: bool

    #: MSL relocates the std-surface / MaterialX-skin record layouts, so their
    #: strides are known only from pipeline reflection. Vulkan reads the
    #: single-authored scalar layout.
    has_reflected_record_layouts: bool

    #: macOS cannot cancel another process's GPU work, so a long dispatch must
    #: be split into bounded command buffers (see CLAUDE.md, Metal dispatch
    #: hygiene). Gates the preview size clamp and the MLT/SPPM batch breadths.
    needs_watchdog_tiling: bool

    #: Slots in the bindless flat-material texture pool. 128 on Vulkan; 119 on
    #: Metal, whose compute argument table caps at 128 textures total. Must
    #: equal the array size compiled into ``shaders/bindings.slang`` for that
    #: target — enforced by test, not by a source comment.
    bindless_texture_capacity: int


VULKAN_CAPABILITIES = BackendCapabilities(
    name="vulkan",
    has_descriptor_sets=True,
    has_frame_sync_objects=True,
    has_external_memory=True,
    has_shared_in_place_write=False,
    has_indirect_dispatch=True,
    has_gpu_skinning=True,
    has_megakernel_record_source=True,
    has_reflected_record_layouts=False,
    needs_watchdog_tiling=False,
    bindless_texture_capacity=128,
)

METAL_CAPABILITIES = BackendCapabilities(
    name="metal",
    has_descriptor_sets=False,
    has_frame_sync_objects=False,
    has_external_memory=False,
    # Both of these are device probes on a real MetalContext; the static record
    # states the pessimistic default and `capabilities()` promotes them.
    has_shared_in_place_write=False,
    has_indirect_dispatch=False,
    has_gpu_skinning=False,
    has_megakernel_record_source=False,
    has_reflected_record_layouts=True,
    needs_watchdog_tiling=True,
    bindless_texture_capacity=119,
)

#: The recording adapter executes nothing, so it claims no device capability it
#: cannot honour. It keeps descriptor sets off (bind-by-name, like Metal) and
#: needs no watchdog tiling — recording a dispatch cannot wedge a GPU.
RECORDING_CAPABILITIES = BackendCapabilities(
    name="recording",
    has_descriptor_sets=False,
    has_frame_sync_objects=False,
    has_external_memory=False,
    has_shared_in_place_write=False,
    has_indirect_dispatch=True,
    has_gpu_skinning=False,
    has_megakernel_record_source=False,
    has_reflected_record_layouts=False,
    needs_watchdog_tiling=False,
    bindless_texture_capacity=119,
)

_BY_NAME = {
    "vulkan": VULKAN_CAPABILITIES,
    "metal": METAL_CAPABILITIES,
    "recording": RECORDING_CAPABILITIES,
}


def capabilities(ctx) -> BackendCapabilities:
    """Return the capability record for ``ctx``'s backend.

    Reads ``ctx.backend_name`` (falling back to ``is_metal`` for a stub context
    that predates it) and folds in the two runtime device probes, so a consumer
    never has to ask "is this Metal *and* does the device support X".
    """
    name = getattr(ctx, "backend_name", None)
    if name not in _BY_NAME:
        name = "metal" if getattr(ctx, "is_metal", False) else "vulkan"
    caps = _BY_NAME[name]
    if name == "metal":
        caps = replace(
            caps,
            has_shared_in_place_write=bool(
                getattr(ctx, "supports_shared_memory", False)),
            has_indirect_dispatch=bool(
                getattr(ctx, "supports_indirect_dispatch", False)),
        )
    return caps


#: The three sibling adapter modules, by backend name.
ADAPTER_MODULES = {
    "vulkan": "skinny.vk_compute",
    "metal": "skinny.metal_compute",
    "recording": "skinny.recording_compute",
}

#: Members that exist on one adapter only, with the reason (design D5). The
#: conformance test asserts the adapter surfaces agree **modulo this table**, so
#: a new one-sided member is a deliberate edit rather than silent drift.
#: Keys are ``"Class"`` or ``"Class.method"`` or ``"module_function"``.
ONE_SIDED_MEMBERS: dict[str, dict[str, str]] = {
    "ExternalTimelineSemaphore": {
        "only": "vulkan",
        "why": "VK_KHR_external_memory timeline handoff to CUDA; the Metal "
               "interop publisher writes UMA shared storage in place instead "
               "(capability has_external_memory / has_shared_in_place_write)",
    },
    "MetalFrameEncoder": {
        "only": "metal",
        "why": "one open Metal command encoder spanning a staged wavefront "
               "frame; Vulkan records into a command buffer it already owns",
    },
    "DebugRasterMetal": {
        "only": "metal",
        "why": "compute rasteriser for the Camera Debug dock; Vulkan uses its "
               "graphics pipeline, which the compute-only Metal context lacks",
    },
    "emit_wavefront_material_modules": {
        "only": "vulkan",
        "why": "SPIR-V per-material shade-module emission; the Metal wavefront "
               "passes compile MSL in-process from the same Slang sources",
    },
    "emit_wavefront_shade_module": {
        "only": "vulkan",
        "why": "see emit_wavefront_material_modules",
    },
    "ComputePipeline.dispatch": {
        "only": "metal",
        "why": "the Metal megakernel dispatch is submit-and-wait through the "
               "pipeline; the Vulkan renderer records vkCmdDispatch into the "
               "frame command buffer it owns (capability has_frame_sync_objects)",
    },
    "ComputePipeline.dispatch_indirect": {
        "only": "metal",
        "why": "see ComputePipeline.dispatch",
    },
    "ComputePipeline.dispatch_kernel": {
        "only": "metal",
        "why": "see ComputePipeline.dispatch",
    },
    "StorageBuffer.write_in_place": {
        "only": "metal",
        "why": "UMA shared-storage in-place write (capability "
               "has_shared_in_place_write); Vulkan uploads through a staging "
               "copy",
    },
    "HostStorageBuffer.write_in_place": {
        "only": "metal",
        "why": "see StorageBuffer.write_in_place",
    },
}


def _param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return [n for n in names if n != "self"]


def _public(name: str) -> bool:
    return not name.startswith("_")


def adapter_surface(module: str, *, src_root: Path | None = None) -> dict:
    """Return the public surface of an adapter module, read from its source.

    ``{"classes": {Cls: {method: [param, ...]}}, "functions": {...},
    "constants": [...]}``, private members excluded. Parsed with :mod:`ast` so
    the conformance test needs neither the ``vulkan`` extension nor a device.
    """
    root = src_root or Path(__file__).resolve().parent.parent
    path = root / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text())
    classes: dict[str, dict[str, list[str]]] = {}
    functions: dict[str, list[str]] = {}
    constants: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not _public(node.name):
                continue
            classes[node.name] = {
                m.name: _param_names(m)
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _public(m.name)
            }
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(node.name):
                functions[node.name] = _param_names(node)
        elif isinstance(node, ast.Assign):
            constants += [
                t.id for t in node.targets
                if isinstance(t, ast.Name) and t.id.isupper() and _public(t.id)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper() and _public(node.target.id):
                constants.append(node.target.id)
    return {
        "classes": classes,
        "functions": functions,
        "constants": sorted(set(constants)),
    }
