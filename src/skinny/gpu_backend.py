"""The declared interface at the Vulkan/Metal seam (change ``gpu-backend-adapter``).

``backend_select.resource_module(ctx)`` hands every construction site one of two
sibling adapter modules — :mod:`skinny.vk_compute` or :mod:`skinny.metal_compute`
— and a third, :mod:`skinny.recording_compute`, that records calls instead of
executing them. This module declares what those three have to agree on:

- :class:`BackendCapabilities` — the facts consumers used to rediscover with a
  vendor branch. **Every field replaces at least one pre-existing live branch**
  (design D1); no speculative capabilities. :func:`capabilities` derives the
  record from a context, folding the four runtime device probes
  (``supports_external_memory``, ``supports_external_semaphore``,
  ``supports_shared_memory``, ``supports_indirect_dispatch``) in **for every
  backend**, so a consumer reads one named fact instead of a backend test plus a
  probe — and so no device-dependent fact is asserted as a family constant.
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
    "DIVERGENT_SIGNATURES",
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

    #: ``VK_KHR_external_memory`` for the CUDA neural-weight handoff. A **device
    #: probe**, not a backend-family fact: a Vulkan device without the extension
    #: (MoltenVK, or any driver that fails the graded device build in
    #: `vk_context`) reports False. Static records state the pessimistic
    #: default; `capabilities()` promotes it from the context.
    has_external_memory: bool

    #: ``VK_KHR_external_semaphore`` + a timeline semaphore, for the
    #: CUDA-write↔Vulkan-read ordering the same handoff needs. Probed
    #: **independently** of the memory extension — `vk_context`'s graded
    #: fallback can enable memory without it — so the CUDA protocol requires
    #: BOTH, and neither implies the other (codex pre-merge review, HIGH 1).
    has_external_semaphore: bool

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
    # Device probes, NOT backend-family facts — `vk_context` builds the device
    # with a graded fallback and can end up with neither, memory only, or both.
    # The pessimistic default here is what `capabilities()` promotes.
    has_external_memory=False,
    has_external_semaphore=False,
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
    has_external_semaphore=False,
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
    has_external_semaphore=False,
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
    that predates it) and folds in the runtime device probes, so a consumer
    never has to ask "is this Metal *and* does the device support X".

    The probe fold is **symmetric across backends**. Every device-dependent
    field is promoted from the context for whichever backend is active, because
    a capability that is device-dependent on one target is not a family constant
    on the other: a Vulkan device may fail to enable external memory (MoltenVK,
    a missing extension, the graded device build in `vk_context`), and asserting
    it statically hands the CUDA publisher a buffer that was silently allocated
    non-exportable, deferring the failure to first publication (codex pre-merge
    review, HIGH 1). Each `getattr` default is the pessimistic one, so a stub
    context that declares no probe keeps the static record's value.
    """
    name = getattr(ctx, "backend_name", None)
    if name not in _BY_NAME:
        name = "metal" if getattr(ctx, "is_metal", False) else "vulkan"
    caps = _BY_NAME[name]
    return replace(
        caps,
        has_external_memory=bool(
            getattr(ctx, "supports_external_memory", caps.has_external_memory)),
        has_external_semaphore=bool(
            getattr(ctx, "supports_external_semaphore",
                    caps.has_external_semaphore)),
        has_shared_in_place_write=bool(
            getattr(ctx, "supports_shared_memory",
                    caps.has_shared_in_place_write)),
        has_indirect_dispatch=bool(
            getattr(ctx, "supports_indirect_dispatch",
                    caps.has_indirect_dispatch)),
    )


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
    "PreviewPipeline.dispatch": {
        "only": "metal",
        "why": "see ComputePipeline.dispatch — the descriptor-set backend "
               "records the preview dispatch into a command buffer it owns",
    },
    "make_sampler": {
        "only": "metal",
        "why": "one shared sampler for the bindless pool, because the Metal "
               "argument table splits combined Sampler2D into texture + "
               "sampler (design D8). Vulkan's pool holds combined samplers, "
               "so a standalone sampler has no meaning there",
    },
    "ComputePipeline.reflect_globals": {
        "only": "recording",
        "why": "declares the shader globals a recorded dispatch must cover "
               "(task 4.3). A device adapter reflects them from the compiled "
               "pipeline instead of being told",
    },
    "PreviewPipeline.reflect_globals": {
        "only": "recording",
        "why": "see ComputePipeline.reflect_globals",
    },
    "Call": {
        "only": "recording",
        "why": "the recording adapter's own log entry — scaffolding for "
               "assertions, not a GPU resource the device adapters could have",
    },
    "Recorder": {
        "only": "recording",
        "why": "see Call — holds the ordered call log",
    },
    "RecordingContext": {
        "only": "recording",
        "why": "a device-free stand-in for VulkanContext / MetalContext; the "
               "device adapters get their context from backend_select",
    },
    "COMMON_SAMPLER_NAME": {
        "only": "metal",
        "why": "the shader-global name the shared bindless sampler binds to; "
               "a bind-by-name concept with no descriptor-set counterpart",
    },
    "DISCRETE_MAP_SAMPLERS": {
        "only": "metal",
        "why": "see COMMON_SAMPLER_NAME — the discrete map sampler globals",
    },
}


#: Members present on every adapter whose *signature* genuinely differs, with
#: the reason. Same rule as :data:`ONE_SIDED_MEMBERS`: declared, never
#: discovered. Keep this table as short as the truth allows — a signature listed
#: here is one a caller cannot write backend-agnostically.
DIVERGENT_SIGNATURES: dict[str, str] = {
    "PreviewPipeline.__init__":
        "the binding models genuinely differ: a descriptor-set backend takes "
        "the scene set-0 layout and an output image view, a bind-by-name "
        "backend takes the scene's MaterialX graph fragments and resolves both "
        "at dispatch. The one call site branches on has_descriptor_sets; every "
        "other use of the class (pack_push, dispatch, destroy) is uniform.",
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


def _default_domain(node: ast.expr) -> str:
    """Classify a default expression into an *argument domain*.

    Parameter names alone cannot catch the divergence this change exists to
    remove: ``format`` took a ``VkFormat`` int on one adapter and a neutral
    string on the other, under the same name (codex pre-merge review, MEDIUM 3).
    The domain is what has to agree, so the conformance test compares this.

    Deliberately coarse — ``"str"`` / ``"int"`` / ``"none"`` / ``"bool"`` /
    ``"expr"`` — because the two adapters legitimately differ in the *value* of
    a default (``"clamp"`` vs ``"clamp_to_edge"`` would be a real difference,
    but ``16`` vs ``16`` is not interesting); what must never differ is the
    KIND, which is what a VkEnum int versus a token string is.
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return "none"
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, str):
            return "str"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
    return "expr"


def _param_domains(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """``{param: domain}`` for every parameter that declares a default."""
    a = fn.args
    out: dict[str, str] = {}
    positional = a.posonlyargs + a.args
    # defaults align to the TAIL of the positional list
    for p, d in zip(positional[len(positional) - len(a.defaults):], a.defaults):
        out[p.arg] = _default_domain(d)
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[p.arg] = _default_domain(d)
    return out


#: ``__init__`` is part of the surface — a constructor whose parameter names
#: drift is exactly the break the conformance test exists to catch.
_DUNDER_SURFACE = frozenset({"__init__"})


def _public(name: str) -> bool:
    return name in _DUNDER_SURFACE or not name.startswith("_")


def adapter_surface(module: str, *, src_root: Path | None = None) -> dict:
    """Return the public surface of an adapter module, read from its source.

    ``{"classes": {Cls: {method: [param, ...]}}, "functions": {...},
    "constants": [...], "domains": {"Cls.method"|"func": {param: domain}}}``,
    private members excluded. Parsed with :mod:`ast` so the conformance test
    needs neither the ``vulkan`` extension nor a device.

    ``domains`` carries the *argument domain* of every defaulted parameter (see
    :func:`_default_domain`), which is what makes the promised "one address-mode
    and format vocabulary" testable rather than aspirational.
    """
    root = src_root or Path(__file__).resolve().parent.parent
    path = root / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text())
    classes: dict[str, dict[str, list[str]]] = {}
    functions: dict[str, list[str]] = {}
    constants: list[str] = []
    domains: dict[str, dict[str, str]] = {}
    declared: dict[str, dict[str, list[str]]] = {}
    bases: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = {
                m.name: _param_names(m)
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _public(m.name)
            }
            declared[node.name] = members
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and _public(m.name):
                    d = _param_domains(m)
                    if d:
                        domains[f"{node.name}.{m.name}"] = d
            bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if not _public(node.name):
                continue
            classes[node.name] = members
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _public(node.name):
                functions[node.name] = _param_names(node)
                d = _param_domains(node)
                if d:
                    domains[node.name] = d
        elif isinstance(node, ast.Assign):
            constants += [
                t.id for t in node.targets
                if isinstance(t, ast.Name) and t.id.isupper() and _public(t.id)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper() and _public(node.target.id):
                constants.append(node.target.id)
    # An inherited member is part of the surface: a shared base in the same
    # module (the recording adapter's `_Resource`) must not read as a gap.
    def _inherited(name: str, seen: frozenset[str]) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for base in bases.get(name, ()):
            if base in seen or base not in declared:
                continue
            merged.update(_inherited(base, seen | {name}))
            merged.update(declared[base])
        return merged

    for name, members in classes.items():
        classes[name] = {**_inherited(name, frozenset()), **members}

    return {
        "domains": domains,
        "classes": classes,
        "functions": functions,
        "constants": sorted(set(constants)),
    }
