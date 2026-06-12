"""Vulkan compute pipeline — compiles Slang shaders to SPIR-V and manages dispatch."""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import vulkan as vk

from skinny.vk_context import VulkanContext


# Hard cap on the bindless flat-material texture array (binding 14). Each
# slot is one combined-image-sampler descriptor. Bumping this requires no
# shader change but consumes more descriptor slots.
BINDLESS_TEXTURE_CAPACITY = 128


def _vk_format_token(fmt):
    """Resolve a backend-neutral image-format token to a ``VkFormat``. Ints (an
    existing ``VkFormat``) pass through unchanged, so all current call sites and
    their SPIR-V/resources are byte-identical; the renderer passes string tokens
    so a construction site is the same on the Metal backend (see metal_compute)."""
    if isinstance(fmt, str):
        return {
            "rgba32f": vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            "rgba32_float": vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            "rgba8_unorm": vk.VK_FORMAT_R8G8B8A8_UNORM,
            "rgba8_srgb": vk.VK_FORMAT_R8G8B8A8_SRGB,
            "r8_unorm": vk.VK_FORMAT_R8_UNORM,
        }.get(fmt, vk.VK_FORMAT_R32G32B32A32_SFLOAT)
    return fmt


def _vk_address_token(mode):
    """Resolve a backend-neutral address-mode token to a ``VkSamplerAddressMode``;
    ints pass through unchanged (byte-identical)."""
    if isinstance(mode, str):
        return {
            "repeat": vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            "clamp": vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            "mirror": vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
            "black": vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            "useMetadata": vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
        }.get(mode, vk.VK_SAMPLER_ADDRESS_MODE_REPEAT)
    return mode

# The backend-agnostic megakernel-source emission (the `generated_materials` /
# per-graph / python-dispatcher Slang that `main_pass.slang` imports) lives in
# the Vulkan-free `megakernel_sources` module so the Metal backend can reuse it
# without importing `vulkan`. Re-exported here for backward compatibility with
# existing `from skinny.vk_compute import …` callers.
from skinny.megakernel_sources import (  # noqa: E402, F401  (re-export)
    GRAPH_BINDING_BASE,
    emit_megakernel_aggregator,
    emit_megakernel_sources,
    python_material_ids,
    scan_python_materials,
)


def emit_wavefront_material_modules(graph_fragments) -> str:
    """Build the wavefront half of the two-emitter split.

    From the SAME GraphFragment list, emit one self-contained
    `evalGraphSurface_<target>(P, N, T, UV, params, inout sp)` per graph: a
    wavefront shade kernel fetches a hit's graph params from its own buffer and
    calls this to overlay the graph outputs onto the seeded StdSurfaceParams.
    Unlike the megakernel switch, there is no compiled-in `graphId` dispatch —
    the runtime picks the per-material pipeline, so adding a material compiles
    just its module.

    P0 scope: the reusable per-material surface evaluation. The
    `[shader("compute")]` shade entry-point wrapper and the `evalBSDF_<target>`
    connection-event entry are deferred to Phase 1, where the wavefront
    path-state struct and descriptor layout they operate on are designed.
    """
    parts: list[str] = [
        "// Auto-generated. Do not edit — written by the wavefront emitter\n"
        "// (vk_compute.emit_wavefront_material_modules). Per-material surface\n"
        "// evaluation shared with the megakernel via the same GraphFragment.\n\n"
        "import mtlx_std_surface;  // StdSurfaceParams\n",
    ]
    for gf in graph_fragments:
        module_name = f"{gf.sanitized_name}_graph"
        assignments = "\n".join(
            f"    sp.{input_name} = g.{input_name};"
            for input_name, _ in gf.outputs
        )
        parts.append(
            f"\nimport generated.{module_name};\n\n"
            f"void applyGraphOutputs_{gf.sanitized_name}("
            f"inout StdSurfaceParams sp, in {gf.outputs_struct} g)\n"
            f"{{\n"
            f"{assignments}\n"
            f"}}\n\n"
            f"// Overlay graph {gf.target_name}'s outputs onto a seeded\n"
            f"// StdSurfaceParams. Called by the per-material wavefront shade\n"
            f"// kernel after it seeds `sp` from the hit's flat constants.\n"
            f"void evalGraphSurface_{gf.sanitized_name}("
            f"float3 P, float3 N, float3 T, float2 UV,\n"
            f"                            in {gf.struct_name} params,\n"
            f"                            inout StdSurfaceParams sp)\n"
            f"{{\n"
            f"    {gf.outputs_struct} g = {gf.func_name}(P, N, T, UV, params);\n"
            f"    applyGraphOutputs_{gf.sanitized_name}(sp, g);\n"
            f"}}\n"
        )
    return "".join(parts)


def emit_wavefront_shade_module(graph_fragment, graph_id: int, binding: int) -> str:
    """Per-material wavefront shade entry for ONE graph — the staged compile
    partition. Imports only this graph's module (generated.<name>_graph), so
    adding a material compiles one small kernel and leaves every other shade
    pipeline's SPIR-V untouched (vs. the megakernel switch, which recompiles on
    any graph-set change). The entry traces, shades only pixels whose material
    maps to `graph_id` (overlaying the graph's outputs on a seeded
    StdSurfaceParams), and writes the base colour; other pixels are left for the
    other materials' passes.
    """
    gf = graph_fragment
    name = gf.sanitized_name
    assignments = "\n".join(f"    sp.{i} = g.{i};" for i, _ in gf.outputs)
    return (
        "// Auto-generated per-material wavefront shade entry — imports only this\n"
        "// graph, so it is an independent compilation unit (the compile-win).\n"
        "import common;\n"
        "import bindings;\n"
        "import scene_trace;\n"
        "import cameras.pinhole;\n"
        "import mtlx_std_surface;\n"
        f"import generated.{name}_graph;\n\n"
        f"[[vk::binding({binding})]] StructuredBuffer<{gf.struct_name}> wfGraphParams_{name};\n\n"
        '[shader("compute")]\n'
        "[numthreads(8, 8, 1)]\n"
        f"void shadeSurface_{name}(uint3 tid: SV_DispatchThreadID)\n"
        "{\n"
        "    uint2 pixel = tid.xy;\n"
        "    if (pixel.x >= fc.width || pixel.y >= fc.height) return;\n"
        "    RNG rng = createRNG(pixel, fc.frameIndex);\n"
        "    PinholeCamera cam; float lensWeight;\n"
        "    Ray ray = cam.generateRay(pixel, rng, lensWeight);\n"
        "    HitInfo hit = traceScene(fc, ray);\n"
        "    if (!hit.hit) return;\n"
        f"    if (materialGraphId(hit.materialId) != {graph_id}u) return;\n"
        f"    {gf.outputs_struct} g = {gf.func_name}(hit.positionObject, "
        "normalize(hit.normal), hit.tangent, hit.uv, "
        f"wfGraphParams_{name}[hit.materialId]);\n"
        "    StdSurfaceParams sp = (StdSurfaceParams)0;\n"
        "    sp.base_color = float3(0.5);\n"
        f"{assignments}\n"
        "    accumBuffer[pixel] = float4(sp.base_color, 1.0);\n"
        "}\n"
    )


class ComputePipeline:
    """Wraps a single Vulkan compute pipeline compiled from a Slang entry point."""

    def __init__(
        self,
        ctx: VulkanContext,
        shader_dir: Path,
        entry_module: str,
        entry_point: str,
        graph_fragments: "list | None" = None,
        *,
        compile_pipeline: bool = True,
    ) -> None:
        self.ctx = ctx
        self.shader_dir = shader_dir
        self.entry_module = entry_module
        self.entry_point = entry_point
        # GraphFragment list (skinny.materialx_runtime). Each fragment is a
        # MaterialXGenSlang-extracted nodegraph evaluator that gets
        # concatenated into shaders/generated_materials.slang. Empty list ⇒
        # aggregator emits no per-graph code (still required by main_pass's
        # import).
        self.graph_fragments = list(graph_fragments) if graph_fragments else []

        import time as _time
        # Backend-independent scene plumbing, built unconditionally: regenerate
        # the python-material genslang, emit generated_materials + per-graph
        # modules + the python dispatcher (consumed by BOTH the megakernel and
        # the wavefront shade kernels), and expose the per-graph binding map
        # (`self.graph_bindings`).
        self.graph_bindings, self.python_material_modules = emit_megakernel_sources(
            self.shader_dir, self.graph_fragments
        )

        # Wavefront mode (`scene_bindings_only`): build the set-0 layout and
        # stop — no main_pass slangc compile and no megakernel driver pipeline.
        # The wavefront stage pipelines reuse this set-0 layout + the emitted
        # material modules.
        if not compile_pipeline:
            self.descriptor_set_layout = self._create_descriptor_set_layout()
            self._spirv_path = None
            self._shader_module = None
            self.pipeline_layout = None
            self.pipeline = None
            return

        t0 = _time.perf_counter()
        print(f"[skinny] slangc → SPIR-V: {entry_module}.slang …", flush=True)
        # Compile BEFORE creating the descriptor-set layout so a slangc failure
        # (caught by the renderer's empty-graph fallback) doesn't leak a layout.
        self._spirv_path = self._compile_slang()
        print(f"[skinny] slangc done in {_time.perf_counter() - t0:.2f}s "
              f"({self._spirv_path.stat().st_size // 1024} KB SPIR-V)", flush=True)
        t0 = _time.perf_counter()
        self._shader_module = self._create_shader_module()
        self.descriptor_set_layout = self._create_descriptor_set_layout()
        self.pipeline_layout = self._create_pipeline_layout()
        print(f"[skinny] driver pipeline compile …", flush=True)
        self.pipeline = self._create_pipeline()
        print(f"[skinny] pipeline ready in {_time.perf_counter() - t0:.2f}s", flush=True)

    @classmethod
    def scene_bindings_only(cls, ctx, shader_dir, graph_fragments=None):
        """Build ONLY the backend-independent scene plumbing — the set-0
        descriptor-set layout, the `generated_materials`/per-graph + python
        dispatcher emission, and the `graph_bindings` map — with NO megakernel
        `main_pass` slangc compile and NO driver compute pipeline.

        Wavefront mode owns one of these so it stands alone; the returned
        object has ``.pipeline is None``. Same emission + layout code as a full
        build, so the set-0 layout is byte-for-byte consistent across both
        backends and the wavefront stage pipelines."""
        return cls(
            ctx, shader_dir,
            entry_module="main_pass", entry_point="mainImage",
            graph_fragments=graph_fragments, compile_pipeline=False,
        )

    # ── Slang → SPIR-V compilation ───────────────────────────────

    # SPIR-V cache: every pipeline build hashes its (entry point + Slang
    # source tree + flags) into `<build>/spv_cache/<hash>.spv`. Pipeline
    # rebuilds (scene reload, mid-session graph swap) re-emit the same
    # generated_materials.slang for repeated scene sets, so the next
    # rebuild hits the cache and skips slangc (≈1.4 s).
    _CACHE_DIRNAME = "spv_cache"
    _CACHE_MAX_ENTRIES = 32  # roughly 32 × 5 MB = 160 MB worst case

    def _build_dir(self) -> Path:
        """Where the SPIR-V cache lives. Mirrors materialx_runtime._build_dir."""
        return Path(__file__).resolve().parents[2] / "build"

    def _cache_key(self, src: Path, flags: tuple[str, ...]) -> str:
        """Stable hash over the Slang source tree + compile flags.

        Walks every `.slang` file under shader_dir (including the
        aggregator + per-graph generated/ files written this turn) and
        every `.slang` under mtlx/genslang. Content hashing is necessary
        because some files (generated_materials.slang) change per scene
        without their mtime changing in a predictable way.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(self.entry_point.encode("utf-8"))
        h.update(b"\0")
        h.update(str(src).encode("utf-8"))
        h.update(b"\0")
        for flag in flags:
            h.update(flag.encode("utf-8"))
            h.update(b"\0")
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        roots = [self.shader_dir, mtlx_genslang]
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.slang")):
                h.update(str(path.relative_to(root)).encode("utf-8"))
                h.update(b"\0")
                h.update(path.read_bytes())
                h.update(b"\0")
        return h.hexdigest()

    def _compile_slang(self) -> Path:
        # Codegen already ran in __init__ (before emission) so the genslang
        # tree is current for both the megakernel and wavefront compiles.
        src = self.shader_dir / f"{self.entry_module}.slang"
        out = self.shader_dir / f"{self.entry_module}.spv"

        slangc = shutil.which("slangc")
        if slangc is None:
            raise RuntimeError("slangc not found on PATH — install the Slang compiler")
        # Include paths:
        #   - shader_dir: main_pass.slang and friends; also hosts the
        #     Slang-compatible `lib/mx_closure_type.glsl` shim.
        #   - mtlx/genslang/: MaterialXGenSlang impl files for skinny's
        #     custom nodedefs. Their `#include "lib/mx_closure_type.glsl"`
        #     resolves to the shader_dir shim above.
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        flags = (
            "-target", "spirv",
            "-entry", self.entry_point,
            "-stage", "compute",
            "-I", str(self.shader_dir),
            "-I", str(mtlx_genslang),
            # Tells the genslang impls to omit gen-prelude-only paths
            # (e.g. mx_environment_irradiance) so they compile standalone
            # in skinny's compute pipeline. The MaterialX gen path doesn't
            # set this and keeps the gen-provided helpers.
            "-D", "SKINNY_COMPUTE_PIPELINE=1",
            "-fvk-use-scalar-layout",
        )

        cache_dir = self._build_dir() / self._CACHE_DIRNAME
        key = self._cache_key(src, flags)
        cached = cache_dir / f"{key}.spv"
        if cached.exists():
            shutil.copyfile(cached, out)
            # Bump mtime so LRU eviction keeps recently-used entries.
            try:
                cached.touch()
            except OSError:
                pass
            return out

        cmd = [slangc, str(src), *flags, "-o", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Slang compilation failed:\n{result.stderr}")
        # Populate cache for next pipeline build over the same source tree.
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, cached)
            # LRU evict: keep the cache bounded so a long-running session
            # iterating many distinct graph sets doesn't fill the disk.
            # `_CACHE_MAX_ENTRIES × ~5 MB` is the upper bound. Order by
            # mtime (Vulkan touches files via copyfile both on hit + miss
            # in our scheme via touch on hit — see `_compile_slang` cache-
            # hit branch).
            entries = sorted(cache_dir.glob("*.spv"),
                             key=lambda p: p.stat().st_mtime)
            for old in entries[: max(0, len(entries) - self._CACHE_MAX_ENTRIES)]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except OSError:
            # Cache writes are best-effort; transient FS errors must not
            # break the pipeline build itself.
            pass
        return out

    # ── Shader module ────────────────────────────────────────────

    def _create_shader_module(self):
        spirv_bytes = self._spirv_path.read_bytes()

        create_info = vk.VkShaderModuleCreateInfo(
            codeSize=len(spirv_bytes),
            pCode=spirv_bytes,
        )
        return vk.vkCreateShaderModule(self.ctx.device, create_info, None)

    # ── Descriptor set layout ────────────────────────────────────

    def _create_descriptor_set_layout(self):
        bindings = [
            # binding 0: uniform buffer (FrameConstants + SkinParams + light)
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 1: storage image (swapchain output)
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 2: storage image (HDR accumulation buffer)
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 3: storage image (R8 HUD alpha overlay)
            vk.VkDescriptorSetLayoutBinding(
                binding=3,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 4: combined image sampler (HDR environment map)
            vk.VkDescriptorSetLayoutBinding(
                binding=4,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 5: storage buffer (mesh vertices)
            vk.VkDescriptorSetLayoutBinding(
                binding=5,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 6: storage buffer (mesh triangle indices)
            vk.VkDescriptorSetLayoutBinding(
                binding=6,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 7: storage buffer (BVH nodes)
            vk.VkDescriptorSetLayoutBinding(
                binding=7,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 8: combined image sampler (tattoo RGBA texture)
            vk.VkDescriptorSetLayoutBinding(
                binding=8,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 9: combined image sampler (tangent-space normal map)
            vk.VkDescriptorSetLayoutBinding(
                binding=9,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 10: combined image sampler (roughness map)
            vk.VkDescriptorSetLayoutBinding(
                binding=10,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 11: combined image sampler (displacement map)
            vk.VkDescriptorSetLayoutBinding(
                binding=11,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 12: storage buffer (TLAS instance records)
            vk.VkDescriptorSetLayoutBinding(
                binding=12,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 13: storage buffer (per-material flat-shading params)
            vk.VkDescriptorSetLayoutBinding(
                binding=13,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 14: bindless flat-material texture array. Descriptor
            # count is the hard cap; PARTIALLY_BOUND lets us leave unused
            # slots empty (the shader gates reads behind a sentinel index).
            vk.VkDescriptorSetLayoutBinding(
                binding=14,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=BINDLESS_TEXTURE_CAPACITY,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 15: per-material skin UBO array. StructuredBuffer<T>
            # (storage buffer, std430 layout) with one MtlxSkinParams
            # record per material slot. Only skin-typed slots carry data.
            # Layout matches the gen-reflected M_skinny_skin_default
            # uniform_block.
            vk.VkDescriptorSetLayoutBinding(
                binding=15,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 16: per-material-slot type codes (uint each, see
            # MATERIAL_TYPE_* in renderer.py). The shader reads
            # materialTypes[hit.materialId] to dispatch between the skin
            # path and evalFlatMaterial.
            vk.VkDescriptorSetLayoutBinding(
                binding=16,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 17: UsdLux.SphereLight records. 32 B each
            # (vec3 position, float radius, vec3 radiance, float pad);
            # capacity SPHERE_LIGHT_CAPACITY. fc.numSphereLights bounds
            # the active range. Empty when no SphereLight prims authored.
            vk.VkDescriptorSetLayoutBinding(
                binding=17,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 18: Emissive triangle records for NEE. 64 B each
            # (v0+pad, v1+pad, v2+pad, emission+area). Built at scene
            # load from instances whose material has non-zero emissiveColor.
            vk.VkDescriptorSetLayoutBinding(
                binding=18,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 19: StdSurfaceParams records. 256 B each; one per
            # material slot. Carries the full MaterialX standard_surface
            # input set for evalStdSurfaceBSDF (mtlx_std_surface.slang).
            vk.VkDescriptorSetLayoutBinding(
                binding=19,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 21: BDPT light-tracer splat buffer. Per-pixel 3 × uint32
            # fixed-point cumulative radiance (Q22.10). Atomic-added by
            # bdpt.slang's splatLightVertex(), consumed by main_pass.slang's
            # final composite. Cleared on accumulation reset.
            vk.VkDescriptorSetLayoutBinding(
                binding=21,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 22: gizmo segment list (read-only). 32-byte records:
            # float2 a + float2 b + float3 color + float halfWidth.
            vk.VkDescriptorSetLayoutBinding(
                binding=22,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 23: thick-lens elements (read-only). float4 records:
            # (radius, thickness, ior, halfAperture) all in world units;
            # consumed by cameras/thick_lens.slang::generateLensRay.
            vk.VkDescriptorSetLayoutBinding(
                binding=23,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 24: per-film-radius exit-pupil bounds (read-only).
            # float4 (xMin, xMax, yMin, yMax) per bin; PBRT
            # `BoundExitPupil`. Used by `SampleExitPupil`.
            vk.VkDescriptorSetLayoutBinding(
                binding=24,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 20: UsdLux.DistantLight records. 32 B each
            # (vec3 direction, float pad, vec3 radiance, float pad);
            # capacity DISTANT_LIGHT_CAPACITY. fc.numDistantLights bounds
            # the active range. Replaces the legacy lightDirection /
            # lightRadiance UBO uniforms so multiple authored distant
            # lights all contribute (iterated as DirectionalLightImpl).
            # Picked from a free slot below GRAPH_BINDING_BASE (=25).
            vk.VkDescriptorSetLayoutBinding(
                binding=20,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # binding 30: BXDF visualizer output buffer (host-visible
            # RWStructuredBuffer<float4>). Main pass writes the picked
            # pixel's HitInfo into slots [0..3]. Future BXDF / BSSRDF eval
            # writes lobe grids starting at slot 32. Bound for all
            # pipelines that use this layout; unused by mainImage paths
            # when pickArmed == 0.
            vk.VkDescriptorSetLayoutBinding(
                binding=30,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # bindings 31/32: environment importance-sampling distribution.
            # 31 = marginal CDF over rows (float[ENV_H+1]); 32 = conditional
            # CDF over columns per row (float[ENV_H*(ENV_W+1)]). Built by
            # environment.build_env_distribution, consumed by environment.slang
            # sampleEnvDir/envPdf for env NEE + MIS. Above the graph range
            # (25..29) and the tool buffer (30).
            vk.VkDescriptorSetLayoutBinding(
                binding=31,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=32,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # bindings 33/34/35: neural directional-proposal frozen weights —
            # weights[] / biases[] / NfLayerHeader[]. Read by the inline flow
            # inverse in sampling/proposal.slang (so referenced by every pipeline
            # that uses this layout, megakernel included); the renderer binds
            # 1-element dummies until the neural proposal is activated. Placed
            # ABOVE the MaterialX graph range (25..29) + env CDFs (31/32).
            vk.VkDescriptorSetLayoutBinding(
                binding=33,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=34,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=35,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            # bindings 36/37: neural training-record dump (task 5.1) — the
            # PathRecord append buffer + its counter ([0]=count, [1]=capacity).
            # Written only by the `mainImageRecord` megakernel entry
            # (integrators/path_record.slang); `mainImage` never references them
            # (dead-stripped → byte-identical). The renderer binds 1-element
            # dummies until a dump runs. See Architecture.md binding map.
            vk.VkDescriptorSetLayoutBinding(
                binding=36,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=37,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        # Bindings GRAPH_BINDING_BASE..(GRAPH_BINDING_BASE + N - 1): one
        # storage buffer per MaterialX nodegraph compiled into this pipeline.
        # Each holds an array of GraphParams_<sanitized> records indexed by
        # material slot (matches the FlatMaterialParams / StdSurfaceParams
        # pattern at bindings 13 / 19).
        for idx in range(len(self.graph_fragments)):
            bindings.append(
                vk.VkDescriptorSetLayoutBinding(
                    binding=GRAPH_BINDING_BASE + idx,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
            )

        # Only binding 14 (the bindless combined-image-sampler array) needs
        # UPDATE_AFTER_BIND — that's what pushes the sampler count past
        # MoltenVK's regular maxPerStageDescriptorSamplers limit (16).
        # Each other descriptor type would require its own device feature
        # flag enabled, so we leave them at 0.
        binding_flags = []
        for b in bindings:
            if b.binding == 14:
                flag = (vk.VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                        | vk.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
            else:
                flag = 0
            binding_flags.append(flag)
        flags_info = vk.VkDescriptorSetLayoutBindingFlagsCreateInfo(
            bindingCount=len(binding_flags),
            pBindingFlags=binding_flags,
        )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
            flags=0x00000002,  # VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
            pNext=flags_info,
            bindingCount=len(bindings),
            pBindings=bindings,
        )
        return vk.vkCreateDescriptorSetLayout(self.ctx.device, layout_info, None)

    # ── Pipeline layout ──────────────────────────────────────────

    def _create_pipeline_layout(self):
        layout_info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[self.descriptor_set_layout],
        )
        return vk.vkCreatePipelineLayout(self.ctx.device, layout_info, None)

    # ── Pipeline ─────────────────────────────────────────────────

    def _create_pipeline(self):
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self._shader_module,
            pName="main",
        )

        pipeline_info = vk.VkComputePipelineCreateInfo(
            stage=stage_info,
            layout=self.pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self.ctx.device, vk.VK_NULL_HANDLE, 1, [pipeline_info], None
        )
        return pipelines[0]

    # ── Cleanup ──────────────────────────────────────────────────

    def destroy(self) -> None:
        # A `scene_bindings_only` build has no compiled pipeline / module /
        # pipeline-layout (those are None); it owns only the set-0 layout.
        if self.pipeline is not None:
            vk.vkDestroyPipeline(self.ctx.device, self.pipeline, None)
        if self.pipeline_layout is not None:
            vk.vkDestroyPipelineLayout(self.ctx.device, self.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self.descriptor_set_layout, None)
        if self._shader_module is not None:
            vk.vkDestroyShaderModule(self.ctx.device, self._shader_module, None)


class UniformBuffer:
    """Host-visible Vulkan buffer for uploading uniform data each frame."""

    def __init__(self, ctx: VulkanContext, size_bytes: int) -> None:
        self.ctx = ctx
        self.size = size_bytes

        buf_info = vk.VkBufferCreateInfo(
            size=size_bytes,
            usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)

        mem_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mem_type_index = self._find_memory_type(
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )

        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.memory = vk.vkAllocateMemory(ctx.device, alloc_info, None)
        vk.vkBindBufferMemory(ctx.device, self.buffer, self.memory, 0)

    @staticmethod
    def _find_memory_type(
        type_filter: int, properties: int, mem_props
    ) -> int:
        for i in range(mem_props.memoryTypeCount):
            if (type_filter & (1 << i)) and (
                mem_props.memoryTypes[i].propertyFlags & properties
            ) == properties:
                return i
        raise RuntimeError("Failed to find suitable memory type")

    def upload(self, data: bytes) -> None:
        ptr = vk.vkMapMemory(self.ctx.device, self.memory, 0, self.size, 0)
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, data, min(len(data), self.size))
        vk.vkUnmapMemory(self.ctx.device, self.memory)

    def destroy(self) -> None:
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)


class StorageImage:
    """Device-local 2D storage image. Used for persistent HDR accumulation."""

    def __init__(
        self,
        ctx: VulkanContext,
        width: int,
        height: int,
        format: int = vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        transfer_src: bool = False,
    ) -> None:
        format = _vk_format_token(format)  # accept backend-neutral tokens (int passes through)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.format = format

        usage = vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        if transfer_src:
            usage |= vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT

        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=format,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.image = vk.vkCreateImage(ctx.device, img_info, None)

        mem_reqs = vk.vkGetImageMemoryRequirements(ctx.device, self.image)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mem_type_index = UniformBuffer._find_memory_type(
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mem_props,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.memory = vk.vkAllocateMemory(ctx.device, alloc_info, None)
        vk.vkBindImageMemory(ctx.device, self.image, self.memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=format,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self.view = vk.vkCreateImageView(ctx.device, view_info, None)

        self._transition_to_general()

    def _transition_to_general(self) -> None:
        """One-shot transition from UNDEFINED to GENERAL for persistent storage use."""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]

        begin_info = vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        )
        vk.vkBeginCommandBuffer(cmd, begin_info)

        barrier = vk.VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier],
        )
        vk.vkEndCommandBuffer(cmd)

        submit_info = vk.VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[cmd],
        )
        vk.vkQueueSubmit(self.ctx.compute_queue, 1, [submit_info], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def destroy(self) -> None:
        vk.vkDestroyImageView(self.ctx.device, self.view, None)
        vk.vkDestroyImage(self.ctx.device, self.image, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)


class ReadbackBuffer:
    """Host-visible staging buffer for GPU-to-CPU image readback."""

    def __init__(self, ctx: VulkanContext, width: int, height: int, bytes_per_pixel: int = 4):
        self.ctx = ctx
        self.width = width
        self.height = height
        self._size = width * height * bytes_per_pixel

        buf_info = vk.VkBufferCreateInfo(
            size=self._size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)

        mem_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mem_type_index = UniformBuffer._find_memory_type(
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.memory = vk.vkAllocateMemory(ctx.device, alloc_info, None)
        vk.vkBindBufferMemory(ctx.device, self.buffer, self.memory, 0)

    def record_copy_from(self, cmd, src_image) -> None:
        """Record vkCmdCopyImageToBuffer. Caller must insert appropriate barriers."""
        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
        )
        vk.vkCmdCopyImageToBuffer(
            cmd, src_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self.buffer, 1, [region],
        )

    def read(self) -> bytes:
        """Map the staging buffer and return a copy of the pixel data."""
        data_ptr = vk.vkMapMemory(self.ctx.device, self.memory, 0, self._size, 0)
        import cffi
        ffi = cffi.FFI()
        buf = ffi.new(f"char[{self._size}]")
        ffi.memmove(buf, data_ptr, self._size)
        vk.vkUnmapMemory(self.ctx.device, self.memory)
        return bytes(ffi.buffer(buf, self._size))

    def destroy(self) -> None:
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)


class PreviewPipeline:
    """Secondary compute pipeline for the Material Graph Editor preview.

    Shares descriptor set 0 with the main `ComputePipeline` (passed in by
    the renderer) so all material data — bindings 0/4/13/14/15/16/19,
    plus per-graph SSBOs starting at `GRAPH_BINDING_BASE` — is visible
    without duplicating writes. Owns one extra set (set 1) carrying the
    preview output `RWTexture2D` and a push-constants range carrying the
    chosen material + primitive + camera.

    Keep `_PUSH_FMT` in sync with `preview_pass.slang::PreviewPushConsts`.
    """

    _PUSH_FMT = "<IIIIffff"  # uint matId, graphId, primKind, size + float yaw, pitch, distance, fovTan
    _PUSH_SIZE = 32

    def __init__(
        self,
        ctx: VulkanContext,
        shader_dir: Path,
        main_descriptor_set_layout,
        output_image_view,
    ) -> None:
        self.ctx = ctx
        self.shader_dir = shader_dir
        self.entry_module = "preview_pass"
        self.entry_point = "previewMain"
        self._main_set_layout = main_descriptor_set_layout
        self._output_view = output_image_view

        self._spirv_path = self._compile_slang()
        self._shader_module = self._create_shader_module()
        self.set1_layout = self._create_set1_layout()
        self.pipeline_layout = self._create_pipeline_layout()
        self.pipeline = self._create_pipeline()
        self.descriptor_pool, self.descriptor_set = self._allocate_set1()

    # ── Slang → SPIR-V ───────────────────────────────────────────

    def _build_dir(self) -> Path:
        return Path(__file__).resolve().parents[2] / "build"

    def _cache_key(self, src: Path, flags: tuple[str, ...]) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.entry_point.encode("utf-8"))
        h.update(b"\0")
        h.update(str(src).encode("utf-8"))
        h.update(b"\0")
        for flag in flags:
            h.update(flag.encode("utf-8"))
            h.update(b"\0")
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        roots = [self.shader_dir, mtlx_genslang]
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.slang")):
                h.update(str(path.relative_to(root)).encode("utf-8"))
                h.update(b"\0")
                h.update(path.read_bytes())
                h.update(b"\0")
        return h.hexdigest()

    def _compile_slang(self) -> Path:
        src = self.shader_dir / f"{self.entry_module}.slang"
        out = self.shader_dir / f"{self.entry_module}.spv"
        slangc = shutil.which("slangc")
        if slangc is None:
            raise RuntimeError("slangc not found on PATH — install the Slang compiler")
        mtlx_genslang = self.shader_dir.parent / "mtlx" / "genslang"
        flags = (
            "-target", "spirv",
            "-entry", self.entry_point,
            "-stage", "compute",
            "-I", str(self.shader_dir),
            "-I", str(mtlx_genslang),
            "-D", "SKINNY_COMPUTE_PIPELINE=1",
            "-fvk-use-scalar-layout",
        )
        cache_dir = self._build_dir() / ComputePipeline._CACHE_DIRNAME
        key = self._cache_key(src, flags)
        cached = cache_dir / f"{key}.spv"
        if cached.exists():
            shutil.copyfile(cached, out)
            try:
                cached.touch()
            except OSError:
                pass
            return out
        cmd = [slangc, str(src), *flags, "-o", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Slang compilation failed:\n{result.stderr}")
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, cached)
        except OSError:
            pass
        return out

    def _create_shader_module(self):
        spirv_bytes = self._spirv_path.read_bytes()
        create_info = vk.VkShaderModuleCreateInfo(
            codeSize=len(spirv_bytes),
            pCode=spirv_bytes,
        )
        return vk.vkCreateShaderModule(self.ctx.device, create_info, None)

    # ── Set 1 layout / pool / set ────────────────────────────────

    def _create_set1_layout(self):
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        info = vk.VkDescriptorSetLayoutCreateInfo(
            bindingCount=1,
            pBindings=[binding],
        )
        return vk.vkCreateDescriptorSetLayout(self.ctx.device, info, None)

    def _allocate_set1(self):
        pool_size = vk.VkDescriptorPoolSize(
            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1,
        )
        pool_info = vk.VkDescriptorPoolCreateInfo(
            maxSets=1,
            poolSizeCount=1,
            pPoolSizes=[pool_size],
        )
        pool = vk.vkCreateDescriptorPool(self.ctx.device, pool_info, None)
        alloc_info = vk.VkDescriptorSetAllocateInfo(
            descriptorPool=pool,
            descriptorSetCount=1,
            pSetLayouts=[self.set1_layout],
        )
        ds = vk.vkAllocateDescriptorSets(self.ctx.device, alloc_info)[0]

        img_info = vk.VkDescriptorImageInfo(
            sampler=vk.VK_NULL_HANDLE,
            imageView=self._output_view,
            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
        )
        write = vk.VkWriteDescriptorSet(
            dstSet=ds,
            dstBinding=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[img_info],
        )
        vk.vkUpdateDescriptorSets(self.ctx.device, 1, [write], 0, None)
        return pool, ds

    # ── Pipeline layout / pipeline ───────────────────────────────

    def _create_pipeline_layout(self):
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=self._PUSH_SIZE,
        )
        info = vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=2,
            pSetLayouts=[self._main_set_layout, self.set1_layout],
            pushConstantRangeCount=1,
            pPushConstantRanges=[push_range],
        )
        return vk.vkCreatePipelineLayout(self.ctx.device, info, None)

    def _create_pipeline(self):
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self._shader_module,
            pName="main",
        )
        info = vk.VkComputePipelineCreateInfo(
            stage=stage,
            layout=self.pipeline_layout,
        )
        pipelines = vk.vkCreateComputePipelines(
            self.ctx.device, vk.VK_NULL_HANDLE, 1, [info], None,
        )
        return pipelines[0]

    @staticmethod
    def pack_push(matId: int, graphId: int, primKind: int, size: int,
                  yaw: float, pitch: float, distance: float, fovTan: float) -> bytes:
        return struct.pack(
            PreviewPipeline._PUSH_FMT,
            int(matId), int(graphId), int(primKind), int(size),
            float(yaw), float(pitch), float(distance), float(fovTan),
        )

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self.descriptor_pool, None)
        vk.vkDestroyPipeline(self.ctx.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self.set1_layout, None)
        vk.vkDestroyShaderModule(self.ctx.device, self._shader_module, None)


class HudOverlay:
    """R8_UNORM alpha mask updated each frame from the host (Pillow → staging → image)."""

    def __init__(self, ctx: VulkanContext, width: int, height: int) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self._byte_count = width * height

        # Host-visible staging buffer (persistently mapped) for CPU writes.
        buf_info = vk.VkBufferCreateInfo(
            size=self._byte_count,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.staging_buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)

        mem_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.staging_buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        self._staging_size = mem_reqs.size
        mem_type_index = UniformBuffer._find_memory_type(
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        alloc_info = vk.VkMemoryAllocateInfo(
            allocationSize=mem_reqs.size,
            memoryTypeIndex=mem_type_index,
        )
        self.staging_memory = vk.vkAllocateMemory(ctx.device, alloc_info, None)
        vk.vkBindBufferMemory(ctx.device, self.staging_buffer, self.staging_memory, 0)

        # Device-local R8_UNORM image that the shader reads as an alpha mask.
        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R8_UNORM,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.image = vk.vkCreateImage(ctx.device, img_info, None)

        img_reqs = vk.vkGetImageMemoryRequirements(ctx.device, self.image)
        img_mem_type = UniformBuffer._find_memory_type(
            img_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mem_props,
        )
        img_alloc = vk.VkMemoryAllocateInfo(
            allocationSize=img_reqs.size,
            memoryTypeIndex=img_mem_type,
        )
        self.image_memory = vk.vkAllocateMemory(ctx.device, img_alloc, None)
        vk.vkBindImageMemory(ctx.device, self.image, self.image_memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=vk.VK_FORMAT_R8_UNORM,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self.view = vk.vkCreateImageView(ctx.device, view_info, None)

        # One-time transition UNDEFINED → GENERAL; per-frame record_copy handles
        # the GENERAL↔TRANSFER_DST flips before the copy and back before dispatch.
        self._transition_initial()

    def _transition_initial(self) -> None:
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        barrier = vk.VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier],
        )
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def upload(self, data: bytes) -> None:
        if len(data) != self._byte_count:
            raise ValueError(
                f"HUD payload is {len(data)} bytes, expected {self._byte_count}"
            )
        ptr = vk.vkMapMemory(self.ctx.device, self.staging_memory, 0, self._staging_size, 0)
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self.ctx.device, self.staging_memory)

    def record_copy(self, cmd) -> None:
        """Inside a recording command buffer: GENERAL → TRANSFER_DST → copy → GENERAL."""
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        )
        to_dst = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            image=self.image,
            subresourceRange=subresource,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, None, 0, None, 1, [to_dst],
        )

        region = vk.VkBufferImageCopy(
            bufferOffset=0,
            bufferRowLength=0,
            bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=0,
                layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
        )
        vk.vkCmdCopyBufferToImage(
            cmd,
            self.staging_buffer,
            self.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [region],
        )

        to_general = vk.VkImageMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            image=self.image,
            subresourceRange=subresource,
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [to_general],
        )

    def destroy(self) -> None:
        vk.vkDestroyImageView(self.ctx.device, self.view, None)
        vk.vkDestroyImage(self.ctx.device, self.image, None)
        vk.vkFreeMemory(self.ctx.device, self.image_memory, None)
        vk.vkDestroyBuffer(self.ctx.device, self.staging_buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.staging_memory, None)


class SampledImage:
    """Device-local 2D image + linear sampler, re-uploadable from host.

    Used for HDR environment maps (RGBA32F) and skin detail maps (RGBA8). The
    default preserves the old behaviour so existing callers don't break.
    Uploads are synchronous — rare (on env / texture / model switch) — and the
    image lives in SHADER_READ_ONLY_OPTIMAL between swaps.
    """

    def __init__(
        self,
        ctx: VulkanContext,
        width: int,
        height: int,
        format: int = vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        bytes_per_pixel: int = 16,
        address_mode_u: int = vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
        address_mode_v: int = vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    ) -> None:
        # Accept backend-neutral string tokens (the renderer passes these so a
        # SampledImage construction site is identical on Metal); existing int
        # VkFormat / VkSamplerAddressMode args pass through unchanged.
        format = _vk_format_token(format)
        address_mode_u = _vk_address_token(address_mode_u)
        address_mode_v = _vk_address_token(address_mode_v)
        self.ctx = ctx
        self.width = width
        self.height = height
        self.format = format
        self.bytes_per_pixel = bytes_per_pixel
        self._address_mode_u = address_mode_u
        self._address_mode_v = address_mode_v
        self._byte_count = width * height * bytes_per_pixel

        # Staging buffer
        buf_info = vk.VkBufferCreateInfo(
            size=self._byte_count,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.staging_buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)
        mem_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.staging_buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        self._staging_size = mem_reqs.size
        staging_type = UniformBuffer._find_memory_type(
            mem_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        self.staging_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=mem_reqs.size,
                memoryTypeIndex=staging_type,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self.staging_buffer, self.staging_memory, 0)

        # Device-local image (sampled + transfer dst)
        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=format,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self.image = vk.vkCreateImage(ctx.device, img_info, None)
        img_reqs = vk.vkGetImageMemoryRequirements(ctx.device, self.image)
        img_type = UniformBuffer._find_memory_type(
            img_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mem_props,
        )
        self.image_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=img_reqs.size,
                memoryTypeIndex=img_type,
            ),
            None,
        )
        vk.vkBindImageMemory(ctx.device, self.image, self.image_memory, 0)

        view_info = vk.VkImageViewCreateInfo(
            image=self.image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
            format=format,
            components=vk.VkComponentMapping(
                r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
            ),
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        self.view = vk.vkCreateImageView(ctx.device, view_info, None)

        # Linear sampler. U/V address modes come from caller (per-texture
        # `inputs:wrapS/wrapT` from USD) so a bindless slot used for a
        # tiled brick map can REPEAT while a clamped foliage cutout uses
        # CLAMP_TO_EDGE. W stays clamped — only 2D images live here.
        sampler_info = vk.VkSamplerCreateInfo(
            magFilter=vk.VK_FILTER_LINEAR,
            minFilter=vk.VK_FILTER_LINEAR,
            mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            addressModeU=address_mode_u,
            addressModeV=address_mode_v,
            addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            anisotropyEnable=vk.VK_FALSE,
            maxAnisotropy=1.0,
            compareEnable=vk.VK_FALSE,
            compareOp=vk.VK_COMPARE_OP_ALWAYS,
            minLod=0.0,
            maxLod=0.0,
            borderColor=vk.VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
            unnormalizedCoordinates=vk.VK_FALSE,
        )
        self.sampler = vk.vkCreateSampler(ctx.device, sampler_info, None)

        # Initial transition: UNDEFINED → SHADER_READ_ONLY_OPTIMAL (empty data).
        self._run_one_shot(self._record_to_shader_read)

    def _run_one_shot(self, record_fn) -> None:
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        record_fn(cmd)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def _record_to_shader_read(self, cmd) -> None:
        barrier = vk.VkImageMemoryBarrier(
            srcAccessMask=0,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            image=self.image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        )
        vk.vkCmdPipelineBarrier(
            cmd,
            vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 0, None, 1, [barrier],
        )

    def upload_sync(self, rgba_f32) -> None:
        """Copy RGBA32F data (H×W×4 numpy array) into the GPU image. Synchronous."""
        data = bytes(rgba_f32)
        if len(data) != self._byte_count:
            raise ValueError(
                f"env upload: got {len(data)} bytes, expected {self._byte_count}"
            )

        ptr = vk.vkMapMemory(self.ctx.device, self.staging_memory, 0, self._staging_size, 0)
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self.ctx.device, self.staging_memory)

        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1,
            baseArrayLayer=0, layerCount=1,
        )

        def record(cmd) -> None:
            to_dst = vk.VkImageMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                image=self.image,
                subresourceRange=subresource,
            )
            vk.vkCmdPipelineBarrier(
                cmd,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, None, 0, None, 1, [to_dst],
            )
            region = vk.VkBufferImageCopy(
                bufferOffset=0,
                bufferRowLength=0,
                bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
            )
            vk.vkCmdCopyBufferToImage(
                cmd,
                self.staging_buffer,
                self.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [region],
            )
            to_read = vk.VkImageMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                image=self.image,
                subresourceRange=subresource,
            )
            vk.vkCmdPipelineBarrier(
                cmd,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None, 1, [to_read],
            )

        # Before uploading for the very first time the image is still in
        # SHADER_READ_ONLY_OPTIMAL (from the ctor transition), so the normal
        # SHADER_READ → TRANSFER_DST flip records cleanly.
        self._run_one_shot(record)

    def destroy(self) -> None:
        vk.vkDestroySampler(self.ctx.device, self.sampler, None)
        vk.vkDestroyImageView(self.ctx.device, self.view, None)
        vk.vkDestroyImage(self.ctx.device, self.image, None)
        vk.vkFreeMemory(self.ctx.device, self.image_memory, None)
        vk.vkDestroyBuffer(self.ctx.device, self.staging_buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.staging_memory, None)


class StorageBuffer:
    """Device-local VK_BUFFER_USAGE_STORAGE buffer with host staging for uploads.

    Mesh data (vertices, indices, BVH nodes) is uploaded once when the user
    switches head models. Since switches are rare we keep a persistent staging
    buffer alongside the device-local one; a one-shot command does the copy.
    """

    def __init__(self, ctx: VulkanContext, size_bytes: int, *, indirect: bool = False,
                 external: bool = False, shared: bool = False) -> None:
        self.ctx = ctx
        self.size = max(int(size_bytes), 16)  # GPU drivers dislike zero-sized buffers
        # `shared` is the Metal UMA in-place-write mode (metal-neural-interop);
        # accepted here for construction-site parity and ignored — Vulkan interop
        # is the exported-memory path below, and `write_in_place` is Metal-only.
        self.shared = False
        # `indirect` adds INDIRECT_BUFFER usage so a build-args kernel can write
        # VkDispatchIndirectCommand triples this buffer holds and the renderer
        # can feed it to vkCmdDispatchIndirect (the wavefront per-material shade).
        self._indirect = bool(indirect)
        # `external` allocates the device-local memory as exportable to CUDA
        # (VK_KHR_external_memory) for the interop weight handoff (task 5.1). A
        # guarded no-op when the device lacks the extension — the buffer is then a
        # plain device-local buffer and `export_handle()` returns None.
        self.external = bool(external) and getattr(ctx, "supports_external_memory", False)

        # Host-visible staging (SRC for uploads, DST for download_sync readback).
        stg_info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.staging_buffer = vk.vkCreateBuffer(ctx.device, stg_info, None)
        stg_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.staging_buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        self._staging_size = stg_reqs.size
        stg_type = UniformBuffer._find_memory_type(
            stg_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        self.staging_memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=stg_reqs.size, memoryTypeIndex=stg_type
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self.staging_buffer, self.staging_memory, 0)

        # Device-local storage buffer
        device_usage = (vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
                        | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
        if self._indirect:
            device_usage |= vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
        ext_buf_info = (
            vk.VkExternalMemoryBufferCreateInfo(handleTypes=ctx._external_memory_handle_type)
            if self.external else None
        )
        buf_info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=device_usage,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            pNext=ext_buf_info,
        )
        self.buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)
        buf_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.buffer)
        buf_type = UniformBuffer._find_memory_type(
            buf_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mem_props,
        )
        # CUDA's import of an OPAQUE_WIN32 buffer (cudaImportExternalMemory) needs
        # a *dedicated* allocation on NVIDIA, so chain VkMemoryDedicatedAllocateInfo
        # onto the export info for the external path. `alloc_size` is the padded
        # allocation size CUDA imports (not the logical `size`).
        self.alloc_size = buf_reqs.size
        alloc_pnext = (
            vk.VkExportMemoryAllocateInfo(
                pNext=vk.VkMemoryDedicatedAllocateInfo(buffer=self.buffer),
                handleTypes=ctx._external_memory_handle_type)
            if self.external else None
        )
        self.memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=buf_reqs.size, memoryTypeIndex=buf_type, pNext=alloc_pnext,
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self.buffer, self.memory, 0)

    def export_handle(self):
        """OS handle to this buffer's device memory for CUDA import
        (``cudaImportExternalMemory``), or None unless allocated ``external=True``
        on a device that supports it (task 5.1). Best-effort: the extension
        function is loaded by name and any failure returns None — the CUDA-side
        import + sync is the seam ``neural_handoff_interop`` fills on the NVIDIA
        box (task 5.2)."""
        if not self.external:
            return None
        try:
            import sys
            if sys.platform == "win32":
                fn = vk.vkGetDeviceProcAddr(self.ctx.device, "vkGetMemoryWin32HandleKHR")
                if fn is None:
                    return None
                info = vk.VkMemoryGetWin32HandleInfoKHR(
                    memory=self.memory,
                    handleType=self.ctx._external_memory_handle_type)
                return fn(self.ctx.device, info)
            fn = vk.vkGetDeviceProcAddr(self.ctx.device, "vkGetMemoryFdKHR")
            if fn is None:
                return None
            info = vk.VkMemoryGetFdInfoKHR(
                memory=self.memory,
                handleType=self.ctx._external_memory_handle_type)
            return fn(self.ctx.device, info)
        except Exception:  # noqa: BLE001 — handle retrieval is the interop seam
            return None

    def upload_sync(self, data: bytes) -> None:
        """Copy ``data`` into the device-local buffer via a one-shot transfer."""
        payload = data if len(data) >= 16 else data + b"\x00" * (16 - len(data))
        if len(payload) > self.size:
            raise ValueError(
                f"StorageBuffer upload: payload {len(payload)}B > buffer {self.size}B"
            )

        ptr = vk.vkMapMemory(self.ctx.device, self.staging_memory, 0, self._staging_size, 0)
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, payload, len(payload))
        vk.vkUnmapMemory(self.ctx.device, self.staging_memory)

        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=len(payload))
        vk.vkCmdCopyBuffer(cmd, self.staging_buffer, self.buffer, 1, [region])
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def upload_range(self, data: bytes, dst_offset: int) -> None:
        """Copy ``data`` into the device-local buffer at byte ``dst_offset`` via a
        one-shot transfer, leaving the rest of the buffer untouched. Used by the
        slab allocator to write a single mesh's slab without re-uploading
        neighbouring slabs."""
        if not data:
            return
        if dst_offset + len(data) > self.size:
            raise ValueError(
                f"StorageBuffer upload_range: {dst_offset}+{len(data)}B "
                f"> buffer {self.size}B"
            )
        ptr = vk.vkMapMemory(self.ctx.device, self.staging_memory, 0, self._staging_size, 0)
        import cffi
        ffi = cffi.FFI()
        ffi.memmove(ptr, data, len(data))
        vk.vkUnmapMemory(self.ctx.device, self.staging_memory)

        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        region = vk.VkBufferCopy(srcOffset=0, dstOffset=int(dst_offset), size=len(data))
        vk.vkCmdCopyBuffer(cmd, self.staging_buffer, self.buffer, 1, [region])
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def download_sync(self, byte_count: int) -> bytes:
        """Copy the first ``byte_count`` bytes of the device-local buffer back to
        host (device → staging → map). For tests / readback of compute output."""
        n = min(int(byte_count), self.size)
        cmd = vk.vkAllocateCommandBuffers(
            self.ctx.device, vk.VkCommandBufferAllocateInfo(
                commandPool=self.ctx.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))
        vk.vkCmdCopyBuffer(cmd, self.buffer, self.staging_buffer, 1,
                           [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=n)])
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(self.ctx.compute_queue, 1,
                         [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
                         vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])
        ptr = vk.vkMapMemory(self.ctx.device, self.staging_memory, 0, self._staging_size, 0)
        import cffi
        ffi = cffi.FFI()
        buf = ffi.new(f"char[{n}]")
        ffi.memmove(buf, ptr, n)
        vk.vkUnmapMemory(self.ctx.device, self.staging_memory)
        return bytes(ffi.buffer(buf, n))

    def fill_zero_sync(self) -> None:
        """Zero the device-local buffer via vkCmdFillBuffer."""
        alloc_info = vk.VkCommandBufferAllocateInfo(
            commandPool=self.ctx.command_pool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        )
        cmd = vk.vkAllocateCommandBuffers(self.ctx.device, alloc_info)[0]
        vk.vkBeginCommandBuffer(
            cmd,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        # vkCmdFillBuffer requires size to be a multiple of 4 and the buffer
        # to be created with TRANSFER_DST_BIT (already set in __init__).
        # Round down to nearest 4-byte boundary defensively.
        fill_size = (self.size // 4) * 4
        vk.vkCmdFillBuffer(cmd, self.buffer, 0, fill_size, 0)
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.ctx.compute_queue, 1,
            [vk.VkSubmitInfo(commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.ctx.compute_queue)
        vk.vkFreeCommandBuffers(self.ctx.device, self.ctx.command_pool, 1, [cmd])

    def destroy(self) -> None:
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)
        vk.vkDestroyBuffer(self.ctx.device, self.staging_buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.staging_memory, None)


class ExternalTimelineSemaphore:
    """A Vulkan timeline semaphore, optionally exportable to CUDA.

    Orders the CUDA weight-write against the Vulkan weight-read for the interop
    weight handoff (change neural-online-training, task 5.2): the trainer's CUDA
    stream signals the timeline at the staged network version; the renderer's
    frame-end swap host-waits that value so the new weights are provably resident
    before the next frame binds the exported buffers. ``export_handle()`` yields
    the OS handle CUDA imports with ``cudaImportExternalSemaphore``; a guarded
    no-op where the device lacks external-semaphore support (then it is a plain
    in-process timeline semaphore and ``export_handle()`` returns None)."""

    def __init__(self, ctx: VulkanContext, initial_value: int = 0) -> None:
        self.ctx = ctx
        self.external = bool(getattr(ctx, "supports_external_semaphore", False))
        type_info = vk.VkSemaphoreTypeCreateInfo(
            semaphoreType=vk.VK_SEMAPHORE_TYPE_TIMELINE,
            initialValue=int(initial_value),
        )
        pnext = type_info
        if self.external:
            pnext = vk.VkExportSemaphoreCreateInfo(
                pNext=type_info,
                handleTypes=ctx._external_semaphore_handle_type,
            )
        self.semaphore = vk.vkCreateSemaphore(
            ctx.device, vk.VkSemaphoreCreateInfo(pNext=pnext), None)

    def export_handle(self):
        """OS handle to this semaphore for CUDA import
        (``cudaImportExternalSemaphore``), or None unless created external on a
        device that supports it. Best-effort — any failure returns None."""
        if not self.external:
            return None
        try:
            import sys
            if sys.platform == "win32":
                fn = vk.vkGetDeviceProcAddr(self.ctx.device, "vkGetSemaphoreWin32HandleKHR")
                if fn is None:
                    return None
                info = vk.VkSemaphoreGetWin32HandleInfoKHR(
                    semaphore=self.semaphore,
                    handleType=self.ctx._external_semaphore_handle_type)
                return fn(self.ctx.device, info)
            fn = vk.vkGetDeviceProcAddr(self.ctx.device, "vkGetSemaphoreFdKHR")
            if fn is None:
                return None
            info = vk.VkSemaphoreGetFdInfoKHR(
                semaphore=self.semaphore,
                handleType=self.ctx._external_semaphore_handle_type)
            return fn(self.ctx.device, info)
        except Exception:  # noqa: BLE001 — handle retrieval is the interop seam
            return None

    def value(self) -> int:
        """Current timeline counter value."""
        return int(vk.vkGetSemaphoreCounterValue(self.ctx.device, self.semaphore))

    def wait(self, value: int, timeout_ns: int = 1_000_000_000) -> bool:
        """Block the host until the counter reaches ``value`` (or timeout). True on
        reach, False on timeout."""
        info = vk.VkSemaphoreWaitInfo(
            semaphoreCount=1, pSemaphores=[self.semaphore], pValues=[int(value)])
        # The `vulkan` binding returns None for VK_SUCCESS and the positive
        # VK_TIMEOUT code on timeout (negative VkResults raise). Reached ⇔ not a
        # timeout.
        res = vk.vkWaitSemaphores(self.ctx.device, info, int(timeout_ns))
        return res is None or res == vk.VK_SUCCESS

    def signal(self, value: int) -> None:
        """Host-signal the timeline to ``value`` (test / fallback path; the trainer
        normally signals from CUDA)."""
        vk.vkSignalSemaphore(
            self.ctx.device,
            vk.VkSemaphoreSignalInfo(semaphore=self.semaphore, value=int(value)))

    def destroy(self) -> None:
        vk.vkDestroySemaphore(self.ctx.device, self.semaphore, None)


class HostStorageBuffer:
    """Host-visible + coherent storage buffer usable directly as an SSBO.

    Skips the device-local + staging round-trip of ``StorageBuffer`` — the
    GPU writes (or reads) the same memory the CPU maps. Performance is
    worse than a device-local buffer for hot-path traffic, but the BXDF
    visualizer only writes a few dozen bytes per pick, so the simplicity
    wins. Used at descriptor binding 30 (toolBuffer).
    """

    def __init__(self, ctx: VulkanContext, size_bytes: int) -> None:
        self.ctx = ctx
        self.size = max(int(size_bytes), 16)

        buf_info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)

        reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.buffer)
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(ctx.physical_device)
        mem_type = UniformBuffer._find_memory_type(
            reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mem_props,
        )
        self._alloc_size = reqs.size
        self.memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(allocationSize=reqs.size, memoryTypeIndex=mem_type),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self.buffer, self.memory, 0)

        # Persistent map. Host-coherent memory can stay mapped for the
        # lifetime of the buffer (no flush required). vkMapMemory returns
        # a _cffi_backend.buffer which supports slice indexing for
        # read / write but not pointer arithmetic; we use slicing.
        self._ptr = vk.vkMapMemory(ctx.device, self.memory, 0, self._alloc_size, 0)
        # Zero on init so first reads don't see garbage.
        self._ptr[0:self.size] = b"\x00" * self.size

    def write(self, data: bytes, offset: int = 0) -> None:
        if offset + len(data) > self.size:
            raise ValueError(
                f"HostStorageBuffer.write: {offset + len(data)}B > buffer {self.size}B"
            )
        self._ptr[offset:offset + len(data)] = bytes(data)

    def read(self, length: int, offset: int = 0) -> bytes:
        if offset + length > self.size:
            raise ValueError(
                f"HostStorageBuffer.read: {offset + length}B > buffer {self.size}B"
            )
        return bytes(self._ptr[offset:offset + length])

    def destroy(self) -> None:
        vk.vkUnmapMemory(self.ctx.device, self.memory)
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)
