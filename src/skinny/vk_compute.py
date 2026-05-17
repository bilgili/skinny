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

# First descriptor binding for MaterialX nodegraph parameter SSBOs. Each
# loaded graph gets its own StructuredBuffer<GraphParams_X> at
# GRAPH_BINDING_BASE + graphIdx; idx 0 == graphId 2 in the dispatch (0=skin,
# 1=flat are reserved). Keep clear of bindings 0..24 used by the renderer.
GRAPH_BINDING_BASE = 25


class ComputePipeline:
    """Wraps a single Vulkan compute pipeline compiled from a Slang entry point."""

    def __init__(
        self,
        ctx: VulkanContext,
        shader_dir: Path,
        entry_module: str,
        entry_point: str,
        graph_fragments: "list | None" = None,
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
        t0 = _time.perf_counter()
        print(f"[skinny] slangc → SPIR-V: {entry_module}.slang …", flush=True)
        self._emit_generated_materials()
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

    # ── Slang → SPIR-V compilation ───────────────────────────────

    def _run_codegen(self) -> None:
        """Regenerate .slang from python_materials/ via embedded SlangPile."""
        import logging
        import sys

        from skinny.slangpile import build_module

        log = logging.getLogger("skinny.codegen")
        materials_dir = Path(__file__).resolve().parent.parent.parent / "python_materials"
        if not materials_dir.is_dir():
            return
        py_files = [f for f in sorted(materials_dir.glob("*.py")) if f.name != "__init__.py"]
        if not py_files:
            return
        out_dir = self.shader_dir.parent / "mtlx" / "genslang"
        sys.path.insert(0, str(materials_dir.parent))
        try:
            for f in py_files:
                mod_name = f"python_materials.{f.stem}"
                log.debug("codegen: %s", mod_name)
                build_module(mod_name, out_dir)
        except Exception as exc:
            log.debug("codegen failed (non-fatal): %s", exc)
        finally:
            sys.path.pop(0)

    def _emit_generated_materials(self) -> None:
        """Materialise shaders/generated_materials.slang + per-graph files.

        `main_pass.slang` `import generated_materials;` always, even when
        the scene carries no MaterialX graphs. Empty list ⇒ aggregator
        emits only the macro-alias prelude and a no-op `evalSceneGraph`
        switch that returns magenta for any graphId (caller never invokes
        it when no graphs are bound).

        Per-graph files are written under `shaders/generated/` so slangc's
        existing `-I shaders/` include path resolves them.
        """
        gen_dir = self.shader_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale per-graph files: a scene reload may drop materials,
        # and stale Slang in the include dir can mask missing wiring or
        # collide on struct names.
        for old in gen_dir.glob("*_graph.slang"):
            old.unlink()

        imports: list[str] = []
        ssbo_decls: list[str] = []
        param_helpers: list[str] = []
        apply_helpers: list[str] = []
        cases: list[str] = []
        for idx, gf in enumerate(self.graph_fragments):
            module_name = f"{gf.sanitized_name}_graph"
            fname = f"{module_name}.slang"
            (gen_dir / fname).write_text(gf.slang_source, encoding="utf-8")
            imports.append(f"import generated.{module_name};")
            binding = GRAPH_BINDING_BASE + idx
            ssbo_decls.append(
                f"[[vk::binding({binding}, 0)]]\n"
                f"StructuredBuffer<{gf.struct_name}> graphParams_{gf.sanitized_name};\n"
            )
            param_helpers.append(
                f"{gf.struct_name} _graphParams_{gf.sanitized_name}(uint matId)\n"
                f"{{\n"
                f"    return graphParams_{gf.sanitized_name}[matId];\n"
                f"}}\n"
            )

            # Per-graph apply: copy each output field onto the matching
            # StdSurfaceParams slot. Built from the fragment's `outputs`
            # metadata so multi-output graphs (brass = specular_roughness +
            # coat_color + coat_roughness) drive several inputs at once.
            assignments = "\n".join(
                f"    sp.{input_name} = g.{input_name};"
                for input_name, _ in gf.outputs
            )
            apply_helpers.append(
                f"void applyGraphOutputs_{gf.sanitized_name}("
                f"inout StdSurfaceParams sp, in {gf.outputs_struct} g)\n"
                f"{{\n"
                f"{assignments}\n"
                f"}}\n"
            )

            cases.append(
                f"        case {idx + 2}u:  // graphId 0=skin, 1=flat reserved\n"
                f"        {{\n"
                f"            {gf.outputs_struct} g = {gf.func_name}(P, N, T, UV, "
                f"_graphParams_{gf.sanitized_name}(matId));\n"
                f"            applyGraphOutputs_{gf.sanitized_name}(sp, g);\n"
                f"            return;\n"
                f"        }}\n"
            )

        # Per-hit base_color override path (FlatMaterial.albedo). Only
        # graphs whose outputs include `base_color` participate; for
        # graphs that drive other inputs (brass: specular_roughness +
        # coat_*) the caller must NOT override albedo, so the case
        # simply returns false and the caller keeps the SSBO constant.
        base_color_cases = ""
        for idx, gf in enumerate(self.graph_fragments):
            has_base = any(i == "base_color" for i, _ in gf.outputs)
            if not has_base:
                continue
            base_color_cases += (
                f"        case {idx + 2}u:\n"
                f"        {{\n"
                f"            {gf.outputs_struct} g = {gf.func_name}(P, N, T, UV, "
                f"_graphParams_{gf.sanitized_name}(matId));\n"
                f"            outColor = g.base_color;\n"
                f"            return true;\n"
                f"        }}\n"
            )

        switch_body = "".join(cases) if cases else ""

        aggregator = (
            "// Auto-generated. Do not edit — written by "
            "ComputePipeline._emit_generated_materials().\n"
            "// Imports each scene MaterialX nodegraph as a Slang module.\n"
            "// Per-graph modules expose only `evalGraph_<target>` + the\n"
            "// matching `GraphParams_<target>` / `GraphOutputs_<target>`\n"
            "// structs; their `internal` helpers stay module-private, so\n"
            "// duplicate symbol names across graphs do not collide.\n\n"
            "import mtlx_std_surface;  // StdSurfaceParams\n\n"
            + "\n".join(imports)
            + ("\n\n" if imports else "\n")
            + "\n".join(ssbo_decls)
            + ("\n" if ssbo_decls else "")
            + "\n".join(param_helpers)
            + ("\n" if param_helpers else "")
            + "\n".join(apply_helpers)
            + ("\n" if apply_helpers else "")
            + "// Evaluate the per-hit nodegraph and overlay each driven\n"
            "// std_surface input on `sp`. graphId 0 / 1 reserved (skin /\n"
            "// flat) — callers gate the call by `materialGraphId(mid) >= 2`.\n"
            "void evalSceneGraph(uint graphId, uint matId,\n"
            "                    float3 P, float3 N, float3 T, float2 UV,\n"
            "                    inout StdSurfaceParams sp)\n"
            "{\n"
            "    switch (graphId)\n"
            "    {\n"
            f"{switch_body}"
            "        default:\n"
            "            sp.base_color = float3(1.0, 0.0, 1.0);\n"
            "            return;\n"
            "    }\n"
            "}\n\n"
            "// Returns true and fills `outColor` only when the active graph\n"
            "// drives std_surface.base_color (marble, wood). Graphs that\n"
            "// drive only other inputs (brass: specular_roughness + coat_*)\n"
            "// return false; the caller keeps the SSBO-uploaded constant\n"
            "// for FlatMaterial.albedo.\n"
            "bool evalSceneGraphBaseColor(uint graphId, uint matId,\n"
            "                              float3 P, float3 N, float3 T, float2 UV,\n"
            "                              out float3 outColor)\n"
            "{\n"
            "    outColor = float3(0.0);\n"
            "    switch (graphId)\n"
            "    {\n"
            f"{base_color_cases}"
            "        default:\n"
            "            return false;\n"
            "    }\n"
            "}\n"
        )
        (self.shader_dir / "generated_materials.slang").write_text(
            aggregator, encoding="utf-8"
        )

        # Expose binding map so the renderer can vkUpdateDescriptorSets into
        # the right slot when uploading per-material graph params.
        self.graph_bindings: dict[str, int] = {
            gf.target_name: GRAPH_BINDING_BASE + idx
            for idx, gf in enumerate(self.graph_fragments)
        }

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
        self._run_codegen()
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
        vk.vkDestroyPipeline(self.ctx.device, self.pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self.descriptor_set_layout, None)
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
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.format = format
        self.bytes_per_pixel = bytes_per_pixel
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

        # Linear sampler; wrap around phi (U) and clamp poles (V).
        sampler_info = vk.VkSamplerCreateInfo(
            magFilter=vk.VK_FILTER_LINEAR,
            minFilter=vk.VK_FILTER_LINEAR,
            mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
            addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
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

    def __init__(self, ctx: VulkanContext, size_bytes: int) -> None:
        self.ctx = ctx
        self.size = max(int(size_bytes), 16)  # GPU drivers dislike zero-sized buffers

        # Host-visible staging
        stg_info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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
        buf_info = vk.VkBufferCreateInfo(
            size=self.size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        )
        self.buffer = vk.vkCreateBuffer(ctx.device, buf_info, None)
        buf_reqs = vk.vkGetBufferMemoryRequirements(ctx.device, self.buffer)
        buf_type = UniformBuffer._find_memory_type(
            buf_reqs.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            mem_props,
        )
        self.memory = vk.vkAllocateMemory(
            ctx.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=buf_reqs.size, memoryTypeIndex=buf_type
            ),
            None,
        )
        vk.vkBindBufferMemory(ctx.device, self.buffer, self.memory, 0)

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
