"""Wavefront execution backend — staged compute passes (P1 §P1-F).

Vulkan only. Owns the per-stream path-state + queue buffers and (in later
steps) the stage compute pipelines, dispatched as a bounce loop before the
frame's accumulation step. Modelled on ``vk_skinning.SkinningPasses``: a
self-contained owner of its own GPU resources, driven by the renderer.

Buffer sizes come from ``wavefront_layout.queue_buffer_sizes`` — the single
source of truth shared with the GPU-free layout tests. This module is the
Phase-1 integration scaffold: buffer allocation lands first; the
generate/intersect/logic/shade stage pipelines and the bounce loop follow,
wired to the renderer's shared scene bindings.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import vulkan as vk

from skinny.vk_compute import StorageBuffer
from skinny.wavefront_layout import queue_buffer_sizes


def _slang_flags(shader_dir: Path, entry: str) -> tuple[str, ...]:
    return (
        "-target", "spirv", "-entry", entry, "-stage", "compute",
        "-I", str(shader_dir), "-fvk-use-scalar-layout",
    )


def _compile_spv(shader_dir: Path, module: str, entry: str) -> Path:
    """Compile shaders/<module>.slang → .spv (scalar layout). Mirrors
    vk_skinning._compile so the wavefront stage kernels build the same way."""
    src = shader_dir / f"{module}.slang"
    out = shader_dir / f"{module}.spv"
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    cmd = [slangc, str(src), *_slang_flags(shader_dir, entry), "-o", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slang compilation failed ({module}):\n{result.stderr}")
    return out


_SHADE_CACHE_DIRNAME = "spv_cache"


def _build_dir() -> Path:
    """Where the SPIR-V cache lives (shared with vk_compute.ComputePipeline)."""
    return Path(__file__).resolve().parents[2] / "build"


def shared_shader_hash(shader_dir: Path) -> bytes:
    """Content hash of the *stable* shader tree — every `.slang` under
    shader_dir EXCEPT the per-scene `generated/` modules and the per-graph
    `wavefront/shade_*` modules. Folded into each per-material cache key so a
    renderer-code change invalidates the cache, while adding/removing a graph
    (which only touches the excluded files) does not."""
    h = hashlib.blake2b(digest_size=16)
    for path in sorted(shader_dir.rglob("*.slang")):
        rel = path.relative_to(shader_dir)
        if rel.parts and rel.parts[0] == "generated":
            continue
        if (len(rel.parts) >= 2 and rel.parts[0] == "wavefront"
                and rel.name.startswith("shade_")):
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.digest()


def compile_shade_module_cached(
    shader_dir: Path, module: str, entry: str,
    dep_sources: list[bytes], shared_hash: bytes,
) -> tuple[Path, bool, str]:
    """Compile a per-material shade module with a content-hash SPIR-V cache —
    the per-material-pipeline compile-win. Returns (spv_path, was_cached, key).

    The key hashes the entry + flags + this module's own source + its generated
    graph dependency(ies) + the shared shader tree. Because it excludes every
    *other* graph's source, adding a new material misses only that material's
    key: resident materials' SPIR-V is copied from cache and slangc is not run
    for them. Mirrors `ComputePipeline._compile_slang`'s on-disk LRU cache but
    at per-module (not whole-kernel) granularity."""
    src = shader_dir / f"{module}.slang"
    out = shader_dir / f"{module}.spv"
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    flags = _slang_flags(shader_dir, entry)

    h = hashlib.blake2b(digest_size=16)
    h.update(entry.encode("utf-8"))
    h.update(b"\0")
    for f in flags:
        h.update(f.encode("utf-8"))
        h.update(b"\0")
    h.update(src.read_bytes())
    h.update(b"\0")
    for dep in dep_sources:
        h.update(dep)
        h.update(b"\0")
    h.update(shared_hash)
    key = h.hexdigest()

    cache_dir = _build_dir() / _SHADE_CACHE_DIRNAME
    cached = cache_dir / f"{key}.spv"
    if cached.exists():
        shutil.copyfile(cached, out)
        try:
            cached.touch()  # LRU bump
        except OSError:
            pass
        return out, True, key

    cmd = [slangc, str(src), *flags, "-o", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slang compilation failed ({module}):\n{result.stderr}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, cache_dir / f"{key}.spv")
    return out, False, key


class BoundComputePass:
    """General single-dispatch compute pass over the framebuffer.

    Compiles `shaders/<module>.slang::<entry>`, builds a descriptor set layout +
    set from an explicit binding spec, and dispatches one thread per pixel. The
    spec lets a wavefront stage kernel bind any mix of the renderer's shared
    resources (UBO / storage image / combined sampler / storage buffer) at the
    binding numbers slangc reflects for it.

    Each spec entry is a dict with `binding`, `type` (a vk descriptor type), and
    the matching resource fields:
      - UNIFORM_BUFFER / STORAGE_BUFFER: `buffer` (VkBuffer), `range` (int)
      - STORAGE_IMAGE: `view` (VkImageView), `layout` (VkImageLayout)
      - COMBINED_IMAGE_SAMPLER: `sampler`, `view`, `layout`
    """

    def __init__(self, ctx, shader_dir: Path, module: str, entry: str,
                 specs: list, width: int, height: int, group: int = 8,
                 spv_path: "Path | None" = None):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)
        self._group = group

        # Reuse a caller-supplied (cached) SPIR-V when given — the shade-module
        # compile-win path compiles via compile_shade_module_cached and passes
        # the result here, so resident materials skip slangc entirely.
        spv = spv_path if spv_path is not None else _compile_spv(shader_dir, module, entry)
        code = spv.read_bytes()
        self._module = vk.vkCreateShaderModule(
            ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None
        )

        # An "array_count" spec is a bindless descriptor array (e.g. the
        # 128-slot texture pool at binding 14). It needs UPDATE_AFTER_BIND +
        # PARTIALLY_BOUND descriptor indexing, mirroring the megakernel layout.
        has_array = any("array_count" in s for s in specs)
        layout_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=s["binding"], descriptorType=s["type"],
                descriptorCount=s.get("array_count", 1),
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for s in specs
        ]
        layout_kwargs = dict(bindingCount=len(layout_bindings), pBindings=layout_bindings)
        if has_array:
            binding_flags = [
                (vk.VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                 | vk.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
                if "array_count" in s else 0
                for s in specs
            ]
            layout_kwargs["pNext"] = vk.VkDescriptorSetLayoutBindingFlagsCreateInfo(
                bindingCount=len(binding_flags), pBindingFlags=binding_flags)
            layout_kwargs["flags"] = 0x00000002  # UPDATE_AFTER_BIND_POOL
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(**layout_kwargs), None)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device,
            vk.VkPipelineLayoutCreateInfo(setLayoutCount=1, pSetLayouts=[self._set_layout]),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self._module, pName="main")
        self._pipeline = vk.vkCreateComputePipelines(
            ctx.device, vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None,
        )[0]

        # Pool sized by counting each descriptor type (array specs count their
        # full capacity).
        type_counts: dict = {}
        for s in specs:
            type_counts[s["type"]] = type_counts.get(s["type"], 0) + s.get("array_count", 1)
        pool_sizes = [vk.VkDescriptorPoolSize(type=t, descriptorCount=c)
                      for t, c in type_counts.items()]
        pool_kwargs = dict(maxSets=1, poolSizeCount=len(pool_sizes), pPoolSizes=pool_sizes)
        if has_array:
            pool_kwargs["flags"] = 0x00000002  # UPDATE_AFTER_BIND_POOL
        self._pool = vk.vkCreateDescriptorPool(
            ctx.device, vk.VkDescriptorPoolCreateInfo(**pool_kwargs), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1, pSetLayouts=[self._set_layout]),
        )[0]

        writes = []
        for s in specs:
            t = s["type"]
            if "array_count" in s:
                # Bindless array — write only the filled slots (PARTIALLY_BOUND
                # leaves the rest invalid; the shader gates reads by sentinel).
                for idx, sampler, view in s["slots"]:
                    info = vk.VkDescriptorImageInfo(
                        sampler=sampler, imageView=view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                    writes.append(vk.VkWriteDescriptorSet(
                        dstSet=self._set, dstBinding=s["binding"], dstArrayElement=idx,
                        descriptorCount=1, descriptorType=t, pImageInfo=[info]))
                continue
            if t in (vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER):
                info = vk.VkDescriptorBufferInfo(buffer=s["buffer"], offset=0, range=s["range"])
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=s["binding"], dstArrayElement=0,
                    descriptorCount=1, descriptorType=t, pBufferInfo=[info]))
            elif t == vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                info = vk.VkDescriptorImageInfo(imageView=s["view"], imageLayout=s["layout"])
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=s["binding"], dstArrayElement=0,
                    descriptorCount=1, descriptorType=t, pImageInfo=[info]))
            elif t == vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                info = vk.VkDescriptorImageInfo(
                    sampler=s["sampler"], imageView=s["view"], imageLayout=s["layout"])
                writes.append(vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=s["binding"], dstArrayElement=0,
                    descriptorCount=1, descriptorType=t, pImageInfo=[info]))
            else:
                raise ValueError(f"unsupported descriptor type {t}")
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def record_dispatch(self, cmd) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 1, [self._set], 0, None)
        gx = (self.width + self._group - 1) // self._group
        gy = (self.height + self._group - 1) // self._group
        vk.vkCmdDispatch(cmd, gx, gy, 1)

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        vk.vkDestroyPipeline(self.ctx.device, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        vk.vkDestroyShaderModule(self.ctx.device, self._module, None)


class ShadePassGroup:
    """Ordered set of per-material wavefront shade passes, dispatched in one
    frame — the staged per-material-pipeline shade (P1 §5.4 / §6.4 compile-win).

    Each member is an independent ``BoundComputePass`` compiled from one graph's
    ``shadeSurface_<name>`` entry (``emit_wavefront_shade_module``), so it is its
    own compilation unit / pipeline: adding a material compiles exactly one
    member and the rest are SPIR-V cache hits. Each member traces and overwrites
    only the accumulation pixels whose material maps to its ``graphId``; together
    they cover every materialised hit. A memory barrier between members orders
    the writes to the shared accumulation image (binding 2).

    Exposes ``record_dispatch`` + ``destroy`` so it drops straight into the
    renderer's ``_wavefront_debug_pass`` seam in place of a single pass.
    """

    def __init__(self, ctx, passes: list) -> None:
        self.ctx = ctx
        self.passes = list(passes)

    def record_dispatch(self, cmd) -> None:
        accum_barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        for i, p in enumerate(self.passes):
            if i > 0:
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 1, [accum_barrier], 0, None, 0, None,
                )
            p.record_dispatch(cmd)

    def destroy(self) -> None:
        for p in self.passes:
            p.destroy()
        self.passes = []


class WavefrontEnvPass:
    """Env-only wavefront compute pass — the Phase-1 integration milestone.

    Dispatches `wavefront_env.slang::wavefrontEnv`, which generates the camera
    ray and writes the environment radiance into the renderer's accumulation
    image (binding 2) using the same fc UBO (binding 0) and env map (binding 4)
    as the megakernel. Owns its own descriptor set layout + set bound to the
    renderer's existing resources; it does NOT touch the megakernel pipeline.

    A/B is in linear HDR: render this vs the megakernel over a geometry-free /
    background region and compare the accumulation image. The staged
    generate→intersect→logic→shade pipeline supersedes this pass.
    """

    _ENV_GROUP = 8  # matches [numthreads(8, 8, 1)] in wavefront_env.slang

    def __init__(self, ctx, shader_dir: Path, *, uniform_buffer, uniform_size,
                 accum_view, env_view, env_sampler, width, height):
        self.ctx = ctx
        self.width = int(width)
        self.height = int(height)

        spv = _compile_spv(shader_dir, "wavefront/wavefront_env", "wavefrontEnv")
        code = spv.read_bytes()
        self._module = vk.vkCreateShaderModule(
            ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None
        )

        # Descriptor set layout: binding 0 = fc UBO, 2 = accum storage image,
        # 4 = env combined-image-sampler (gaps at 1/3 are fine). Matches the
        # binding numbers slangc reflects for wavefront_env.
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=4, descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device,
            vk.VkDescriptorSetLayoutCreateInfo(bindingCount=len(bindings), pBindings=bindings),
            None,
        )
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device,
            vk.VkPipelineLayoutCreateInfo(setLayoutCount=1, pSetLayouts=[self._set_layout]),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self._module, pName="main",
        )
        self._pipeline = vk.vkCreateComputePipelines(
            ctx.device, vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None,
        )[0]

        self._pool = vk.vkCreateDescriptorPool(
            ctx.device,
            vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=3,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1),
                    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, descriptorCount=1),
                    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptorCount=1),
                ],
            ),
            None,
        )
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1, pSetLayouts=[self._set_layout],
            ),
        )[0]

        ubo_info = vk.VkDescriptorBufferInfo(buffer=uniform_buffer, offset=0, range=uniform_size)
        accum_info = vk.VkDescriptorImageInfo(
            imageView=accum_view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
        )
        env_info = vk.VkDescriptorImageInfo(
            sampler=env_sampler, imageView=env_view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        )
        vk.vkUpdateDescriptorSets(
            ctx.device, 3,
            [
                vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=0, dstArrayElement=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, pBufferInfo=[ubo_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=2, dstArrayElement=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, pImageInfo=[accum_info],
                ),
                vk.VkWriteDescriptorSet(
                    dstSet=self._set, dstBinding=4, dstArrayElement=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, pImageInfo=[env_info],
                ),
            ],
            0, None,
        )

    def record_dispatch(self, cmd) -> None:
        """Bind the env pipeline + descriptor set and dispatch one thread/pixel.
        Recorded in place of the megakernel dispatch when wavefront is active;
        writes the accumulation image (binding 2)."""
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 1, [self._set], 0, None,
        )
        gx = (self.width + self._ENV_GROUP - 1) // self._ENV_GROUP
        gy = (self.height + self._ENV_GROUP - 1) // self._ENV_GROUP
        vk.vkCmdDispatch(cmd, gx, gy, 1)

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        vk.vkDestroyPipeline(self.ctx.device, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        vk.vkDestroyShaderModule(self.ctx.device, self._module, None)


def _compile_full_spv(shader_dir: Path, module: str, entry: str, out_name: str) -> Path:
    """Compile a wavefront kernel that pulls in the full material/integrator
    tree (integrators.path → skin/python/flat materials), so it needs the same
    include paths + define as the megakernel. Writes to a per-entry .spv so
    sibling entries from one module don't clobber each other."""
    src = shader_dir / f"{module}.slang"
    out = shader_dir / f"{out_name}.spv"
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    mtlx_genslang = shader_dir.parent / "mtlx" / "genslang"
    cmd = [
        slangc, str(src), "-target", "spirv", "-entry", entry, "-stage", "compute",
        "-I", str(shader_dir), "-I", str(mtlx_genslang),
        "-D", "SKINNY_COMPUTE_PIPELINE=1", "-fvk-use-scalar-layout", "-o", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slang compilation failed ({module}:{entry}):\n{result.stderr}")
    return out


class WavefrontPathPass:
    """Staged wavefront path tracer — the real per-frame wavefront dispatch.

    Owns three compute pipelines compiled from ``wavefront/wavefront_path.slang``
    (generate / bounce / resolve) and a per-frame bounce loop that round-trips a
    per-lane path-state record through VRAM. Set 0 is the renderer's existing
    megakernel scene descriptor set (passed at dispatch); set 1 binding 0 is the
    path-state buffer this owns. The estimator is reused verbatim from
    integrators/path.slang, so the accumulated image matches the megakernel.

    Modelled on ``SkinningPasses``: a self-contained owner of GPU resources
    driven by the renderer; ``record_dispatch`` records the whole loop into the
    frame command buffer in place of the megakernel dispatch.
    """

    _GROUP = 64  # matches [numthreads(64, 1, 1)] in wavefront_path.slang
    MAX_BOUNCES = 6  # lockstep with WF_MAX_BOUNCES in the shader
    STREAM_CAP = 1 << 20  # max lanes per stream — bounds path-state VRAM (~68 MB)

    HIT_STRIDE = 96  # ≥ sizeof(HitInfo) (≈92 B scalar) — headroom

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 state_buffer, state_range: int, hit_buffer, hit_range: int,
                 stream_size: int, num_pixels: int) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)   # slots in the path-state buffer
        self.num_pixels = int(num_pixels)     # total pixels to cover, tiled

        modules = {}
        for entry, out_name in (
            ("wfPathGenerate", "wavefront/_wfpath_generate"),
            ("wfPathIntersect", "wavefront/_wfpath_intersect"),
            ("wfPathShade", "wavefront/_wfpath_shade"),
            ("wfPathResolve", "wavefront/_wfpath_resolve"),
        ):
            spv = _compile_full_spv(shader_dir, "wavefront/wavefront_path", entry, out_name)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        # Set 1: path-state (0) + hit (1) storage buffers.
        state_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in (0, 1)
        ]
        self._state_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=2, pBindings=state_bindings), None)

        # Pipeline layout: [set 0 = megakernel scene set, set 1 = path state] +
        # a 4-byte push constant (the per-tile stream base, read by generate).
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=4)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=2, pSetLayouts=[scene_set_layout, self._state_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)

        self._pipelines = {}
        for entry, mod in modules.items():
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
            self._pipelines[entry] = vk.vkCreateComputePipelines(
                ctx.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None)[0]

        self._pool = vk.vkCreateDescriptorPool(
            ctx.device, vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)]), None)
        self._state_set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._state_layout]))[0]
        writes = []
        for b, (buf, rng) in enumerate(((state_buffer, state_range), (hit_buffer, hit_range))):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._state_set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _bind(self, cmd, entry, scene_set) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._state_set], 0, None)

    def _dispatch(self, cmd) -> None:
        groups = (self.stream_size + self._GROUP - 1) // self._GROUP
        vk.vkCmdDispatch(cmd, groups, 1, 1)

    def _push_base(self, cmd, stream_base) -> None:
        import struct

        import cffi
        buf = cffi.FFI().new("char[]", struct.pack("I", int(stream_base)))
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, buf)

    def record_dispatch(self, cmd, scene_set) -> None:
        """Record the tiled bounce loop. The frame's pixels are processed in
        fixed-size streams (`stream_size` slots); for each tile, push its
        `streamBase` then generate → bounce ×MAX → resolve. `scene_set` is the
        renderer's per-frame megakernel descriptor set (set 0). Path-state VRAM
        is bounded by `stream_size`, not the pixel count."""
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)

        def mem_barrier():
            vk.vkCmdPipelineBarrier(
                cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [barrier], 0, None, 0, None)

        stream_base = 0
        first = True
        while stream_base < self.num_pixels:
            if not first:
                mem_barrier()  # previous tile's resolve before this tile reuses wfState
            first = False
            self._push_base(cmd, stream_base)
            self._bind(cmd, "wfPathGenerate", scene_set)
            self._dispatch(cmd)
            for _ in range(self.MAX_BOUNCES):
                # Each bounce: intersect (trace → hit / miss-terminate), then
                # shade (NEE · sample · next ray) reading the stashed hit.
                mem_barrier()
                self._bind(cmd, "wfPathIntersect", scene_set)
                self._dispatch(cmd)
                mem_barrier()
                self._bind(cmd, "wfPathShade", scene_set)
                self._dispatch(cmd)
            mem_barrier()
            self._bind(cmd, "wfPathResolve", scene_set)
            self._dispatch(cmd)
            stream_base += self.stream_size

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._state_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)


class IndirectPaintPass:
    """Per-material-queue dispatch over `wavefront/indirect_paint.slang`, the
    carrier for verifying the wavefront indirect-dispatch path (tasks 3.2 / 3.3
    / 9.3). Binds a counting-sort `materialQueue` + a per-lane `paintOut`; for
    each material slot it pushes (sliceBase, sliceCount, value) and dispatches
    the slot's lanes either indirectly (`vkCmdDispatchIndirect` over a
    build_args-shaped buffer) or via a conservative direct dispatch (worst-case
    groups + the kernel's empty-lane early-out). The two must produce identical
    output."""

    _GROUP = 64  # matches [numthreads(64, 1, 1)] in indirect_paint.slang

    def __init__(self, ctx, shader_dir: Path, queue_buf, queue_range: int,
                 out_buf, out_range: int) -> None:
        self.ctx = ctx
        spv = _compile_spv(shader_dir, "wavefront/indirect_paint", "wfIndirectPaint")
        code = spv.read_bytes()
        self._module = vk.vkCreateShaderModule(
            ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)

        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in (0, 1)
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=2, pBindings=bindings), None)
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=12)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[self._set_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)
        stage = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self._module, pName="main")
        self._pipeline = vk.vkCreateComputePipelines(
            ctx.device, vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None)[0]

        self._pool = vk.vkCreateDescriptorPool(
            ctx.device, vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)]), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._set_layout]))[0]
        writes = []
        for b, (buf, rng) in enumerate(((queue_buf, queue_range), (out_buf, out_range))):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _push(self, cmd, base, count, value):
        import struct

        import cffi
        data = struct.pack("3I", int(base), int(count), int(value))
        # python-vulkan binds `const void* pValues` via cffi — pass a sized char[].
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, buf)

    def record_indirect(self, cmd, slots, indirect_buf) -> None:
        """One indirect dispatch per material slot; group count read from
        `indirect_buf` at slot*12 (the VkDispatchIndirectCommand build_args
        writes). `slots` is a list of (sliceBase, sliceCount, value)."""
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 1, [self._set], 0, None)
        for slot, (base, count, value) in enumerate(slots):
            self._push(cmd, base, count, value)
            vk.vkCmdDispatchIndirect(cmd, indirect_buf, slot * 12)

    def record_direct(self, cmd, slots, worst_groups) -> None:
        """Conservative fallback: dispatch worst-case `worst_groups` for every
        slot; the kernel's `tid.x >= sliceCount` early-out skips empty lanes."""
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 1, [self._set], 0, None)
        for base, count, value in slots:
            self._push(cmd, base, count, value)
            vk.vkCmdDispatch(cmd, int(worst_groups), 1, 1)

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        vk.vkDestroyPipeline(self.ctx.device, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        vk.vkDestroyShaderModule(self.ctx.device, self._module, None)


class WavefrontBdptPass:
    """Staged wavefront bidirectional path tracer (Phase 3). Three pipelines
    from ``wavefront/wavefront_bdpt.slang`` (walk / connect / resolve) over
    per-lane eye + light subpath-vertex buffers + an aux buffer. Set 0 is the
    megakernel scene descriptor set (incl. lightSplatBuffer); set 1 holds the
    subpath/aux buffers. The estimator is reused verbatim from
    integrators/bdpt.slang, so the accumulated image matches the megakernel
    bdpt. Matches its scope: flat first-hit, pinhole camera."""

    _GROUP = 64           # matches [numthreads(64, 1, 1)] in wavefront_bdpt.slang
    BDPT_MAX_VERTS = 7    # lockstep with bdpt.slang BDPT_MAX_VERTS
    VERTEX_STRIDE = 128   # ≥ sizeof(BDPTVertex) (≈120 B scalar) — headroom
    AUX_STRIDE = 64       # ≥ sizeof(WfBdptAux) (≈44 B scalar)
    # Smaller cap than the path tracer: each lane owns 2×BDPT_MAX_VERTS vertices
    # (eye+light), so vertex VRAM = stream × 7 × 128 × 2. 1<<18 ≈ 470 MB.
    STREAM_CAP = 1 << 18

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 eye_buf, light_buf, aux_buf, vert_range: int, aux_range: int,
                 stream_size: int, num_pixels: int) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.num_pixels = int(num_pixels)

        modules = {}
        for entry, out_name in (
            ("wfBdptWalk", "wavefront/_wfbdpt_walk"),
            ("wfBdptConnect", "wavefront/_wfbdpt_connect"),
            ("wfBdptResolve", "wavefront/_wfbdpt_resolve"),
        ):
            spv = _compile_full_spv(shader_dir, "wavefront/wavefront_bdpt", entry, out_name)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        # Set 1: eye (0), light (1), aux (2) storage buffers.
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in (0, 1, 2)
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=3, pBindings=bindings), None)
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=4)  # streamBase
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=2, pSetLayouts=[scene_set_layout, self._set_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)

        self._pipelines = {}
        for entry, mod in modules.items():
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
            self._pipelines[entry] = vk.vkCreateComputePipelines(
                ctx.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None)[0]

        self._pool = vk.vkCreateDescriptorPool(
            ctx.device, vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=3)]), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._set_layout]))[0]
        writes = []
        for b, (buf, rng) in enumerate(
                ((eye_buf, vert_range), (light_buf, vert_range), (aux_buf, aux_range))):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _stage(self, cmd, entry, scene_set) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._set], 0, None)
        groups = (self.stream_size + self._GROUP - 1) // self._GROUP
        vk.vkCmdDispatch(cmd, groups, 1, 1)

    def _push_base(self, cmd, stream_base) -> None:
        import struct

        import cffi
        buf = cffi.FFI().new("char[]", struct.pack("I", int(stream_base)))
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, buf)

    def record_dispatch(self, cmd, scene_set) -> None:
        """Tiled walk → connect → resolve. The frame's pixels are processed in
        fixed-size streams; the eye/light/aux subpath buffers are bounded by
        `stream_size`, not the pixel count."""
        barrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)

        def mem_barrier():
            vk.vkCmdPipelineBarrier(
                cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [barrier], 0, None, 0, None)

        stream_base = 0
        first = True
        while stream_base < self.num_pixels:
            if not first:
                mem_barrier()  # prior tile's resolve before this tile reuses the buffers
            first = False
            self._push_base(cmd, stream_base)
            self._stage(cmd, "wfBdptWalk", scene_set)
            mem_barrier()
            self._stage(cmd, "wfBdptConnect", scene_set)
            mem_barrier()
            self._stage(cmd, "wfBdptResolve", scene_set)
            stream_base += self.stream_size

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)


class WavefrontPasses:
    """Owns the wavefront stage buffers for a given stream size + material count.

    The path-state and queue buffers are sized to the active path stream; the
    per-material counting-sort buffers to the material count. They are
    reallocated when either changes (a scene reload that adds materials, or a
    stream-size retune).
    """

    def __init__(self, ctx, stream_size: int, num_materials: int) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.num_materials = int(num_materials)
        self.buffer_sizes = queue_buffer_sizes(self.stream_size, self.num_materials)
        self.buffers: dict[str, StorageBuffer] = {
            name: StorageBuffer(ctx, size) for name, size in self.buffer_sizes.items()
        }

    def destroy(self) -> None:
        """Release all stage buffers. Idempotent."""
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
