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
    # SKINNY_WAVEFRONT marks every wavefront-pass compile so shared shader code
    # (e.g. the subsurface dispatch in path.slang) can select the wavefront-only
    # estimator — the 3D interior subsurface walk — while the megakernel
    # (vk_compute.py, no such define) keeps the watchdog-safe 1D slab.
    return (
        "-target", "spirv", "-entry", entry, "-stage", "compute",
        "-I", str(shader_dir), "-fvk-use-scalar-layout",
        "-D", "SKINNY_WAVEFRONT=1",
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


def _compile_full_spv(shader_dir: Path, module: str, entry: str, out_name: str,
                      defines: tuple[str, ...] = (), tag: str = "",
                      spectral: bool = False) -> Path:
    """Compile a wavefront kernel that pulls in the full material/integrator
    tree (integrators.path → skin/python/flat materials), so it needs the same
    include paths + define as the megakernel. Writes to a per-entry .spv so
    sibling entries from one module don't clobber each other.

    ``defines`` are extra ``-D`` tokens (the neural size/precision config —
    study change neural-precision-size-study); ``tag`` is folded into the .spv
    filename so distinct configs never clobber each other's module (the pipeline
    cache key). The default config passes ``defines=()`` and ``tag=""`` → the
    flags + filename are unchanged → byte-identical to the shipped kernel.

    ``spectral`` adds the ``-DSKINNY_SPECTRAL=1`` compile define (matching the
    megakernel's ComputePipeline spectral variant) and a distinct ``_spectral``
    .spv name so the spectral wavefront kernels never alias the RGB cache
    (spectral-wavefront 5.2)."""
    src = shader_dir / f"{module}.slang"
    spectral_suffix = "_spectral" if spectral else ""
    out = shader_dir / f"{out_name}{tag}{spectral_suffix}.spv"
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    mtlx_genslang = shader_dir.parent / "mtlx" / "genslang"
    cmd = [
        slangc, str(src), "-target", "spirv", "-entry", entry, "-stage", "compute",
        "-I", str(shader_dir), "-I", str(mtlx_genslang),
        "-D", "SKINNY_COMPUTE_PIPELINE=1", "-D", "SKINNY_WAVEFRONT=1",
        *(("-D", "SKINNY_SPECTRAL=1") if spectral else ()),
        *defines, "-fvk-use-scalar-layout",
        "-o", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slang compilation failed ({module}:{entry}):\n{result.stderr}")
    return out


class _VkPathRecorder:
    """Vulkan adapter for :func:`skinny.wavefront_driver.record_path_loop`.

    Sequences :class:`WavefrontPathPass`'s existing ``vk`` command helpers,
    reproducing the historical inline ``record_dispatch`` byte-for-byte. Holds
    the frame command buffer + the bound scene descriptor set and exposes the
    pass's pipelines, queue buffers, and plugins through the backend-neutral
    :class:`skinny.wavefront_driver.WavefrontRecorder` protocol.
    """

    def __init__(self, p: "WavefrontPathPass", cmd, scene_set) -> None:
        self._p = p
        self._cmd = cmd
        self._scene = scene_set
        # COMPUTE→COMPUTE (+ indirect read for the args buffer build_args wrote).
        self._cbarrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
            | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT)

    @property
    def stream_size(self) -> int:
        return self._p.stream_size

    @property
    def has_neural(self) -> bool:
        return self._p._neural is not None

    @property
    def has_restir(self) -> bool:
        return self._p._restir is not None

    def barrier(self) -> None:
        vk.vkCmdPipelineBarrier(
            self._cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            | vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, [self._cbarrier], 0, None, 0, None)

    def flush_heavy_eye(self) -> None:
        # No-op on Vulkan: no GPU watchdog, so the whole frame records into one
        # command buffer as before (change wavefront-nonflat-tiled-fallback).
        pass

    def clear_counts(self) -> None:
        cmd = self._cmd
        cnt = self._p._buffers["slot_count"]
        cur = self._p._buffers["slot_cursor"]
        vk.vkCmdFillBuffer(cmd, cnt.buffer, 0, cnt.size, 0)
        vk.vkCmdFillBuffer(cmd, cur.buffer, 0, cur.size, 0)
        tb = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
        vk.vkCmdPipelineBarrier(
            cmd, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [tb], 0, None, 0, None)

    def push_tile(self, stream_base: int) -> None:
        self._p._push(self._cmd, 0, [stream_base, 0, self._p.stream_size])

    def dispatch_full(self, entry: str) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        self._p._dispatch(self._cmd)

    def dispatch_one(self, entry: str) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        vk.vkCmdDispatch(self._cmd, 1, 1, 1)

    def shade(self, slot: int, entry: str) -> None:
        self._p._push(self._cmd, 4, [slot])           # shadeSlot
        self._p._bind(self._cmd, entry, self._scene)
        vk.vkCmdDispatchIndirect(self._cmd, self._p._indirect_buf, slot * 12)

    def neural_prepass(self) -> None:
        self._p._neural.record(self._cmd, self._scene)

    def restir_primary_direct(self) -> None:
        self._p._restir.record_primary_direct(self._cmd, self._scene)


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
    NUM_SLOTS = 2  # lockstep with WF_NUM_SLOTS (0 = flat, 1 = non-flat catch-all)

    HIT_STRIDE = 96  # ≥ sizeof(HitInfo) (≈92 B scalar) — headroom
    NEURAL_STRIDE = 32  # = sizeof(WfNeuralSample) (interfaces.slang): wi,pdf,version,valid,2×pad
    REC_VERTEX_STRIDE = 76  # = sizeof(RecVertex) (wavefront/wf_records.slang): 6×float3 + uint

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 state_buffer, state_range: int, hit_buffer, hit_range: int,
                 stream_size: int, num_pixels: int, build_catchall: bool = True,
                 record_capacity: int = 0, neural_config=None,
                 spectral: bool = False) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)   # slots in the path-state buffer
        self.num_pixels = int(num_pixels)     # total pixels to cover, tiled
        self.build_catchall = bool(build_catchall)
        # Spectral wavefront variant (spectral-wavefront 5.2): compiles the staged
        # kernels with -DSKINNY_SPECTRAL so the shared carriers/helpers take their
        # hero-wavelength spectral branch. RGB (default) stays byte-identical.
        self._spectral = bool(spectral)
        # Wavefront-native path records (change wavefront-native-path-records):
        # per-lane vertex stack (binding 9) + stacked count (binding 10) in this
        # pass's set 1, in SEPARATE buffers (NOT inside the by-value-copied
        # WavefrontPathState). Full-size only while online training drives the
        # record drain; 1-element dummies otherwise so the default render is
        # byte-identical and pays no extra VRAM/bandwidth. The shade/terminate
        # kernels touch them only under fc.recordMode.
        self.record_capacity = int(record_capacity)
        # Neural size/precision config (study change neural-precision-size-study).
        # The flat/catch-all shade kernels import the inline flow inverse
        # (proposal → neural_proposal → neural_flow), so they compile against the
        # selected NF_* dims + NF_WT/NF_CT precision. Default config → no `-D` and
        # an empty tag → the shade .spv is byte-identical to the shipped kernel.
        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        self._neural_config = neural_config
        self._nf_defines = neural_config.slang_defines()
        self._nf_tag = f"_{neural_config.cache_tag}" if self._nf_defines else ""
        # Optional reuse plugin (ReSTIR DI) — scheduled at bounce 0 between the
        # primary intersect and the shade. None = identity reuse (stock NEE).
        self._restir = None
        # Optional neural-proposal pre-pass — scheduled EVERY bounce between
        # scatter and shade, writing the per-lane (wi, pdf) the flat shade reads.
        self._neural = None

        # The flat shade kernel handles flat + MaterialX-graph materials and
        # imports no skin/python (small). The heavy catch-all (wfPathShade) is
        # compiled only when the scene actually has a non-flat material — so a
        # common flat scene never compiles the ~2.8 MB kernel.
        entries = [
            ("wfPathGenerate", "wavefront/_wfpath_generate"),
            ("wfPathIntersect", "wavefront/_wfpath_intersect"),
            ("wfBuildArgs", "wavefront/_wfpath_buildargs"),
            ("wfScatter", "wavefront/_wfpath_scatter"),
            ("wfPathShadeFlat", "wavefront/_wfpath_shade_flat"),
            ("wfPathResolve", "wavefront/_wfpath_resolve"),
        ]
        if self.build_catchall:
            entries.append(("wfPathShade", "wavefront/_wfpath_shade"))
        modules = {}
        for entry, out_name in entries:
            spv = _compile_full_spv(shader_dir, "wavefront/wavefront_path", entry, out_name,
                                    defines=self._nf_defines, tag=self._nf_tag,
                                    spectral=self._spectral)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        # Counting-sort queue buffers (this pass owns them): laneSlot (2),
        # slotCount (3), slotOffset (4), slotQueue (5), slotCursor (6),
        # indirectArgs (7). slotCount/cursor are zeroed each bounce; indirectArgs
        # needs INDIRECT_BUFFER usage for vkCmdDispatchIndirect.
        self._buffers = {}
        self._buffers["lane_slot"] = StorageBuffer(ctx, self.stream_size * 4)
        self._buffers["slot_count"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["slot_offset"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["slot_queue"] = StorageBuffer(ctx, self.stream_size * 4)
        self._buffers["slot_cursor"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["indirect"] = StorageBuffer(ctx, self.NUM_SLOTS * 12, indirect=True)
        self._indirect_buf = self._buffers["indirect"].buffer
        # Per-lane neural-proposal forward sample (set 1, binding 8). Owned here
        # (always allocated, zero-init valid==0); the wavefront neural pre-pass
        # writes it when the neural proposal is active. The flat shade kernel
        # reads wfNeural[i] — so the binding is always in this set's layout.
        self._buffers["neural"] = StorageBuffer(ctx, self.stream_size * self.NEURAL_STRIDE)
        self.neural_buf = self._buffers["neural"]
        # Record stack (9) + per-lane count (10). Dummy (1 element) when not
        # recording; the shaders never touch them then (fc.recordMode == 0).
        rec_stack_elems = max(1, self.record_capacity * self.MAX_BOUNCES)
        rec_count_elems = max(1, self.record_capacity)
        self._buffers["rec_stack"] = StorageBuffer(ctx, rec_stack_elems * self.REC_VERTEX_STRIDE)
        self._buffers["rec_count"] = StorageBuffer(ctx, rec_count_elems * 4)

        # Set 1: path-state (0), hit (1), the 6 queue buffers (2..7), neural (8),
        # record stack (9), record count (10).
        num_bindings = 11
        state_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in range(num_bindings)
        ]
        self._state_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=num_bindings, pBindings=state_bindings), None)

        # Pipeline layout: [set 0 = megakernel scene set, set 1 = path state] +
        # a 12-byte push constant {streamBase, shadeSlot, streamSize}.
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=12)
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
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=num_bindings)]), None)
        self._state_set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._state_layout]))[0]
        bound = [
            (state_buffer, state_range), (hit_buffer, hit_range),
            (self._buffers["lane_slot"].buffer, self._buffers["lane_slot"].size),
            (self._buffers["slot_count"].buffer, self._buffers["slot_count"].size),
            (self._buffers["slot_offset"].buffer, self._buffers["slot_offset"].size),
            (self._buffers["slot_queue"].buffer, self._buffers["slot_queue"].size),
            (self._buffers["slot_cursor"].buffer, self._buffers["slot_cursor"].size),
            (self._buffers["indirect"].buffer, self._buffers["indirect"].size),
            (self._buffers["neural"].buffer, self._buffers["neural"].size),
            (self._buffers["rec_stack"].buffer, self._buffers["rec_stack"].size),
            (self._buffers["rec_count"].buffer, self._buffers["rec_count"].size),
        ]
        writes = []
        for b, (buf, rng) in enumerate(bound):
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

    def _push(self, cmd, offset, values) -> None:
        import struct

        import cffi
        data = struct.pack(f"{len(values)}I", *[int(v) for v in values])
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            int(offset), len(data), buf)

    def set_restir(self, restir) -> None:
        """Attach (or clear) the ReSTIR DI reuse plugin scheduled at bounce 0."""
        self._restir = restir

    def set_neural(self, neural) -> None:
        """Attach (or clear) the neural-proposal pre-pass run every bounce."""
        self._neural = neural

    def record_dispatch(self, cmd, scene_set) -> None:
        """Record the tiled, counting-sorted bounce loop into ``cmd``.

        The stage order lives in the backend-neutral
        :func:`skinny.wavefront_driver.record_path_loop`; this method supplies
        the Vulkan adapter (:class:`_VkPathRecorder`) that records each primitive
        with ``vk`` commands. The Metal backend implements the same protocol with
        a slang-rhi encoder, so both share one definition of the loop."""
        from skinny.wavefront_driver import record_path_loop

        record_path_loop(
            _VkPathRecorder(self, cmd, scene_set),
            num_pixels=self.num_pixels,
            stream_size=self.stream_size,
            max_bounces=self.MAX_BOUNCES,
            build_catchall=self.build_catchall,
        )

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._state_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)
        for buf in self._buffers.values():
            buf.destroy()


class _VkSppmRecorder:
    """Vulkan adapter for :func:`skinny.wavefront_driver.record_sppm_loop`.

    Records the SPPM stage primitives with ``vk`` commands. All SPPM dispatches
    have host-known counts (num_pixels / num_cells / photons), so every stage is
    a plain ``vkCmdDispatch`` — no indirect dispatch, hence no Metal-style
    readback stall on either backend.
    """

    def __init__(self, p: "WavefrontSppmPass", cmd, scene_set) -> None:
        self._p = p
        self._cmd = cmd
        self._scene = scene_set
        self._cbarrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)

    @property
    def stream_size(self) -> int:
        return self._p.stream_size

    def barrier(self) -> None:
        vk.vkCmdPipelineBarrier(
            self._cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [self._cbarrier], 0, None, 0, None)

    def flush_heavy_eye(self) -> None:
        # No-op on Vulkan: no GPU watchdog (change
        # wavefront-nonflat-tiled-fallback).
        pass

    def flush(self) -> None:
        # No-op on Vulkan: no GPU watchdog (SPPM phase-boundary flush is a
        # Metal-only guard, change spectral-wavefront).
        pass

    def _fill(self, buf, offset: int, size: int) -> None:
        vk.vkCmdFillBuffer(self._cmd, buf.buffer, int(offset), int(size), 0)

    def _transfer_barrier(self) -> None:
        tb = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
        vk.vkCmdPipelineBarrier(
            self._cmd, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [tb], 0, None, 0, None)

    def clear_visible_points(self) -> None:
        # Zero the persistent visible-point buffer at frame 0 so the n==0
        # first-activation radius init in wfSppmEye is reliable.
        vp = self._p._buffers["visible_points"]
        self._fill(vp, 0, vp.size)
        self._transfer_barrier()

    def clear_grid(self) -> None:
        # Zero gridCount [0 .. nc) and gridCursor [2nc .. 3nc); gridOffset and
        # sortedIdx are fully overwritten by the scan / scatter stages.
        grid = self._p._buffers["grid"]
        nc_bytes = self._p.num_cells * 4
        self._fill(grid, 0, nc_bytes)               # gridCount
        self._fill(grid, 2 * nc_bytes, nc_bytes)    # gridCursor
        self._transfer_barrier()

    def clear_accum(self) -> None:
        acc = self._p._buffers["accum"]
        self._fill(acc, 0, acc.size)
        self._transfer_barrier()

    def push_tile(self, stream_base: int) -> None:
        self._p._push(self._cmd, 0, [stream_base, 0, self._p.stream_size])

    def dispatch_full(self, entry: str) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        groups = (self._p.stream_size + self._p._GROUP - 1) // self._p._GROUP
        vk.vkCmdDispatch(self._cmd, max(groups, 1), 1, 1)

    def dispatch_one(self, entry: str) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        vk.vkCmdDispatch(self._cmd, 1, 1, 1)

    def dispatch_count(self, entry: str, count: int, group_size: int) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        groups = (int(count) + group_size - 1) // group_size
        vk.vkCmdDispatch(self._cmd, max(groups, 1), 1, 1)


class WavefrontSppmPass:
    """Staged wavefront Stochastic Progressive Photon Mapping pass (change
    photon-mapping-sppm). Owns eight compute pipelines compiled from
    ``integrators/wavefront_sppm.slang`` and four set-1 buffers; ``record_dispatch``
    records one SPPM pass (== one accumulation frame) via the backend-neutral
    :func:`skinny.wavefront_driver.record_sppm_loop`.

    Set 0 is the renderer's shared megakernel scene set (provides ``fc`` with the
    SPPM tail + scene geometry / materials / lights). Set 1 binding 0..3 are the
    visible-point, per-pass accumulator, grid, and scan-scratch buffers, sized by
    ``num_pixels`` (the persistent per-pixel estimator), NOT ``stream_size``.
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] on the tiled eye/update kernels
    STREAM_CAP = 1 << 20

    _ENTRIES = [
        ("wfSppmEye", "integrators/_wfsppm_eye"),
        ("wfSppmGridCount", "integrators/_wfsppm_grid_count"),
        ("wfSppmGridScanBlock", "integrators/_wfsppm_scan_block"),
        ("wfSppmGridScanBlockSums", "integrators/_wfsppm_scan_sums"),
        ("wfSppmGridScanAdd", "integrators/_wfsppm_scan_add"),
        ("wfSppmGridScatter", "integrators/_wfsppm_scatter"),
        ("wfSppmPhotonTrace", "integrators/_wfsppm_photon"),
        ("wfSppmUpdate", "integrators/_wfsppm_update"),
    ]

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 stream_size: int, num_pixels: int, spectral: bool = False) -> None:
        from skinny.wavefront_layout import (
            sppm_buffer_sizes,
            sppm_grid_buffer_sizes,
            sppm_grid_cell_count,
        )

        self.ctx = ctx
        self.num_pixels = int(num_pixels)
        self.stream_size = int(min(stream_size, self.STREAM_CAP))
        self.num_cells = sppm_grid_cell_count(self.num_pixels)
        # Spectral wavefront variant (spectral-wavefront 5.2): per-λ photon flux +
        # per-pass hero-wavelength resolve; RGB (default) stays byte-identical.
        self._spectral = bool(spectral)

        modules = {}
        for entry, out_name in self._ENTRIES:
            spv = _compile_full_spv(shader_dir, "integrators/wavefront_sppm", entry, out_name,
                                    spectral=self._spectral)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        grid_sizes = sppm_grid_buffer_sizes(self.num_pixels)
        # Spectral (Spectrum beta/ld + conductorMetalId; 4-channel SppmAccum)
        # widens both per-pixel buffers; RGB (spectral=False) stays byte-identical.
        sppm_sizes = sppm_buffer_sizes(self.num_pixels, spectral=self._spectral)
        self._buffers = {}
        self._buffers["visible_points"] = StorageBuffer(ctx, sppm_sizes["visible_points"])
        self._buffers["accum"] = StorageBuffer(ctx, sppm_sizes["sppm_accum"])
        self._buffers["grid"] = StorageBuffer(ctx, grid_sizes["grid_combined"])
        self._buffers["scan"] = StorageBuffer(ctx, grid_sizes["scan_scratch"])

        # Set 1: visiblePoints (0), accum (1), grid (2), scanScratch (3).
        num_bindings = 4
        set_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in range(num_bindings)
        ]
        self._state_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=num_bindings, pBindings=set_bindings), None)

        # Pipeline layout: [scene set 0, sppm set 1] + 12-byte push constant
        # {streamBase, shadeSlot(unused), streamSize}.
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=12)
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
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=num_bindings)]), None)
        self._state_set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._state_layout]))[0]
        bound = [self._buffers["visible_points"], self._buffers["accum"],
                 self._buffers["grid"], self._buffers["scan"]]
        writes = []
        for b, buf in enumerate(bound):
            info = vk.VkDescriptorBufferInfo(buffer=buf.buffer, offset=0, range=buf.size)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._state_set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _bind(self, cmd, entry, scene_set) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._state_set], 0, None)

    def _push(self, cmd, offset, values) -> None:
        import struct

        import cffi
        data = struct.pack(f"{len(values)}I", *[int(v) for v in values])
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, int(offset), len(data), buf)

    def record_dispatch(self, cmd, scene_set, *, photons: int, first_frame: bool) -> None:
        """Record one SPPM pass into ``cmd`` (the per-frame command buffer)."""
        from skinny.wavefront_driver import record_sppm_loop

        record_sppm_loop(
            _VkSppmRecorder(self, cmd, scene_set),
            num_pixels=self.num_pixels, stream_size=self.stream_size,
            num_cells=self.num_cells, photons=int(photons), first_frame=bool(first_frame))

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._state_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)
        for buf in self._buffers.values():
            buf.destroy()


class _VkMltRecorder:
    """Vulkan adapter for the :mod:`skinny.wavefront_driver` MLT phase
    recorders (``record_mlt_bootstrap`` / ``record_mlt_init`` /
    ``record_mlt_frame``, change mlt-integrator).

    Every MLT dispatch has a host-known count (bootstrap samples / chains /
    pixels), so each stage is a plain ``vkCmdDispatch`` over a 64-wide window
    pushed via the shared 12-byte ``{streamBase, shadeSlot, streamSize}``
    tile push constant — no indirect dispatch, no readback stall.
    """

    def __init__(self, p: "WavefrontMltPass", cmd, scene_set) -> None:
        self._p = p
        self._cmd = cmd
        self._scene = scene_set
        self._cbarrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)

    def barrier(self) -> None:
        vk.vkCmdPipelineBarrier(
            self._cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [self._cbarrier], 0, None, 0, None)

    def flush(self) -> None:
        # No-op on Vulkan: no GPU watchdog (the MLT sub-batch flush is a
        # Metal-only guard — dispatch hygiene, design D7).
        pass

    def push_window(self, base: int, size: int) -> None:
        # MLT tile push: streamBase + EXACT streamSize (shadeSlot unused).
        self._p._push(self._cmd, 0, [int(base), 0, int(size)])

    def dispatch_count(self, entry: str, count: int, group_size: int) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        groups = (int(count) + group_size - 1) // group_size
        vk.vkCmdDispatch(self._cmd, max(groups, 1), 1, 1)


class WavefrontMltPass:
    """Staged wavefront PSSMLT pass (change mlt-integrator). Owns the five
    MLT chain buffers (sizes from ``wavefront_layout.mlt_buffer_sizes``) and
    four compute pipelines compiled from ``wavefront/wavefront_mlt.slang``
    under ``-DSKINNY_MLT=1`` (which swaps common.slang's RNG for the
    primary-sample-space sampler — distinct ``_mlt`` .spv names so the RGB
    kernel cache is never aliased).

    Set 0 is the renderer's shared wavefront scene descriptor set, whose
    layout carries the MLT bindings 52–56 (``ComputePipeline.mlt_bindings``);
    the renderer writes this pass's buffers into those slots at build. There
    is no set 1 — the kernels bind everything through the scene set.

    Per accumulation reset the renderer runs the synchronous host round-trip:
    ``record_bootstrap`` → ``read_bootstrap_weights`` →
    ``mlt_bootstrap.resample_chain_seeds`` → ``upload_chain_seeds`` →
    ``record_init``; then every frame records ``record_frame``. The
    bootstrap-normalization ``b`` and the ``seeded`` flag are renderer-owned
    state stashed on the pass.
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] on all four MLT kernels

    _ENTRIES = [
        ("wfMltBootstrap", "wavefront/_wfmlt_bootstrap"),
        ("wfMltInit", "wavefront/_wfmlt_init"),
        ("wfMltMutate", "wavefront/_wfmlt_mutate"),
        ("wfMltResolve", "wavefront/_wfmlt_resolve"),
    ]

    # Binding → mlt_buffer_sizes key (bindings 52–56 of the scene set-0
    # layout; 52 lives in common.slang's SKINNY_MLT block, 53–56 in
    # wavefront_mlt.slang).
    _BINDINGS = (
        (52, "mlt_primary_samples"),
        (53, "mlt_chain_meta"),
        (54, "mlt_current_records"),
        (55, "mlt_bootstrap_weights"),
        (56, "mlt_chain_seeds"),
    )

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 num_pixels: int, num_chains: int, bootstrap_samples: int,
                 spectral: bool = False) -> None:
        from skinny.wavefront_layout import mlt_buffer_sizes

        self.ctx = ctx
        self.num_pixels = int(num_pixels)
        self.num_chains = int(num_chains)
        self.bootstrap_samples = int(bootstrap_samples)
        self.spectral = bool(spectral)
        # Renderer-owned per-accumulation state: the bootstrap-normalization b
        # (resolve scale, design D4) and whether the chains have been seeded.
        self.b = 0.0
        self.seeded = False

        modules = {}
        for entry, out_name in self._ENTRIES:
            # Spectral MLT (change spectral-mlt) adds -DSKINNY_SPECTRAL and a
            # distinct `_spectral` .spv suffix (via _compile_full_spv) so the
            # SpectralBDPTIntegrator target never aliases the RGB MLT cache.
            spv = _compile_full_spv(
                shader_dir, "wavefront/wavefront_mlt", entry, out_name,
                defines=("-D", "SKINNY_MLT=1"), tag="_mlt", spectral=self.spectral)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        sizes = mlt_buffer_sizes(self.num_chains, self.bootstrap_samples)
        self._buffers = {key: StorageBuffer(ctx, sizes[key])
                         for _, key in self._BINDINGS}

        # Pipeline layout: [scene set 0 only] + the shared 12-byte tile push
        # constant {streamBase, shadeSlot(unused), streamSize}. The scene
        # layout already declares 52–56, so no pass-owned descriptor set.
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=12)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=1, pSetLayouts=[scene_set_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)

        self._pipelines = {}
        for entry, mod in modules.items():
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
            self._pipelines[entry] = vk.vkCreateComputePipelines(
                ctx.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None)[0]

    @property
    def descriptor_bindings(self):
        """``(binding, StorageBuffer)`` pairs the renderer writes into the
        shared scene descriptor sets (slots 52–56)."""
        return tuple((b, self._buffers[key]) for b, key in self._BINDINGS)

    def _bind(self, cmd, entry, scene_set) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 1, [scene_set], 0, None)

    def _push(self, cmd, offset, values) -> None:
        import struct

        import cffi
        data = struct.pack(f"{len(values)}I", *[int(v) for v in values])
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, int(offset), len(data), buf)

    def record_bootstrap(self, cmd, scene_set) -> None:
        """Record the bootstrap phase (one thread = one bootstrap sample)."""
        from skinny.wavefront_driver import record_mlt_bootstrap

        record_mlt_bootstrap(
            _VkMltRecorder(self, cmd, scene_set),
            bootstrap_samples=self.bootstrap_samples, num_chains=self.num_chains)

    def record_init(self, cmd, scene_set) -> None:
        """Record the chain-init phase (replays the resampled seeds the host
        uploaded via :meth:`upload_chain_seeds`)."""
        from skinny.wavefront_driver import record_mlt_init

        record_mlt_init(
            _VkMltRecorder(self, cmd, scene_set), num_chains=self.num_chains)

    def record_frame(self, cmd, scene_set, *, iterations: int) -> None:
        """Record one MLT accumulation frame (mutate × iterations + resolve)."""
        from skinny.wavefront_driver import record_mlt_frame

        record_mlt_frame(
            _VkMltRecorder(self, cmd, scene_set),
            num_pixels=self.num_pixels, num_chains=self.num_chains,
            iterations=int(iterations))

    def read_bootstrap_weights(self):
        """Host readback of the bootstrap scalar contributions (float32,
        ``bootstrap_samples`` entries) — the device→staging→map path every
        StorageBuffer carries."""
        import numpy as np

        raw = self._buffers["mlt_bootstrap_weights"].download_sync(
            self.bootstrap_samples * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def upload_chain_seeds(self, seeds) -> None:
        """Upload the resampled per-chain bootstrap indices (uint32,
        ``num_chains`` entries)."""
        import numpy as np

        arr = np.ascontiguousarray(seeds, dtype=np.uint32)
        if arr.size != self.num_chains:
            raise ValueError(
                f"MLT chain seeds: expected {self.num_chains} entries, got {arr.size}")
        self._buffers["mlt_chain_seeds"].upload_sync(arr.tobytes())

    def destroy(self) -> None:
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)
        for buf in self._buffers.values():
            buf.destroy()


class WavefrontNeuralProposalPass:
    """Neural directional-proposal pre-pass (wavefront-only).

    One compute kernel (``wavefront/neural_proposal_pass.slang::wfNeuralProposal``):
    for every live lane it reads the path-state + HitInfo, builds the spline-flow
    condition, draws a forward sample, and writes the per-lane ``(wi, pdf)`` into
    the buffer the flat shade kernel reads (WavefrontPathPass set-1 binding 8).
    Scheduled between scatter and shade each bounce, mirroring ``RestirDiPass``.

    Set 0 is the renderer's shared scene set (provides ``fc`` + the neural weight
    buffers at 33/34/35); set 1 is this pass's own state / hit / neural-out. The
    out buffer is owned by ``WavefrontPathPass`` and shared here, so the two
    passes index the same per-lane records.
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] in neural_proposal_pass.slang

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 state_buffer, state_range: int, hit_buffer, hit_range: int,
                 neural_buffer, neural_range: int, stream_size: int,
                 network_version: int = 0, neural_config=None) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.network_version = int(network_version)

        # Forward-pass compile at the selected size/precision (study change
        # neural-precision-size-study). Default config → no `-D`, empty tag →
        # the shipped `_wfneural.spv`.
        if neural_config is None:
            from skinny.sampling.neural_weights import NeuralBuildConfig
            neural_config = NeuralBuildConfig()
        nf_defines = neural_config.slang_defines()
        nf_tag = f"_{neural_config.cache_tag}" if nf_defines else ""
        spv = _compile_full_spv(shader_dir, "wavefront/neural_proposal_pass",
                                "wfNeuralProposal", "wavefront/_wfneural",
                                defines=nf_defines, tag=nf_tag)
        code = spv.read_bytes()
        self._module = vk.vkCreateShaderModule(
            ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)

        # Set 1: state (0), hit (1), neural-out (2).
        set_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in range(3)
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=3, pBindings=set_bindings), None)

        # Push constant: {streamSize, networkVersion, _pad0, _pad1} = 16 B.
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=16)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=2, pSetLayouts=[scene_set_layout, self._set_layout],
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
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=3)]), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._set_layout]))[0]
        bound = [(state_buffer, state_range), (hit_buffer, hit_range),
                 (neural_buffer, neural_range)]
        writes = []
        for b, (buf, rng) in enumerate(bound):
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def record(self, cmd, scene_set) -> None:
        """Dispatch the forward pass over the whole stream (one thread per lane).

        Records bind + push + dispatch; the caller barriers around it (the shade
        kernel reads the neural buffer this writes)."""
        import struct

        import cffi
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._set], 0, None)
        data = struct.pack("IIII", self.stream_size, self.network_version, 0, 0)
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                              0, len(data), buf)
        groups = (self.stream_size + self._GROUP - 1) // self._GROUP
        vk.vkCmdDispatch(cmd, groups, 1, 1)

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        vk.vkDestroyPipeline(self.ctx.device, self._pipeline, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        vk.vkDestroyShaderModule(self.ctx.device, self._module, None)
        self._buffers = {}


class RestirDiPass:
    """ReSTIR DI wavefront pass set (the GPU side; the host-side selector is
    sampling.reuse.RestirDiReuse).

    Milestone 1 (plumbing): one compute pass (``restir/restir_primary.slang::
    restirPrimaryDirect``) computes primary-hit direct lighting and adds it into
    the path-state radiance, scheduled at bounce 0 by WavefrontPathPass's reuse
    hook (the shade kernel's depth-0 reuseDirect is gated to zero, so this is the
    sole primary-NEE source). Shares the renderer's scene descriptor set (set 0)
    and the wavefront path-state + hit buffers (set 1). Reservoir buffers + the
    RIS / spatial / temporal passes land in later milestones.
    """

    _GROUP = 64  # matches [numthreads(64,1,1)] in restir_primary.slang
    RESERVOIR_STRIDE = 32  # ≥ sizeof(Reservoir) (28 B scalar) — headroom
    # Default ReSTIR config (mirrors restir_primary.slang RestirPC). flags bit0
    # spatial, bit1 temporal. Renderer overrides via RestirDiReuse.
    DEFAULT_CONFIG = dict(flags=0x3, mLight=8, spatialK=5, spatialRadius=16.0,
                          normalThresh=0.9, depthThresh=0.1, mCap=20, mBsdf=1)

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 state_buffer, state_range: int, hit_buffer, hit_range: int,
                 stream_size: int, config: dict | None = None) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Three pipelines: fill (initial RIS → reservoirA + G-buffer) → spatial
        # (reservoirA → reservoirB, domain-checked neighbour merge) → resolve
        # (reservoirB → shadow ray → radiance).
        self._modules = {}
        self._pipelines = {}
        for entry, out_name in (("restirFill", "restir/_restir_fill"),
                                ("restirSpatial", "restir/_restir_spatial"),
                                ("restirResolve", "restir/_restir_resolve")):
            spv = _compile_full_spv(shader_dir, "restir/restir_primary", entry, out_name)
            code = spv.read_bytes()
            self._modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)

        # Per-pixel ReSTIR-owned buffers: ping-pong reservoirs (A/B) + a G-buffer
        # (pos+normal, 24 B → 32 B headroom) for the spatial domain check.
        self._resA = StorageBuffer(ctx, self.stream_size * self.RESERVOIR_STRIDE)
        self._resB = StorageBuffer(ctx, self.stream_size * self.RESERVOIR_STRIDE)
        self._gbuffer = StorageBuffer(ctx, self.stream_size * 32)

        # Set 1: state (0) + hit (1) [shared] + reservoirA (2) + reservoirB (3) +
        # G-buffer (4) [owned].
        set1_bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in range(5)
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=5, pBindings=set1_bindings), None)

        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=36)
        self._pipe_layout = vk.vkCreatePipelineLayout(
            ctx.device, vk.VkPipelineLayoutCreateInfo(
                setLayoutCount=2, pSetLayouts=[scene_set_layout, self._set_layout],
                pushConstantRangeCount=1, pPushConstantRanges=[push_range]), None)

        for entry, mod in self._modules.items():
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
            self._pipelines[entry] = vk.vkCreateComputePipelines(
                ctx.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkComputePipelineCreateInfo(stage=stage, layout=self._pipe_layout)], None)[0]

        self._pool = vk.vkCreateDescriptorPool(
            ctx.device, vk.VkDescriptorPoolCreateInfo(
                maxSets=1, poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5)]), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._set_layout]))[0]
        writes = []
        bound = [(state_buffer, state_range), (hit_buffer, hit_range),
                 (self._resA.buffer, self._resA.size),
                 (self._resB.buffer, self._resB.size),
                 (self._gbuffer.buffer, self._gbuffer.size)]
        for b, (buf, rng) in enumerate(bound):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _barrier(self, cmd) -> None:
        b = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
        vk.vkCmdPipelineBarrier(
            cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [b], 0, None, 0, None)

    def record_primary_direct(self, cmd, scene_set) -> None:
        """Dispatch the ReSTIR primary-direct passes (fill → resolve) over one
        tile's lanes. Called at bounce 0 (after the primary intersect, before
        shade)."""
        import struct

        import cffi
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._set], 0, None)
        c = self.config
        # RestirPC: streamSize, flags, mLight, spatialK, spatialRadius,
        # normalThresh, depthThresh, mCap, mBsdf (scalar layout, 36 B).
        data = struct.pack("IIIIfffII", self.stream_size, int(c["flags"]),
                           int(c["mLight"]), int(c["spatialK"]), float(c["spatialRadius"]),
                           float(c["normalThresh"]), float(c["depthThresh"]), int(c["mCap"]),
                           int(c.get("mBsdf", 1)))
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 36, buf)
        groups = (self.stream_size + self._GROUP - 1) // self._GROUP
        for k, entry in enumerate(("restirFill", "restirSpatial", "restirResolve")):
            if k > 0:
                self._barrier(cmd)
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
            vk.vkCmdDispatch(cmd, groups, 1, 1)

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)
        self._resA.destroy()
        self._resB.destroy()
        self._gbuffer.destroy()


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


class _VkBdptRecorder:
    """Vulkan adapter for :func:`skinny.wavefront_driver.record_bdpt_loop`.

    Sequences :class:`WavefrontBdptPass`'s existing ``vk`` command helpers,
    reproducing the historical inline ``record_dispatch`` byte-for-byte.
    ``shade(slot, entry)`` is the BDPT "indirect over a compacted slot queue"
    primitive (bounce-eye/-light + the split connect kernels); the neural and
    ReSTIR hooks never fire (``has_neural``/``has_restir`` are ``False``)."""

    def __init__(self, p: "WavefrontBdptPass", cmd, scene_set) -> None:
        self._p = p
        self._cmd = cmd
        self._scene = scene_set
        # COMPUTE→COMPUTE (+ indirect read of the args build_args wrote).
        self._cbarrier = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
            | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT)

    @property
    def stream_size(self) -> int:
        return self._p.stream_size

    @property
    def has_neural(self) -> bool:
        return False

    @property
    def has_restir(self) -> bool:
        return False

    def barrier(self) -> None:
        vk.vkCmdPipelineBarrier(
            self._cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            | vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, [self._cbarrier], 0, None, 0, None)

    def flush_heavy_eye(self) -> None:
        # No-op on Vulkan: no GPU watchdog, so the whole frame records into one
        # command buffer as before (change wavefront-nonflat-tiled-fallback).
        pass

    def clear_counts(self) -> None:
        cmd = self._cmd
        cnt = self._p._buffers["slot_count"]
        cur = self._p._buffers["slot_cursor"]
        # WAR guard: a prior stage (the previous bounce's queue read, or the
        # prior tile's connect) read slot_count/cursor via the indirect
        # dispatch. The COMPUTE→COMPUTE barrier does NOT order those reads
        # before the TRANSFER fill below, so without this the fill races them —
        # mild at one tile, badly corrupting at many. COMPUTE→TRANSFER fixes it.
        pre = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT)
        vk.vkCmdPipelineBarrier(
            cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, [pre], 0, None, 0, None)
        vk.vkCmdFillBuffer(cmd, cnt.buffer, 0, cnt.size, 0)
        vk.vkCmdFillBuffer(cmd, cur.buffer, 0, cur.size, 0)
        tb = vk.VkMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
        vk.vkCmdPipelineBarrier(
            cmd, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, [tb], 0, None, 0, None)

    def push_tile(self, stream_base: int) -> None:
        self._p._push(self._cmd, 0, [stream_base, 0, self._p.stream_size])

    def dispatch_full(self, entry: str) -> None:
        self._p._stage(self._cmd, entry, self._scene)

    def dispatch_one(self, entry: str) -> None:
        self._p._bind(self._cmd, entry, self._scene)
        vk.vkCmdDispatch(self._cmd, 1, 1, 1)

    def shade(self, slot: int, entry: str) -> None:
        self._p._push(self._cmd, 4, [slot])      # shadeSlot for wfBdptQueueLane
        self._p._bind(self._cmd, entry, self._scene)
        vk.vkCmdDispatchIndirect(self._cmd, self._p._indirect_buf, slot * 12)

    def neural_prepass(self) -> None:  # pragma: no cover — never scheduled
        raise NotImplementedError("bdpt loop has no neural pre-pass")

    def restir_primary_direct(self) -> None:  # pragma: no cover — never scheduled
        raise NotImplementedError("bdpt loop has no ReSTIR hook")


class WavefrontBdptPass:
    """Staged wavefront bidirectional path tracer (Phase 3). Pipelines from
    ``wavefront/wavefront_bdpt.slang``: walk (subpath build) → connect counting
    sort (classify / build_args / scatter) → split connect (nee / full,
    indirect-dispatched per slot) → resolve, over per-lane eye + light
    subpath-vertex buffers + an aux buffer + the connect counting-sort buffers
    (this pass owns them). Set 0 is the megakernel scene descriptor set (incl.
    lightSplatBuffer); set 1 holds the subpath/aux/queue buffers. The connect
    estimator is reused verbatim from integrators/bdpt.slang, so the accumulated
    image matches the megakernel bdpt. The strategy split skips dead lanes and
    runs the heavy generic+MIS double loop only for lanes with a ≥2-vertex light
    subpath. Matches the megakernel scope: flat first-hit, pinhole camera."""

    _GROUP = 64           # matches [numthreads(64, 1, 1)] in wavefront_bdpt.slang
    BDPT_MAX_VERTS = 7    # lockstep with bdpt.slang BDPT_MAX_VERTS
    VERTEX_STRIDE = 128   # ≥ sizeof(BDPTVertex) (≈120 B scalar) — headroom
    AUX_STRIDE = 128      # ≥ sizeof(WfBdptAux) (≈92 B scalar w/ eye-walk state)
    # Connect counting-sort slots — lockstep with WF_BDPT_SLOT_* in the shader.
    NUM_SLOTS = 2
    SLOT_NEE = 0
    SLOT_FULL = 1
    # Eye-walk extend bounces: gen-eye seeds eye[0..1], the loop extends eye[2..].
    EYE_BOUNCES = BDPT_MAX_VERTS - 2
    # Light-walk extend bounces: gen-light seeds light[0], the loop extends light[1..].
    LIGHT_BOUNCES = BDPT_MAX_VERTS - 1
    # Smaller cap than the path tracer: each lane owns 2×BDPT_MAX_VERTS vertices
    # (eye+light), so vertex VRAM = stream × 7 × 128 × 2. 1<<18 ≈ 470 MB.
    STREAM_CAP = 1 << 18

    WALK_MODES = ("fused", "eye", "eye_light")

    def __init__(self, ctx, shader_dir: Path, scene_set_layout,
                 eye_buf, light_buf, aux_buf, vert_range: int, aux_range: int,
                 stream_size: int, num_pixels: int,
                 walk_mode: str = "fused", spectral: bool = False) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.num_pixels = int(num_pixels)
        if walk_mode not in self.WALK_MODES:
            raise ValueError(f"unknown bdpt walk_mode {walk_mode!r} (expected {self.WALK_MODES})")
        self.walk_mode = walk_mode
        # Spectral wavefront variant (spectral-wavefront 5.2): per-λ eye/light
        # walks + splat resolve; RGB (default) stays byte-identical.
        self._spectral = bool(spectral)

        # The connect counting sort (classify / build_args / scatter) + split
        # connect (nee / full, indirect) + resolve are shared by all walk modes;
        # only the subpath-build kernels differ:
        #   fused      — one wfBdptWalk kernel (eye+light+splat); the S1 win.
        #   eye        — staged eye walk + fused light tail.
        #   eye_light  — fully staged eye + light walks + standalone splat.
        # Only the active mode's kernels are compiled/built (no wasted slangc).
        shared = [
            ("wfBdptClassify", "wavefront/_wfbdpt_classify"),
            ("wfBdptBuildArgs", "wavefront/_wfbdpt_buildargs"),
            ("wfBdptScatter", "wavefront/_wfbdpt_scatter"),
            ("wfBdptConnectNee", "wavefront/_wfbdpt_connect_nee"),
            ("wfBdptConnectFull", "wavefront/_wfbdpt_connect_full"),
            ("wfBdptResolve", "wavefront/_wfbdpt_resolve"),
        ]
        staged_eye = [
            ("wfBdptGenEye", "wavefront/_wfbdpt_gen_eye"),
            ("wfBdptWalkClassify", "wavefront/_wfbdpt_walk_classify"),
            ("wfBdptBounceEye", "wavefront/_wfbdpt_bounce_eye"),
        ]
        if walk_mode == "fused":
            entries = [("wfBdptWalk", "wavefront/_wfbdpt_walk")] + shared
        elif walk_mode == "eye":
            entries = staged_eye + [("wfBdptLightTail", "wavefront/_wfbdpt_light_tail")] + shared
        else:  # eye_light
            entries = staged_eye + [
                ("wfBdptGenLight", "wavefront/_wfbdpt_gen_light"),
                ("wfBdptBounceLight", "wavefront/_wfbdpt_bounce_light"),
                ("wfBdptSplat", "wavefront/_wfbdpt_splat"),
            ] + shared
        modules = {}
        for entry, out_name in entries:
            spv = _compile_full_spv(shader_dir, "wavefront/wavefront_bdpt", entry, out_name,
                                    spectral=self._spectral)
            code = spv.read_bytes()
            modules[entry] = vk.vkCreateShaderModule(
                ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        self._modules = modules

        # Connect-stage counting-sort buffers (this pass owns them), mirroring
        # WavefrontPathPass: laneSlot (3), slotCount (4), slotOffset (5),
        # slotQueue (6), slotCursor (7), indirectArgs (8). slotCount/cursor are
        # zeroed each tile; indirectArgs needs INDIRECT_BUFFER for the connect
        # vkCmdDispatchIndirect.
        self._buffers = {}
        self._buffers["lane_slot"] = StorageBuffer(ctx, self.stream_size * 4)
        self._buffers["slot_count"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["slot_offset"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["slot_queue"] = StorageBuffer(ctx, self.stream_size * 4)
        self._buffers["slot_cursor"] = StorageBuffer(ctx, self.NUM_SLOTS * 4)
        self._buffers["indirect"] = StorageBuffer(ctx, self.NUM_SLOTS * 12, indirect=True)
        self._indirect_buf = self._buffers["indirect"].buffer

        # Set 1: eye (0), light (1), aux (2), + the 6 counting-sort buffers (3..8).
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=b, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)
            for b in range(9)
        ]
        self._set_layout = vk.vkCreateDescriptorSetLayout(
            ctx.device, vk.VkDescriptorSetLayoutCreateInfo(
                bindingCount=9, pBindings=bindings), None)
        # Push constant {streamBase, shadeSlot, streamSize} (12 B).
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=12)
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
                    type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=9)]), None)
        self._set = vk.vkAllocateDescriptorSets(
            ctx.device, vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self._pool, descriptorSetCount=1,
                pSetLayouts=[self._set_layout]))[0]
        bound = [
            (eye_buf, vert_range), (light_buf, vert_range), (aux_buf, aux_range),
            (self._buffers["lane_slot"].buffer, self._buffers["lane_slot"].size),
            (self._buffers["slot_count"].buffer, self._buffers["slot_count"].size),
            (self._buffers["slot_offset"].buffer, self._buffers["slot_offset"].size),
            (self._buffers["slot_queue"].buffer, self._buffers["slot_queue"].size),
            (self._buffers["slot_cursor"].buffer, self._buffers["slot_cursor"].size),
            (self._buffers["indirect"].buffer, self._buffers["indirect"].size),
        ]
        writes = []
        for b, (buf, rng) in enumerate(bound):
            info = vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=rng)
            writes.append(vk.VkWriteDescriptorSet(
                dstSet=self._set, dstBinding=b, dstArrayElement=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info]))
        vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)

    def _bind(self, cmd, entry, scene_set) -> None:
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipelines[entry])
        vk.vkCmdBindDescriptorSets(
            cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipe_layout,
            0, 2, [scene_set, self._set], 0, None)

    def _dispatch_full(self, cmd) -> None:
        groups = (self.stream_size + self._GROUP - 1) // self._GROUP
        vk.vkCmdDispatch(cmd, groups, 1, 1)

    def _stage(self, cmd, entry, scene_set) -> None:
        """Bind + full-stream dispatch (walk / classify / scatter / resolve)."""
        self._bind(cmd, entry, scene_set)
        self._dispatch_full(cmd)

    def _push(self, cmd, offset, values) -> None:
        import struct

        import cffi
        data = struct.pack(f"{len(values)}I", *[int(v) for v in values])
        buf = cffi.FFI().new("char[]", data)
        vk.vkCmdPushConstants(
            cmd, self._pipe_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
            int(offset), len(data), buf)

    def record_dispatch(self, cmd, scene_set) -> None:
        """Record the tiled fully-staged bdpt loop into ``cmd``.

        The stage order lives in the backend-neutral
        :func:`skinny.wavefront_driver.record_bdpt_loop`; this method supplies
        the Vulkan adapter (:class:`_VkBdptRecorder`) that records each
        primitive with ``vk`` commands. The Metal backend implements the same
        protocol with a slang-rhi encoder, so both share one definition of the
        loop."""
        from skinny.wavefront_driver import record_bdpt_loop

        record_bdpt_loop(
            _VkBdptRecorder(self, cmd, scene_set),
            num_pixels=self.num_pixels,
            stream_size=self.stream_size,
            walk_mode=self.walk_mode,
            eye_bounces=self.EYE_BOUNCES,
            light_bounces=self.LIGHT_BOUNCES,
            slot_nee=self.SLOT_NEE,
            slot_full=self.SLOT_FULL,
        )

    def destroy(self) -> None:
        vk.vkDestroyDescriptorPool(self.ctx.device, self._pool, None)
        for p in self._pipelines.values():
            vk.vkDestroyPipeline(self.ctx.device, p, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._pipe_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.ctx.device, self._set_layout, None)
        for m in self._modules.values():
            vk.vkDestroyShaderModule(self.ctx.device, m, None)
        for buf in self._buffers.values():
            buf.destroy()
        self._buffers = {}


class WavefrontPasses:
    """Owns the wavefront stage buffers for a given stream size + material count.

    The path-state and queue buffers are sized to the active path stream; the
    per-material counting-sort buffers to the material count. They are
    reallocated when either changes (a scene reload that adds materials, or a
    stream-size retune).
    """

    def __init__(self, ctx, stream_size: int, num_materials: int,
                 spectral: bool = False) -> None:
        self.ctx = ctx
        self.stream_size = int(stream_size)
        self.num_materials = int(num_materials)
        self.buffer_sizes = queue_buffer_sizes(
            self.stream_size, self.num_materials, spectral=bool(spectral))
        self.buffers: dict[str, StorageBuffer] = {
            name: StorageBuffer(ctx, size) for name, size in self.buffer_sizes.items()
        }

    def destroy(self) -> None:
        """Release all stage buffers. Idempotent."""
        for buf in self.buffers.values():
            buf.destroy()
        self.buffers = {}
