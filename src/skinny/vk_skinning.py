"""Standalone GPU compute pipelines for UsdSkel skinning + BVH refit.

Kept separate from the main `ComputePipeline` (which hard-codes the 33-binding
main-pass layout): these own small descriptor set layouts so `main_pass` and its
SPIR-V are untouched (design D4). They run as an isolated one-shot submit before
the frame's render — no edit to the shared render command recording.

skin.slang deforms rest vertices into the shared `vertex_buffer`; bvh_refit.slang
refits each skinned mesh's BVH in `bvh_buffer`. Both read/write the same buffers
the main pass reads, so after the submit completes the render sees deformed
geometry. Vulkan only — the Metal backend is a separate follow-up.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import cffi
import numpy as np
import vulkan as vk

from skinny.vk_compute import StorageBuffer

_FFI = cffi.FFI()


def _push_buf(data: bytes):
    """python-vulkan binds `const void* pValues` via cffi; pass a sized char[]."""
    return _FFI.new("char[]", data)


_WORKGROUP = 64


def _compile(shader_dir: Path, module: str, entry: str) -> Path:
    """Compile shaders/<module>.slang → .spv (scalar layout). Returns the path."""
    src = shader_dir / f"{module}.slang"
    out = shader_dir / f"{module}.spv"
    slangc = shutil.which("slangc")
    if slangc is None:
        raise RuntimeError("slangc not found on PATH — install the Slang compiler")
    cmd = [
        slangc, str(src), "-target", "spirv", "-entry", entry,
        "-stage", "compute", "-I", str(shader_dir),
        "-fvk-use-scalar-layout", "-o", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slang compilation failed ({module}):\n{result.stderr}")
    return out


def _shader_module(ctx, spv: Path):
    code = spv.read_bytes()
    return vk.vkCreateShaderModule(
        ctx.device, vk.VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None
    )


def _ssbo_layout(ctx, count: int):
    """Descriptor set layout with `count` STORAGE_BUFFER bindings (0..count-1)."""
    bindings = [
        vk.VkDescriptorSetLayoutBinding(
            binding=i,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        )
        for i in range(count)
    ]
    return vk.vkCreateDescriptorSetLayout(
        ctx.device,
        vk.VkDescriptorSetLayoutCreateInfo(bindingCount=count, pBindings=bindings),
        None,
    )


def _pipeline(ctx, module, set_layout, push_size: int):
    push = vk.VkPushConstantRange(
        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=push_size
    )
    layout = vk.vkCreatePipelineLayout(
        ctx.device,
        vk.VkPipelineLayoutCreateInfo(
            setLayoutCount=1, pSetLayouts=[set_layout],
            pushConstantRangeCount=1, pPushConstantRanges=[push],
        ),
        None,
    )
    stage = vk.VkPipelineShaderStageCreateInfo(
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main",
    )
    pipe = vk.vkCreateComputePipelines(
        ctx.device, vk.VK_NULL_HANDLE, 1,
        [vk.VkComputePipelineCreateInfo(stage=stage, layout=layout)], None,
    )[0]
    return layout, pipe


def _alloc_set(ctx, set_layout, buffers):
    """Allocate a descriptor set and bind `buffers` to bindings 0..n-1."""
    pool = vk.vkCreateDescriptorPool(
        ctx.device,
        vk.VkDescriptorPoolCreateInfo(
            maxSets=1, poolSizeCount=1,
            pPoolSizes=[vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=len(buffers),
            )],
        ),
        None,
    )
    ds = vk.vkAllocateDescriptorSets(
        ctx.device,
        vk.VkDescriptorSetAllocateInfo(
            descriptorPool=pool, descriptorSetCount=1, pSetLayouts=[set_layout],
        ),
    )[0]
    writes = []
    for i, buf in enumerate(buffers):
        info = vk.VkDescriptorBufferInfo(buffer=buf.buffer, offset=0, range=buf.size)
        writes.append(vk.VkWriteDescriptorSet(
            dstSet=ds, dstBinding=i, descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[info],
        ))
    vk.vkUpdateDescriptorSets(ctx.device, len(writes), writes, 0, None)
    return pool, ds


class _SkinnedMeshGPU:
    """Per-skinned-mesh GPU resources + offsets into the shared buffers."""

    def __init__(self, ctx, binding, rest_vertex_bytes: bytes,
                 vertex_offset: int, node_offset: int, node_count: int,
                 index_offset: int):
        self.ctx = ctx
        self.binding = binding
        self.vertex_count = binding.rest_points.shape[0]
        self.joint_count = len(binding.skel_query.GetJointOrder())
        self.vertex_offset = int(vertex_offset)
        self.node_offset = int(node_offset)
        self.node_count = int(node_count)
        self.index_offset = int(index_offset)

        # Static buffers (built once).
        self.rest = StorageBuffer(ctx, len(rest_vertex_bytes))
        self.rest.upload_sync(rest_vertex_bytes)

        ji = np.zeros((self.vertex_count, 4), np.int32)
        jw = np.zeros((self.vertex_count, 4), np.float32)
        infl = min(binding.influences, 4)
        ji[:, :infl] = binding.joint_indices[:, :infl]
        jw[:, :infl] = binding.joint_weights[:, :infl]
        self.jidx = StorageBuffer(ctx, ji.nbytes)
        self.jidx.upload_sync(ji.tobytes())
        self.jwt = StorageBuffer(ctx, jw.nbytes)
        self.jwt.upload_sync(jw.tobytes())

        # Joint matrices (re-uploaded per frame): joint_count × 4×4 float32.
        self.jmats = StorageBuffer(ctx, self.joint_count * 64)

    def upload_joint_matrices(self, mats: np.ndarray) -> None:
        self.jmats.upload_sync(np.ascontiguousarray(mats, np.float32).tobytes())

    def destroy(self) -> None:
        for b in (self.rest, self.jidx, self.jwt, self.jmats):
            b.destroy()


class SkinningPasses:
    """Owns the skin + refit pipelines and per-mesh resources."""

    _SKIN_PUSH = "<II"        # vertexCount, outOffset
    _REFIT_PUSH = "<IIII"     # nodeCount, nodeOffset, indexOffset, vertexOffset

    def __init__(self, ctx, shader_dir: Path,
                 vertex_buffer, index_buffer, bvh_buffer):
        self.ctx = ctx
        self.vertex_buffer = vertex_buffer
        self.index_buffer = index_buffer
        self.bvh_buffer = bvh_buffer

        skin_spv = _compile(shader_dir, "skin", "skinMain")
        refit_spv = _compile(shader_dir, "bvh_refit", "refitMain")
        self._skin_mod = _shader_module(ctx, skin_spv)
        self._refit_mod = _shader_module(ctx, refit_spv)

        self._skin_set_layout = _ssbo_layout(ctx, 5)
        self._refit_set_layout = _ssbo_layout(ctx, 3)
        self._skin_layout, self._skin_pipe = _pipeline(
            ctx, self._skin_mod, self._skin_set_layout, 8)
        self._refit_layout, self._refit_pipe = _pipeline(
            ctx, self._refit_mod, self._refit_set_layout, 16)

        self._meshes: list[_SkinnedMeshGPU] = []
        self._pools: list = []
        self._skin_sets: list = []
        self._refit_sets: list = []

    @property
    def meshes(self) -> list[_SkinnedMeshGPU]:
        return self._meshes

    def prepare(self, mesh_gpus: list[_SkinnedMeshGPU]) -> None:
        """Allocate descriptor sets for the given per-mesh resources."""
        self._free_sets()
        self._meshes = mesh_gpus
        for m in mesh_gpus:
            pool_s, ds_s = _alloc_set(
                self.ctx, self._skin_set_layout,
                [m.rest, m.jidx, m.jwt, m.jmats, self.vertex_buffer],
            )
            pool_r, ds_r = _alloc_set(
                self.ctx, self._refit_set_layout,
                [self.vertex_buffer, self.index_buffer, self.bvh_buffer],
            )
            self._pools += [pool_s, pool_r]
            self._skin_sets.append(ds_s)
            self._refit_sets.append(ds_r)

    def dispatch(self, do_refit: bool = True) -> None:
        """Run skin (+ optional refit) for all prepared meshes, synchronously.

        Joint matrices must already be uploaded for this frame. Isolated
        one-shot submit; the subsequent render reads the updated buffers.
        """
        if not self._meshes:
            return
        ctx = self.ctx
        vk.vkDeviceWaitIdle(ctx.device)
        cmd = vk.vkAllocateCommandBuffers(ctx.device, vk.VkCommandBufferAllocateInfo(
            commandPool=ctx.command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ))[0]
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))

        # Skin: deform vertices into the shared vertex buffer.
        vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._skin_pipe)
        for m, ds in zip(self._meshes, self._skin_sets):
            vk.vkCmdBindDescriptorSets(
                cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._skin_layout,
                0, 1, [ds], 0, None)
            vk.vkCmdPushConstants(
                cmd, self._skin_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 8,
                _push_buf(struct.pack(self._SKIN_PUSH, m.vertex_count, m.vertex_offset)))
            groups = (m.vertex_count + _WORKGROUP - 1) // _WORKGROUP
            vk.vkCmdDispatch(cmd, groups, 1, 1)

        if do_refit:
            self._barrier(cmd, self.vertex_buffer)  # skin writes → refit reads
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._refit_pipe)
            for m, ds in zip(self._meshes, self._refit_sets):
                vk.vkCmdBindDescriptorSets(
                    cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._refit_layout,
                    0, 1, [ds], 0, None)
                vk.vkCmdPushConstants(
                    cmd, self._refit_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 16,
                    _push_buf(struct.pack(self._REFIT_PUSH, m.node_count, m.node_offset,
                                          m.index_offset, m.vertex_offset)))
                vk.vkCmdDispatch(cmd, 1, 1, 1)

        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(ctx.compute_queue, 1, [vk.VkSubmitInfo(
            commandBufferCount=1, pCommandBuffers=[cmd])], vk.VK_NULL_HANDLE)
        vk.vkQueueWaitIdle(ctx.compute_queue)
        vk.vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, [cmd])

    def _barrier(self, cmd, buf) -> None:
        b = vk.VkBufferMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=buf.buffer, offset=0, size=buf.size,
        )
        vk.vkCmdPipelineBarrier(
            cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, None, 1, [b], 0, None)

    def _free_sets(self) -> None:
        for pool in self._pools:
            vk.vkDestroyDescriptorPool(self.ctx.device, pool, None)
        self._pools = []
        self._skin_sets = []
        self._refit_sets = []

    def destroy(self) -> None:
        self._free_sets()
        for m in self._meshes:
            m.destroy()
        self._meshes = []
        vk.vkDestroyPipeline(self.ctx.device, self._skin_pipe, None)
        vk.vkDestroyPipeline(self.ctx.device, self._refit_pipe, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._skin_layout, None)
        vk.vkDestroyPipelineLayout(self.ctx.device, self._refit_layout, None)
