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

import shutil
import subprocess
from pathlib import Path

import vulkan as vk

from skinny.vk_compute import StorageBuffer
from skinny.wavefront_layout import queue_buffer_sizes


def _compile_spv(shader_dir: Path, module: str, entry: str) -> Path:
    """Compile shaders/<module>.slang → .spv (scalar layout). Mirrors
    vk_skinning._compile so the wavefront stage kernels build the same way."""
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
