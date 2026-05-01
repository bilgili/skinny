"""Vulkan compute pipeline — compiles Slang shaders to SPIR-V and manages dispatch."""

from __future__ import annotations

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
BINDLESS_TEXTURE_CAPACITY = 16


class ComputePipeline:
    """Wraps a single Vulkan compute pipeline compiled from a Slang entry point."""

    def __init__(
        self,
        ctx: VulkanContext,
        shader_dir: Path,
        entry_module: str,
        entry_point: str,
    ) -> None:
        self.ctx = ctx
        self.shader_dir = shader_dir
        self.entry_module = entry_module
        self.entry_point = entry_point

        self._spirv_path = self._compile_slang()
        self._shader_module = self._create_shader_module()
        self.descriptor_set_layout = self._create_descriptor_set_layout()
        self.pipeline_layout = self._create_pipeline_layout()
        self.pipeline = self._create_pipeline()

    # ── Slang → SPIR-V compilation ───────────────────────────────

    def _compile_slang(self) -> Path:
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
        cmd = [
            slangc,
            str(src),
            "-target", "spirv",
            "-entry", self.entry_point,
            "-stage", "compute",
            "-o", str(out),
            "-I", str(self.shader_dir),
            "-I", str(mtlx_genslang),
            # Tells the genslang impls to omit gen-prelude-only paths
            # (e.g. mx_environment_irradiance) so they compile standalone
            # in skinny's compute pipeline. The MaterialX gen path doesn't
            # set this and keeps the gen-provided helpers.
            "-D", "SKINNY_COMPUTE_PIPELINE=1",
            "-fvk-use-scalar-layout",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Slang compilation failed:\n{result.stderr}")
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
            # record per material slot. Slot 0 = legacy SDF-head skin;
            # slots > 0 = USD-bound skin-typed materials. Layout matches
            # the gen-reflected M_skinny_skin_default uniform_block.
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
            # binding 20: Per-material procedural evaluation params. 96 B
            # each (ProceduralParams struct, scalar layout). type == 0
            # means no procedural eval; type == 1 is 3D marble noise.
            vk.VkDescriptorSetLayoutBinding(
                binding=20,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]

        # Per-binding flags — only binding 14 needs PARTIALLY_BOUND. Vulkan
        # requires the array length to match `bindings`, so all other
        # bindings get a zero flag.
        binding_flags = [0] * len(bindings)
        # Find binding 14's index (it's not always second-to-last after
        # binding 15 was added) and set the PARTIALLY_BOUND flag there.
        for i, b in enumerate(bindings):
            if b.binding == 14:
                binding_flags[i] = vk.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                break
        flags_info = vk.VkDescriptorSetLayoutBindingFlagsCreateInfo(
            bindingCount=len(binding_flags),
            pBindingFlags=binding_flags,
        )
        layout_info = vk.VkDescriptorSetLayoutCreateInfo(
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
    ) -> None:
        self.ctx = ctx
        self.width = width
        self.height = height
        self.format = format

        img_info = vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=format,
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

    def destroy(self) -> None:
        vk.vkDestroyBuffer(self.ctx.device, self.buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.memory, None)
        vk.vkDestroyBuffer(self.ctx.device, self.staging_buffer, None)
        vk.vkFreeMemory(self.ctx.device, self.staging_memory, None)
