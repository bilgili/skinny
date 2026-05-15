"""Vulkan ↔ gfx enum mapping helpers and shared utilities."""

from __future__ import annotations

import vulkan as vk

from skinny.gfx.types import (
    AddressMode,
    BindingKind,
    BufferUsage,
    FilterMode,
    Format,
    ImageState,
    ImageUsage,
    PipelineStage,
    ShaderStage,
)


# ── Format ──────────────────────────────────────────────────────────

_FORMAT_TO_VK: dict[Format, int] = {
    Format.UNDEFINED: vk.VK_FORMAT_UNDEFINED,
    Format.R8G8B8A8_UNORM: vk.VK_FORMAT_R8G8B8A8_UNORM,
    Format.R8G8B8A8_SRGB: vk.VK_FORMAT_R8G8B8A8_SRGB,
    Format.B8G8R8A8_UNORM: vk.VK_FORMAT_B8G8R8A8_UNORM,
    Format.B8G8R8A8_SRGB: vk.VK_FORMAT_B8G8R8A8_SRGB,
    Format.R16G16B16A16_SFLOAT: vk.VK_FORMAT_R16G16B16A16_SFLOAT,
    Format.R32G32B32A32_SFLOAT: vk.VK_FORMAT_R32G32B32A32_SFLOAT,
    Format.R32_UINT: vk.VK_FORMAT_R32_UINT,
    Format.R32_SFLOAT: vk.VK_FORMAT_R32_SFLOAT,
    Format.D32_SFLOAT: vk.VK_FORMAT_D32_SFLOAT,
}


def vk_format(fmt: Format) -> int:
    return _FORMAT_TO_VK[fmt]


def format_bytes(fmt: Format) -> int:
    return {
        Format.R8G8B8A8_UNORM: 4,
        Format.R8G8B8A8_SRGB: 4,
        Format.B8G8R8A8_UNORM: 4,
        Format.B8G8R8A8_SRGB: 4,
        Format.R16G16B16A16_SFLOAT: 8,
        Format.R32G32B32A32_SFLOAT: 16,
        Format.R32_UINT: 4,
        Format.R32_SFLOAT: 4,
        Format.D32_SFLOAT: 4,
    }[fmt]


# ── Buffer / Image usage ────────────────────────────────────────────

def vk_buffer_usage(usage: BufferUsage) -> int:
    flags = 0
    if usage & BufferUsage.UNIFORM:
        flags |= vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    if usage & BufferUsage.STORAGE:
        flags |= vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    if usage & BufferUsage.INDEX:
        flags |= vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    if usage & BufferUsage.VERTEX:
        flags |= vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
    if usage & BufferUsage.INDIRECT:
        flags |= vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
    if usage & BufferUsage.TRANSFER_SRC:
        flags |= vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    if usage & BufferUsage.TRANSFER_DST:
        flags |= vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    return flags


def vk_image_usage(usage: ImageUsage) -> int:
    flags = 0
    if usage & ImageUsage.SAMPLED:
        flags |= vk.VK_IMAGE_USAGE_SAMPLED_BIT
    if usage & ImageUsage.STORAGE:
        flags |= vk.VK_IMAGE_USAGE_STORAGE_BIT
    if usage & ImageUsage.COLOR_ATTACHMENT:
        flags |= vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    if usage & ImageUsage.DEPTH_ATTACHMENT:
        flags |= vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    if usage & ImageUsage.TRANSFER_SRC:
        flags |= vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    if usage & ImageUsage.TRANSFER_DST:
        flags |= vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
    return flags


# ── Image state ─────────────────────────────────────────────────────

def vk_image_layout(state: ImageState) -> int:
    return {
        ImageState.UNDEFINED: vk.VK_IMAGE_LAYOUT_UNDEFINED,
        ImageState.GENERAL: vk.VK_IMAGE_LAYOUT_GENERAL,
        ImageState.SHADER_READ: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        ImageState.SHADER_WRITE: vk.VK_IMAGE_LAYOUT_GENERAL,
        ImageState.TRANSFER_SRC: vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        ImageState.TRANSFER_DST: vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        ImageState.COLOR_ATTACHMENT: vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ImageState.DEPTH_ATTACHMENT: vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ImageState.PRESENT: vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    }[state]


def vk_access_for_state(state: ImageState) -> int:
    return {
        ImageState.UNDEFINED: 0,
        ImageState.GENERAL: vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        ImageState.SHADER_READ: vk.VK_ACCESS_SHADER_READ_BIT,
        ImageState.SHADER_WRITE: vk.VK_ACCESS_SHADER_WRITE_BIT,
        ImageState.TRANSFER_SRC: vk.VK_ACCESS_TRANSFER_READ_BIT,
        ImageState.TRANSFER_DST: vk.VK_ACCESS_TRANSFER_WRITE_BIT,
        ImageState.COLOR_ATTACHMENT: vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        ImageState.DEPTH_ATTACHMENT: vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        ImageState.PRESENT: 0,
    }[state]


# ── Pipeline stage ──────────────────────────────────────────────────

def vk_pipeline_stage(stage: PipelineStage) -> int:
    flags = 0
    if stage & PipelineStage.COMPUTE_SHADER:
        flags |= vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    if stage & PipelineStage.VERTEX_SHADER:
        flags |= vk.VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
    if stage & PipelineStage.FRAGMENT_SHADER:
        flags |= vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
    if stage & PipelineStage.TRANSFER:
        flags |= vk.VK_PIPELINE_STAGE_TRANSFER_BIT
    if stage & PipelineStage.COLOR_ATTACHMENT_OUTPUT:
        flags |= vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    if stage & PipelineStage.TOP:
        flags |= vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
    if stage & PipelineStage.BOTTOM:
        flags |= vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
    return flags or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT


# ── Shader stage ────────────────────────────────────────────────────

def vk_shader_stage(stage: ShaderStage) -> int:
    flags = 0
    if stage & ShaderStage.VERTEX:
        flags |= vk.VK_SHADER_STAGE_VERTEX_BIT
    if stage & ShaderStage.FRAGMENT:
        flags |= vk.VK_SHADER_STAGE_FRAGMENT_BIT
    if stage & ShaderStage.COMPUTE:
        flags |= vk.VK_SHADER_STAGE_COMPUTE_BIT
    return flags


# ── Descriptor binding kind ─────────────────────────────────────────

def vk_descriptor_type(kind: BindingKind) -> int:
    return {
        BindingKind.UNIFORM_BUFFER: vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        BindingKind.STORAGE_BUFFER: vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        BindingKind.STORAGE_IMAGE: vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        BindingKind.SAMPLED_IMAGE: vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        BindingKind.COMBINED_IMAGE_SAMPLER: vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        BindingKind.SAMPLER: vk.VK_DESCRIPTOR_TYPE_SAMPLER,
    }[kind]


# ── Sampler ─────────────────────────────────────────────────────────

def vk_filter(f: FilterMode) -> int:
    return vk.VK_FILTER_NEAREST if f == FilterMode.NEAREST else vk.VK_FILTER_LINEAR


def vk_mipmap_mode(f: FilterMode) -> int:
    return (
        vk.VK_SAMPLER_MIPMAP_MODE_NEAREST if f == FilterMode.NEAREST
        else vk.VK_SAMPLER_MIPMAP_MODE_LINEAR
    )


def vk_address_mode(a: AddressMode) -> int:
    return {
        AddressMode.REPEAT: vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
        AddressMode.CLAMP_TO_EDGE: vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        AddressMode.CLAMP_TO_BORDER: vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        AddressMode.MIRRORED_REPEAT: vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    }[a]


# ── Memory ──────────────────────────────────────────────────────────

def find_memory_type(
    physical_device,
    type_filter: int,
    properties: int,
) -> int:
    """Pick the first memory type satisfying both the resource's typeBits
    mask and the requested property flags. Raises if none found."""
    mem_props = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    for i in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << i)) and (
            mem_props.memoryTypes[i].propertyFlags & properties
        ) == properties:
            return i
    raise RuntimeError("No suitable Vulkan memory type for the requested properties")
