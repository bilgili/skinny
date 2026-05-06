// Shared uniform and descriptor bindings consumed by all material evaluators.
// Declared here once so Slang sees a single definition across imports.

import common;
import mtlx_std_surface;

uniform FrameConstants fc;
uniform float3         lightDirection;
uniform float3         lightRadiance;

[[vk::binding(1)]]  RWTexture2D<float4> outputBuffer;
[[vk::binding(2)]]  RWTexture2D<float4> accumBuffer;
[[vk::binding(3)]]  [[vk::image_format("r8")]] RWTexture2D<float> hudMask;
[[vk::binding(13)]] StructuredBuffer<FlatMaterialParams> flatMaterials;
[[vk::binding(14)]] Sampler2D<float4> flatMaterialTextures[16];
[[vk::binding(16)]] StructuredBuffer<uint> materialTypes;
[[vk::binding(19)]] StructuredBuffer<StdSurfaceParams> stdSurfaceParams;
[[vk::binding(20)]] StructuredBuffer<ProceduralParams> proceduralParams;

static const uint MATERIAL_TYPE_SKIN         = 0u;
static const uint MATERIAL_TYPE_FLAT         = 1u;
static const uint MATERIAL_TYPE_DEBUG_NORMAL = 2u;

static const uint MATERIAL_FURNACE_BIT = 10u;

uint effectiveFurnaceMode(uint materialId)
{
    if (fc.furnaceMode != 0u) return 1u;
    return (materialTypes[materialId] >> MATERIAL_FURNACE_BIT) & 1u;
}
