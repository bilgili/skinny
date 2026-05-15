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
[[vk::binding(14)]] Sampler2D<float4> flatMaterialTextures[128];
[[vk::binding(16)]] StructuredBuffer<uint> materialTypes;
[[vk::binding(19)]] StructuredBuffer<StdSurfaceParams> stdSurfaceParams;
// BDPT light-tracer splat buffer: per-pixel R, G, B as fixed-point uint
// (Q22.10). Indexed as [pixel.y * width * 3 + pixel.x * 3 + channel].
[[vk::binding(21)]] RWStructuredBuffer<uint> lightSplatBuffer;
[[vk::binding(22)]] StructuredBuffer<GizmoSegment> gizmoSegments;
[[vk::binding(23)]] StructuredBuffer<float4>       lensElements;
[[vk::binding(24)]] StructuredBuffer<float4>       exitPupilBounds;
// BXDF visualizer output. Host-visible RWStructuredBuffer<float4>. Layout:
//   [0..7]  : tool header (input, written by CPU before each eval)
//             [0] = (toolMode, lockMode, nTheta, nPhi) as uint bits
//             [1] = (matId, _, _, _)               as uint bits
//             [2] = (dLocked.xyz, _)               tangent-space locked dir
//             [3] = (P.xyz, _)                     picked world position
//             [4] = (N.xyz, _)                     world normal
//             [5] = (T.xyz, _)                     world tangent
//             [6] = (UV.xy, _, _)                  texture UV
//             [7] = (xi.xyz, _)                    BSSRDF entrance world pos
//   [8..11] : pick HitInfo output
//             [8]  = (position.xyz, t)
//             [9]  = (normal.xyz, asfloat(materialId))
//             [10] = (tangent.xyz, hitFlag)
//             [11] = (uv.xy, hasTangent, _)
//   [16..N] : BXDF eval grid output (n_theta × n_phi float4 entries)
//             slot index = 16 + j*n_theta + i  where (i, j) = (theta, phi)
[[vk::binding(30)]] RWStructuredBuffer<float4>     toolBuffer;

static const uint TOOL_MODE_OFF    = 0u;
static const uint TOOL_MODE_PICK   = 1u;  // pick path uses fc.pickArmed, not toolMode
static const uint TOOL_MODE_BXDF   = 2u;
static const uint TOOL_MODE_BSSRDF = 3u;

static const uint MATERIAL_TYPE_SKIN         = 0u;
static const uint MATERIAL_TYPE_FLAT         = 1u;
static const uint MATERIAL_TYPE_DEBUG_NORMAL = 2u;

static const uint MATERIAL_FURNACE_BIT = 10u;

// MaterialX graph id lives in the upper byte of materialTypes[matId].
// 0 ⇒ no graph (use the legacy flat / std_surface SSBO path). 2..255 ⇒
// graph index assigned by materialx_runtime.assign_graph_ids; matches
// `case Nu:` in generated_materials.slang::evalSceneGraph.
static const uint MATERIAL_GRAPH_SHIFT = 16u;
static const uint MATERIAL_GRAPH_MASK  = 0xffu;

uint materialGraphId(uint matId)
{
    return (materialTypes[matId] >> MATERIAL_GRAPH_SHIFT) & MATERIAL_GRAPH_MASK;
}

uint effectiveFurnaceMode(uint materialId)
{
    if (fc.furnaceMode != 0u) return 1u;
    return (materialTypes[materialId] >> MATERIAL_FURNACE_BIT) & 1u;
}
