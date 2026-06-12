// Shared uniform and descriptor bindings consumed by all material evaluators.
// Declared here once so Slang sees a single definition across imports.

import common;
import mtlx_std_surface;

uniform FrameConstants fc;

// Bindless flat-material texture pool (binding 14). On Vulkan it is a combined
// Sampler2D[128] (one UPDATE_AFTER_BIND combined-image-sampler array). Native
// Metal caps a compute kernel at 16 sampler arguments and 128 texture
// arguments; a combined Sampler2D[128] lowers to array<texture2d,128> +
// array<sampler,128>, blowing past both limits once the discrete maps are
// added. So on the Metal target the pool is a plain Texture2D array sampled
// through one shared `commonSampler` (binding 38) — collapsing 128 samplers to
// 1 — and its capacity is trimmed to leave room for the ~8 discrete textures
// inside Metal's 128-texture budget. The Vulkan declaration (and its SPIR-V) is
// unchanged. See FLATTEX_SAMPLE_LEVEL/GRAD in flat_shading.slang & mtlx_gen_shim.slang.
//
// The Metal declaration MUST come before every other texture-typed global
// (change metal-wavefront-parity, task 3.3): Slang's Metal backend emits the
// texture ARRAY parameter without an explicit [[texture(N)]] attribute, so
// Metal auto-assigns its slots starting at 0 — while scalar texture globals
// get explicit indices from the program-wide reflected order. If any texture
// global precedes the array in that order but is dead-stripped from a kernel
// (e.g. outputBuffer/accumBuffer/hudMask in the wavefront shade kernels), the
// array's actual slots shift below its reflected slots and every bindless
// sample aliases the wrong texture (the wood-renders-gray bug: array slot 0
// read outputBuffer). Declaring the array first pins reflected base == Metal
// auto-assigned base == 0 in every kernel, dead-strip or not. The megakernel
// keeps every preceding texture live, which is why it never hit this.
#if defined(SKINNY_METAL)
[[vk::binding(14)]] Texture2D<float4> flatMaterialTextures[120];
[[vk::binding(38)]] SamplerState commonSampler;
#endif

[[vk::binding(20)]] StructuredBuffer<DistantLight> distantLights;

[[vk::binding(1)]]  RWTexture2D<float4> outputBuffer;
[[vk::binding(2)]]  RWTexture2D<float4> accumBuffer;
[[vk::binding(3)]]  [[vk::image_format("r8")]] RWTexture2D<float> hudMask;
[[vk::binding(13)]] StructuredBuffer<FlatMaterialParams> flatMaterials;
#if !defined(SKINNY_METAL)
[[vk::binding(14)]] Sampler2D<float4> flatMaterialTextures[128];
#endif
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
// Metal slot-cap gate (change metal-wavefront-parity phase 6): the wavefront
// program with the neural proposal un-stubbed (SKINNY_METAL_NEURAL) sits
// exactly at Metal's 31-buffer-slot argument table, so the three buffer
// globals that are dead in every wavefront kernel — toolBuffer here plus
// recordBuf/recordCounter (integrators/path_record_common.slang) — are
// compiled out of that build to make room for neuralWeights/Biases/Layers.
// The megakernel (SKINNY_METAL only) and Vulkan (no SKINNY_METAL) builds are
// unaffected; no wavefront kernel references the tool dispatch.
#if !(defined(SKINNY_METAL) && defined(SKINNY_METAL_NEURAL))
[[vk::binding(30)]] RWStructuredBuffer<float4>     toolBuffer;
#endif

static const uint TOOL_MODE_OFF        = 0u;
static const uint TOOL_MODE_PICK       = 1u;  // pick path uses fc.pickArmed, not toolMode
static const uint TOOL_MODE_BXDF       = 2u;
static const uint TOOL_MODE_BSSRDF     = 3u;
static const uint TOOL_MODE_STRUCTURAL = 4u;  // per-pixel structural AOV → toolBuffer

// Structural-AOV (TOOL_MODE_STRUCTURAL) writes one float4 per pixel —
// (hit-mask, instanceId, materialId, depth) — into toolBuffer starting at this
// slot, for the Metal↔Vulkan structural-parity test (6.1). Past the 16-slot
// header/pick region; overlaps the BXDF grid output, but the tool modes are
// mutually exclusive (only one armed per frame). Mirrors renderer.TOOL_STRUCT_AOV_BASE.
static const uint TOOL_STRUCT_AOV_BASE = 16u;

static const uint MATERIAL_TYPE_SKIN         = 0u;
static const uint MATERIAL_TYPE_FLAT         = 1u;
static const uint MATERIAL_TYPE_DEBUG_NORMAL = 2u;
static const uint MATERIAL_TYPE_PYTHON       = 3u;

static const uint MATERIAL_FURNACE_BIT = 10u;

// MaterialX graph id lives in the upper byte of materialTypes[matId].
// 0 ⇒ no graph (use the legacy flat / std_surface SSBO path). 2..255 ⇒
// graph index assigned by materialx_runtime.assign_graph_ids; matches
// `case Nu:` in generated_materials.slang::evalSceneGraph.
static const uint MATERIAL_GRAPH_SHIFT = 16u;
static const uint MATERIAL_GRAPH_MASK  = 0xffu;

// Python-material id (slangpile `python_materials/*.py` → `IMaterial`
// struct). 0..255 ⇒ index into the dispatcher switch emitted by
// `vk_compute._emit_python_dispatcher`. Only consulted when material
// type == MATERIAL_TYPE_PYTHON.
static const uint MATERIAL_PYMAT_SHIFT = 24u;
static const uint MATERIAL_PYMAT_MASK  = 0xffu;

uint materialGraphId(uint matId)
{
    return (materialTypes[matId] >> MATERIAL_GRAPH_SHIFT) & MATERIAL_GRAPH_MASK;
}

uint pythonMaterialId(uint matId)
{
    return (materialTypes[matId] >> MATERIAL_PYMAT_SHIFT) & MATERIAL_PYMAT_MASK;
}

uint effectiveFurnaceMode(uint materialId)
{
    if (fc.furnaceMode != 0u) return 1u;
    return (materialTypes[materialId] >> MATERIAL_FURNACE_BIT) & 1u;
}
