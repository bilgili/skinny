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
// Pool trimmed 120 -> 119 (nanovdb-volume-rendering): the Metal compute
// argument table caps at 128 textures, and 120 (pool) + 5 discrete maps +
// output/accum/hudMask filled it EXACTLY — adding the `volumeDensity` 3D
// grid at 129 made every Metal pipeline fail silently (all-black frames).
// MUST equal metal_compute.BINDLESS_TEXTURE_CAPACITY.
[[vk::binding(14)]] Texture2D<float4> flatMaterialTextures[119];
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
// Metal records-build slot-cap gate (change metal-record-drain): the
// neural wavefront program sits exactly at Metal's 31-buffer-slot argument
// table, so the records build (SKINNY_METAL_RECORDS) frees the two slots the
// record stream needs by compiling out the two globals that are inert in a
// records render: the splat buffer (records are path-integrator-only, where
// the splat is all zeros) and the gizmo overlay segments (no gizmo overlay on
// training frames — documented). wf_display.slang guards their reads with the
// same condition. The bdpt, megakernel, and Vulkan builds are unaffected.
#if !(defined(SKINNY_METAL) && defined(SKINNY_METAL_RECORDS))
[[vk::binding(21)]] RWStructuredBuffer<uint> lightSplatBuffer;
[[vk::binding(22)]] StructuredBuffer<GizmoSegment> gizmoSegments;
#endif
[[vk::binding(23)]] StructuredBuffer<float4>       lensElements;
[[vk::binding(24)]] StructuredBuffer<float4>       exitPupilBounds;
// Heterogeneous-medium density grid (nanovdb-volume-rendering, design D3): ONE
// R16F 3D texture, normalized to [0,1] on the CPU (divide by the grid's value
// max — folded into the packed σ so the texel IS the density multiplier).
// Sampled by densityAt's MEDIUM_NANOVDB case through the folded world→uvw
// affine in FlatMaterialParams. ALWAYS bound: the renderer seeds a 1×1×1 zero
// fallback so the descriptor is valid with no volume in the scene. Binding 26
// is the next free Vulkan slot (25 = graphParamsCombined, 30/31/33..37 taken);
// on Metal the combined Sampler3D splits into Texture3D + SamplerState like the
// discrete 2D maps (design D8), with the sampler at 44 (after 39–43). This is a
// texture, not a buffer — Metal's 31-slot buffer argument table is unaffected.
#if defined(SKINNY_METAL)
[[vk::binding(26)]] Texture3D<float> volumeDensity;
[[vk::binding(44)]] SamplerState volumeDensitySampler;
#else
[[vk::binding(26)]] Sampler3D<float> volumeDensity;
#endif

// ── Hero-wavelength spectral upsampling tables (change spectral-rendering) ──
// Bound ONLY in the spectral megakernel variant (-DSKINNY_SPECTRAL); absent from
// the RGB build so its descriptor layout AND SPIR-V stay byte-identical. Fed to
// spectrum.slang's rgbSigmoidCoeffs / upsampleReflectance / upsampleIlluminant.
//   spectralScale : RES node array               (res floats;      binding 45)
//   spectralData  : flat [3][res][res][res][3]   sigmoid-coeff grid (binding 46)
//   spectralD65   : CIE D65 SPD on 360–830/5 nm  (d65Count floats;  binding 47)
// The vendored table resolution + D65 length are fixed, so they ride as compile
// constants rather than FrameConstants fields — no std140 change to the shared
// UBO, so the RGB build's UBO packing is untouched. bindings 45–47 are the next
// free Vulkan slots above the Metal sampler range (38–44); on Metal they bind by
// name (spectralScale/Data/D65), so the vk::binding index is inert there.
#if defined(SKINNY_SPECTRAL)
import spectrum;
static const int SPECTRAL_TABLE_RES = 64;   // rgb2spec_srgb.npz `res`
static const int SPECTRAL_D65_COUNT = 95;   // spectral_curves.npz d65 grid length
[[vk::binding(45)]] StructuredBuffer<float> spectralScale;
[[vk::binding(46)]] StructuredBuffer<float> spectralData;
[[vk::binding(47)]] StructuredBuffer<float> spectralD65;

// Bound-buffer wrappers around spectrum.slang's parameter-driven upsamplers, so
// call sites in the transport stay short. Reflectance: sigmoid only. Illuminant:
// sigmoid × D65 (pbrt RGBIlluminantSpectrum).
Spectrum upsampleReflectanceBound(float3 rgb, SampledWavelengths sw)
{
    return upsampleReflectance(rgb, sw, SPECTRAL_TABLE_RES, spectralScale, spectralData);
}
Spectrum upsampleIlluminantBound(float3 rgb, SampledWavelengths sw)
{
    return upsampleIlluminant(rgb, sw, SPECTRAL_TABLE_RES, spectralScale,
                              spectralData, spectralD65, SPECTRAL_D65_COUNT);
}

// Named-conductor eta/k curves (change spectral-rendering, Group 6.2). vk binding
// 48; uploaded only when spectral. Layout: 4 metals (au, ag, al, cu — ids 1..4)
// each [eta(95) | k(95)] on the 360–830/5 nm grid (95 = SPECTRAL_D65_COUNT), so
// stride 190 floats/metal. Absent from the RGB build.
[[vk::binding(48)]] StructuredBuffer<float> spectralMetals;
static const int SPECTRAL_METAL_STRIDE = 2 * SPECTRAL_D65_COUNT;  // eta+k per metal

// eta(λ)/k(λ) at the 4 hero wavelengths for named metal id 1..4 (linear interp on
// the 5 nm grid, endpoint-clamped — mirrors sampleCurve5nm with a base offset).
void namedMetalEtaK(uint metalId, SampledWavelengths sw, out float4 eta, out float4 k)
{
    int base  = int(metalId - 1u) * SPECTRAL_METAL_STRIDE;
    int kbase = base + SPECTRAL_D65_COUNT;
    [ForceUnroll]
    for (int i = 0; i < 4; ++i)
    {
        float t = (sw.lambda[i] - SPECTRUM_LAMBDA_MIN) / 5.0;
        int   j = clamp(int(t), 0, SPECTRAL_D65_COUNT - 2);
        float f = clamp(t - float(j), 0.0, 1.0);
        eta[i] = lerp(spectralMetals[base + j],  spectralMetals[base + j + 1],  f);
        k[i]   = lerp(spectralMetals[kbase + j], spectralMetals[kbase + j + 1], f);
    }
}
#endif
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
// Volumetric subsurface (pbrt-subsurface-volumetric): a dielectric boundary + a
// homogeneous interior medium, transported by an interior random walk. The medium
// coefficients are packed into FlatMaterialParams (binding 13) — no new buffer, to
// respect the Metal 31-buffer cap (see design.md) — and read via resolveMedium.
static const uint MATERIAL_TYPE_SUBSURFACE   = 4u;
// Free-standing participating medium (nanovdb-volume-rendering): a pbrt
// `Material "interface"` boundary shape with a MediumInterface. Index-matched
// pass-through boundary (no Fresnel, no refraction, no BSDF) + the medium walk
// (materials/subsurface/volume_walk.slang). Medium coefficients + the world→uvw
// grid transform ride inline in FlatMaterialParams (binding 13), the density
// field in the volumeDensity texture (binding 26).
static const uint MATERIAL_TYPE_VOLUME       = 5u;

// Medium source kinds — the density-seam dispatch tag. HOMOGENEOUS: densityAt ≡ 1,
// majorant = σ_t (the dielectric-bounded subsurface interior AND a homogeneous
// free-standing interior). NANOVDB: densityAt = trilinear sample of the
// normalized density grid (binding 26); the global majorant is still the packed
// σ_t because the grid's value max is folded into σ on the CPU. CLOUD: pbrt's
// procedural CloudMedium — analytic Perlin-fBm density (cloud_noise.slang), no
// texture; density clamps to [0,1] so the packed σ_t is again the exact
// global majorant.
static const uint MEDIUM_HOMOGENEOUS = 0u;
static const uint MEDIUM_NANOVDB     = 1u;
static const uint MEDIUM_CLOUD       = 2u;

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
