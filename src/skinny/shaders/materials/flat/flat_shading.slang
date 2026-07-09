// Shared flat-material (StandardSurface) helpers — Schlick fresnel,
// dielectric fresnel, MaterialX-graph dispatch, hit-data fetch. GGX
// direction sampling lives in shaders/samplers/ggx.slang; the colored
// weight is computed inside FlatMaterial.sample.

import common;
import bindings;
import mtlx_closures;
import mtlx_std_surface;
import generated_materials;

// `NonUniformResourceIndex` is a divergence hint for bindless indexing. Slang's
// Metal target rejects it in the compute stage (error E36107 — unavailable);
// it only annotates divergence, so dropping it on Metal is sound. `NRI(x)` is
// identity on the in-process Metal compile (which defines `SKINNY_METAL`) and
// `NonUniformResourceIndex(x)` on Vulkan. The Vulkan SPIR-V stays byte-identical
// because `SKINNY_METAL` is never defined on the Vulkan slangc path, so the
// preprocessor reproduces the original token stream exactly. (Preprocessor
// macros do not cross `import` boundaries, so this is defined per file.)
#if defined(SKINNY_METAL)
#define NRI(x) (x)
#else
#define NRI(x) NonUniformResourceIndex(x)
#endif

// Bindless flat-material texture sampling. On the Metal target the pool is a
// plain `Texture2D` array (binding 14) read through the shared `commonSampler`
// (binding 38) — see bindings.slang — so a sample passes the sampler
// explicitly. On Vulkan the pool is a combined `Sampler2D` array, so the macro
// expands to the exact prior `flatMaterialTextures[NRI(i)].SampleLevel(uv, lod)`
// token stream → byte-identical SPIR-V. (Macros do not cross `import`; defined
// per file.)
#if defined(SKINNY_METAL)
#define FLATTEX_SAMPLE_LEVEL(I, UV, LOD)    flatMaterialTextures[NRI(I)].SampleLevel(commonSampler, UV, LOD)
#define FLATTEX_SAMPLE_GRAD(I, UV, DX, DY)  flatMaterialTextures[NRI(I)].SampleGrad(commonSampler, UV, DX, DY)
#else
#define FLATTEX_SAMPLE_LEVEL(I, UV, LOD)    flatMaterialTextures[NRI(I)].SampleLevel(UV, LOD)
#define FLATTEX_SAMPLE_GRAD(I, UV, DX, DY)  flatMaterialTextures[NRI(I)].SampleGrad(UV, DX, DY)
#endif

float relativeLuminance(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 schlickF0(float3 albedo, float metallic, float specular, float ior)
{
    float r0 = (1.0 - ior) / (1.0 + ior);
    float dielectricF0 = r0 * r0 * specular;
    return lerp(float3(dielectricF0), albedo, metallic);
}

float3 schlickFresnelColor(float cosTheta, float3 F0)
{
    float c = clamp(1.0 - cosTheta, 0.0, 1.0);
    float c5 = c * c * c * c * c;
    return F0 + (float3(1.0) - F0) * c5;
}

float fresnelDielectric(float cosI, float eta)
{
    float sinT2 = eta * eta * (1.0 - cosI * cosI);
    if (sinT2 > 1.0)
        return 1.0;
    float cosT = sqrt(max(1.0 - sinT2, 0.0));
    float rS = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rP = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rS * rS + rP * rP);
}

#if defined(SKINNY_SPECTRAL)
// Complex arithmetic over float2 (re, im) for the conductor Fresnel below.
float2 cmul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float2 cdiv(float2 a, float2 b)
{
    float d = b.x * b.x + b.y * b.y;
    return float2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}
float  cnorm(float2 z) { return z.x * z.x + z.y * z.y; }
float2 csqrt(float2 z)
{
    float r = length(z);
    float2 res = float2(sqrt(max((r + z.x) * 0.5, 0.0)), sqrt(max((r - z.x) * 0.5, 0.0)));
    return (z.y < 0.0) ? float2(res.x, -res.y) : res;
}

// Unpolarized Fresnel reflectance of a conductor with complex index η+ik at
// incidence cosθ (pbrt v4 FrComplex, spectrum.cpp). Per single wavelength; the
// spectral path evaluates it at each hero λ with that λ's tabulated eta/k.
float fresnelConductor(float cosThetaI, float eta, float k)
{
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    float2 e    = float2(eta, k);
    float  sin2 = 1.0 - cosThetaI * cosThetaI;
    float2 sin2t = cdiv(float2(sin2, 0.0), cmul(e, e));
    float2 cost  = csqrt(float2(1.0, 0.0) - sin2t);
    float2 ci    = float2(cosThetaI, 0.0);
    float2 eci   = cmul(e, ci);
    float2 rParl = cdiv(eci - cost, eci + cost);
    float2 rPerp = cdiv(ci - cmul(e, cost), ci + cmul(e, cost));
    return 0.5 * (cnorm(rParl) + cnorm(rPerp));
}
#endif

struct FlatHitMat
{
    float3 albedo;
    float  roughness;
    float  metallic;
    float  specular;
    float  ior;
    float  opacity;
    float3 emission;
    float  coat;
    float  coatRoughness;
    float  coatIOR;
    float3 coatColor;
    // Stage-2 rich inputs. Defaults reproduce prior behavior:
    // transmissionColor = albedo, specularColor = white, diffuseRoughness = 0.
    float3 transmissionColor;
    float3 specularColor;
    float  diffuseRoughness;
};

struct FlatHitData
{
    FlatHitMat mat;
    float3 N;
};

// Pull one scalar channel out of an RGBA sample using the FlatMaterialParams
// channel-selector code (0=rgb→.r, 1=r, 2=g, 3=b, 4=a). The "rgb" code
// returns .r so a caller asking for "the scalar value" still gets the red
// channel — which matches what UsdPreviewSurface does when a float input is
// connected to a vec3 output without an explicit `outputs:r` selector.
float pickChannel(float4 sample, uint code)
{
    if (code == 4u) return sample.a;
    if (code == 3u) return sample.b;
    if (code == 2u) return sample.g;
    return sample.r;
}

// Extract a 4-bit channel selector for the given input from
// FlatMaterialParams.channelMask. Defaults defined in renderer.py map
// each input to its conventional output channel (.a for opacity, .r for
// roughness/metallic).
uint channelFor(uint mask, uint shift)
{
    return (mask >> shift) & 0xFu;
}

bool isCutoutTransparent(HitInfo h)
{
    uint matType = materialTypes[h.materialId] & 0xFFu;
    if (matType != MATERIAL_TYPE_FLAT) return false;
    FlatMaterialParams p = flatMaterials[h.materialId];
    if (p.opacityThreshold <= 0.0) return false;
    float alpha = 1.0;
    if (p.opacityTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.opacityTextureIdx, h.uv, 0.0);
        alpha = pickChannel(s, channelFor(p.channelMask, 12u));
    }
    return alpha < p.opacityThreshold;
}

FlatHitData fetchFlatHitData(HitInfo h)
{
    FlatHitData out_;
    if (h.materialId == 0u)
    {
        out_.mat.albedo       = float3(0.7);
        out_.mat.roughness    = 0.5;
        out_.mat.metallic     = 0.0;
        out_.mat.specular     = 0.5;
        out_.mat.ior          = 1.4;
        out_.mat.opacity      = 1.0;
        out_.mat.emission     = float3(0.0);
        out_.mat.coat         = 0.0;
        out_.mat.coatRoughness = 0.0;
        out_.mat.coatIOR      = 1.5;
        out_.mat.coatColor    = float3(1.0);
        out_.mat.transmissionColor = float3(0.7);   // = albedo fallback
        out_.mat.specularColor     = float3(1.0);
        out_.mat.diffuseRoughness  = 0.0;
        out_.N                = h.normal;
        return out_;
    }

    FlatMaterialParams p = flatMaterials[h.materialId];

    out_.mat.albedo = p.diffuseColor;
    if (p.diffuseTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.diffuseTextureIdx, h.uv, 0.0);
        uint cc = channelFor(p.channelMask, 0u);
        // For diffuse the rgb (code 0) path keeps the vector fetch;
        // explicit scalar channels broadcast to a grey triplet so
        // `outputs:r` returns (r,r,r).
        out_.mat.albedo = (cc == 0u) ? s.rgb : float3(pickChannel(s, cc));
    }

    out_.mat.roughness = clamp(p.roughness, 0.04, 1.0);
    if (p.roughnessTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.roughnessTextureIdx, h.uv, 0.0);
        float r = pickChannel(s, channelFor(p.channelMask, 4u));
        out_.mat.roughness = clamp(r, 0.04, 1.0);
    }

    out_.mat.metallic = clamp(p.metallic, 0.0, 1.0);
    if (p.metallicTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.metallicTextureIdx, h.uv, 0.0);
        float m = pickChannel(s, channelFor(p.channelMask, 8u));
        out_.mat.metallic = clamp(m, 0.0, 1.0);
    }

    out_.mat.specular = clamp(p.specular, 0.0, 1.0);
    out_.mat.ior = max(p.ior, 1.0);
    // Two opacity modes:
    //   Cutout (opacity texture + threshold > 0): the path/bdpt integrators
    //     already drop `alpha < threshold` hits via isCutoutTransparent, so
    //     any hit that reaches here has alpha >= threshold and the BSDF
    //     must see opacity = 1.0. Skip the texture fetch — it cannot
    //     influence the result.
    //   Alpha-blend (opacity texture, threshold == 0, OR constant opacity):
    //     sample the texture (or use the constant) and let FlatMaterial.sample
    //     route (1 - opacity) of evaluations through delta transmission /
    //     dielectric refraction.
    out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
    bool cutoutMode = (p.opacityTextureIdx != 0xFFFFFFFFu) && (p.opacityThreshold > 0.0);
    if (p.opacityTextureIdx != 0xFFFFFFFFu && !cutoutMode)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.opacityTextureIdx, h.uv, 0.0);
        out_.mat.opacity = pickChannel(s, channelFor(p.channelMask, 12u));
    }
    else if (cutoutMode)
    {
        out_.mat.opacity = 1.0;
    }
    out_.mat.coat = clamp(p.coat, 0.0, 1.0);
    out_.mat.coatRoughness = clamp(p.coatRoughness, 0.04, 1.0);
    out_.mat.coatIOR = max(p.coatIOR, 1.0);
    out_.mat.coatColor = p.coatColor;

    // Stage-2 rich inputs. The packer already resolves the fallbacks
    // (transmissionColor ← transmission_color else diffuseColor; specularColor
    // ← specular_color else white; diffuseRoughness ← diffuse_roughness else 0),
    // so the lobe model reads them directly.
    out_.mat.transmissionColor = p.transmissionColor;
    out_.mat.specularColor     = p.specularColor;
    out_.mat.diffuseRoughness  = clamp(p.diffuseRoughness, 0.0, 1.0);

    out_.mat.emission = p.emissiveColor;
    if (p.emissiveTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = FLATTEX_SAMPLE_LEVEL(p.emissiveTextureIdx, h.uv, 0.0);
        uint cc = channelFor(p.channelMask, 16u);
        float3 emTex = (cc == 0u) ? s.rgb : float3(pickChannel(s, cc));
        if (all(out_.mat.emission == float3(0.0)))
            out_.mat.emission = emTex;
        else
            out_.mat.emission *= emTex;
    }

    out_.N = h.normal;
    if (p.normalTextureIdx != 0xFFFFFFFFu && h.hasTangent)
    {
        // UsdUVTexture per-channel remap: tangentN = sample * scale + bias.
        // For OpenGL normal maps that's (2,2,2)*s + (-1,-1,-1) → [-1,1].
        // DirectX-style maps come in as scale.y=-2, bias.y=+1 which sign-
        // flips the green channel before reconstructing the tangent
        // vector. Z is rebuilt from XY so per-channel scale errors on B
        // don't leak into the surface normal direction.
        float3 sampled = FLATTEX_SAMPLE_LEVEL(p.normalTextureIdx, h.uv, 0.0).xyz;
        float3 tn = sampled * p.normalScale + p.normalBias;
        tn.z = sqrt(max(1.0 - dot(tn.xy, tn.xy), 0.0));
        float3 N = h.normal;
        float3 T = normalize(h.tangent - N * dot(h.tangent, N));
        float3 B = cross(N, T);
        out_.N = normalize(T * tn.x + B * tn.y + N * tn.z);
    }
    return out_;
}

StdSurfaceParams loadStdSurfaceParams(uint matId, float3 posObject,
                                       float3 normal, float3 tangent,
                                       float2 uv)
{
    StdSurfaceParams sp = stdSurfaceParams[matId];
    uint graphId = materialGraphId(matId);
    if (graphId >= 2u)
    {
        // Per-hit MaterialX nodegraph evaluator (gen-emitted via
        // MaterialLibrary.generate_for_compute). evalSceneGraph mutates
        // `sp` in place — each graph-driven std_surface input is
        // overwritten with the graph's per-pixel result, others keep
        // the SSBO-uploaded constant.
        evalSceneGraph(graphId, matId, posObject, normal, tangent, uv, sp);
    }
    return sp;
}

