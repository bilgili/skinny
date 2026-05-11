// Shared flat-material (StandardSurface) helpers — Schlick fresnel,
// dielectric fresnel, procedural color, hit-data fetch. GGX direction
// sampling lives in shaders/samplers/ggx.slang; the colored weight is
// computed inside FlatMaterial.sample.

import common;
import bindings;
import mtlx_closures;
import mtlx_std_surface;
import mtlx_noise;

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
};

struct FlatHitData
{
    FlatHitMat mat;
    float3 N;
};

bool isCutoutTransparent(HitInfo h)
{
    uint matType = materialTypes[h.materialId] & 0xFFu;
    if (matType != MATERIAL_TYPE_FLAT) return false;
    FlatMaterialParams p = flatMaterials[h.materialId];
    if (p.opacityThreshold <= 0.0) return false;
    float alpha = 1.0;
    if (p.opacityTextureIdx != 0xFFFFFFFFu)
        alpha = flatMaterialTextures[NonUniformResourceIndex(p.opacityTextureIdx)]
                    .SampleLevel(h.uv, 0.0).a;
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
        out_.N                = h.normal;
        return out_;
    }

    FlatMaterialParams p = flatMaterials[h.materialId];

    out_.mat.albedo = p.diffuseColor;
    if (p.diffuseTextureIdx != 0xFFFFFFFFu)
    {
        out_.mat.albedo = flatMaterialTextures[NonUniformResourceIndex(p.diffuseTextureIdx)]
                              .SampleLevel(h.uv, 0.0).rgb;
    }

    out_.mat.roughness = clamp(p.roughness, 0.04, 1.0);
    if (p.roughnessTextureIdx != 0xFFFFFFFFu)
    {
        float r = flatMaterialTextures[NonUniformResourceIndex(p.roughnessTextureIdx)]
                      .SampleLevel(h.uv, 0.0).r;
        out_.mat.roughness = clamp(r, 0.04, 1.0);
    }

    out_.mat.metallic = clamp(p.metallic, 0.0, 1.0);
    if (p.metallicTextureIdx != 0xFFFFFFFFu)
    {
        float m = flatMaterialTextures[NonUniformResourceIndex(p.metallicTextureIdx)]
                      .SampleLevel(h.uv, 0.0).r;
        out_.mat.metallic = clamp(m, 0.0, 1.0);
    }

    out_.mat.specular = clamp(p.specular, 0.0, 1.0);
    out_.mat.ior = max(p.ior, 1.0);
    out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
    if (p.opacityTextureIdx != 0xFFFFFFFFu)
    {
        float a = flatMaterialTextures[NonUniformResourceIndex(p.opacityTextureIdx)]
                      .SampleLevel(h.uv, 0.0).a;
        out_.mat.opacity = a;
    }
    if (p.opacityThreshold > 0.0 && out_.mat.opacity < p.opacityThreshold)
    {
        out_.mat.opacity = 0.0;
    }
    out_.mat.coat = clamp(p.coat, 0.0, 1.0);
    out_.mat.coatRoughness = clamp(p.coatRoughness, 0.04, 1.0);
    out_.mat.coatIOR = max(p.coatIOR, 1.0);
    out_.mat.coatColor = p.coatColor;

    out_.mat.emission = p.emissiveColor;
    if (p.emissiveTextureIdx != 0xFFFFFFFFu)
    {
        float3 emTex = flatMaterialTextures[NonUniformResourceIndex(p.emissiveTextureIdx)]
                           .SampleLevel(h.uv, 0.0).rgb;
        if (all(out_.mat.emission == float3(0.0)))
            out_.mat.emission = emTex;
        else
            out_.mat.emission *= emTex;
    }

    out_.N = h.normal;
    if (p.normalTextureIdx != 0xFFFFFFFFu && h.hasTangent)
    {
        float3 sampled = flatMaterialTextures[NonUniformResourceIndex(p.normalTextureIdx)]
                            .SampleLevel(h.uv, 0.0).xyz * 2.0 - 1.0;
        float3 N = h.normal;
        float3 T = normalize(h.tangent - N * dot(h.tangent, N));
        float3 B = cross(N, T);
        out_.N = normalize(T * sampled.x + B * sampled.y + N * sampled.z);
    }
    return out_;
}

float3 evalProceduralColor(ProceduralParams pp, float3 posObject)
{
    if (pp.type == PROC_TYPE_MARBLE_3D)
    {
        float add_xyz_out = dot(posObject, pp.add_xyz_in2);
        float3 scale_pos_out = posObject * pp.scale_pos_in2;
        float scale_xyz_out = add_xyz_out * pp.scale_xyz_in2;
        float noise_out = 0.0;
        mx_fractal3d_float(pp.noise_amplitude, pp.noise_octaves,
                           pp.noise_lacunarity, pp.noise_diminish,
                           scale_pos_out, noise_out);
        float scale_noise_out = noise_out * pp.scale_noise_in2;
        float sum_out = scale_xyz_out + scale_noise_out;
        float sin_out = sin(sum_out);
        float scale_out = sin_out * pp.scale_in2;
        float bias_out = scale_out + pp.bias_in2;
        float power_out = pow(max(bias_out, 0.0), pp.power_in2);
        return lerp(pp.color_mix_bg, pp.color_mix_fg, clamp(power_out, 0.0, 1.0));
    }
    return float3(-1.0);
}

StdSurfaceParams loadStdSurfaceParams(uint matId, float3 posObject)
{
    StdSurfaceParams sp = stdSurfaceParams[matId];
    ProceduralParams pp = proceduralParams[matId];
    if (pp.type != PROC_TYPE_NONE)
    {
        float3 color = evalProceduralColor(pp, posObject);
        if (color.x >= 0.0)
        {
            sp.base_color = color;
            sp.subsurface_color = color;
        }
    }
    return sp;
}

