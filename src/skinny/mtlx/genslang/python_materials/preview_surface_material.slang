import common;

import interfaces;

public struct PreviewSurfaceInputs
{
    public float3 diffuseColor;
    public float roughness;
    public float metallic;
    public float specular;
    public float3 emissiveColor;
    public float opacity;
    public float ior;
    public float normalScale;
}

public struct PreviewSurfaceMaterial : IMaterial
{
    public PreviewSurfaceInputs params;

    BSDFSample sample(float3 wo, inout RNG rng)
    {
        BSDFSample s;
        float2 u = rng.next2();
        float3 wi = sampleCosineHemisphereTS(u);
        s.wi = wi;
        s.weight = params.diffuseColor;
        s.pdf = cosineHemispherePdfTS(wi);
        s.emission = params.emissiveColor;
        s.valid = (wi.z > 0.0);
        s.transmitted = false;
        return s;
    }

    BSDFSample evaluate(float3 wo, float3 wi)
    {
        BSDFSample e;
        float NdotL = max(wi.z, 0.0);
        e.response = (params.diffuseColor * (NdotL / PI));
        e.pdf = (NdotL / PI);
        e.wi = float3(0.0, 0.0, 0.0);
        e.weight = float3(0.0, 0.0, 0.0);
        e.emission = float3(0.0, 0.0, 0.0);
        e.valid = false;
        e.transmitted = false;
        return e;
    }
}

public float checker_pattern(float2 uv, float scale)
{
    var su = floor((uv.x * scale));
    var sv = floor((uv.y * scale));
    return fmod((su + sv), 2.0);
}

public float3 perturb_normal(float3 N, float scale)
{
    return normalize(lerp(float3(0.0, 0.0, 1.0), N, scale));
}
