// GGX microfacet importance sampler. Draws half-vectors from the GGX
// normal distribution in tangent space (N = +Z), reflects wo to produce wi.

import common;
import interfaces;

struct GGXSampler : ISampler
{
    float roughness;

    float3 sample(float3 wo, float2 uv)
    {
        float a  = max(roughness * roughness, 1e-4);
        float a2 = a * a;

        float cosTheta = sqrt((1.0 - uv.y) / (1.0 + (a2 - 1.0) * uv.y));
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi      = 2.0 * PI * uv.x;

        float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

        return reflect(-wo, H);
    }

    float pdf(float3 wi, float3 wo)
    {
        float3 H     = normalize(wo + wi);
        float  NdotH = max(H.z, 0.0);
        float  VdotH = max(dot(wo, H), 1e-6);
        if (wi.z <= 0.0 || NdotH <= 0.0)
            return 0.0;

        float a  = max(roughness * roughness, 1e-4);
        float a2 = a * a;
        float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D  = a2 / (PI * d * d);

        return D * NdotH / (4.0 * VdotH);
    }
}
