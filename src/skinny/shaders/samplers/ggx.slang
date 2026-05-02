// GGX microfacet importance sampler. Draws half-vectors from the GGX
// normal distribution, reflects V to produce the sample direction.
// Configuration (N, V, roughness) stored in struct fields.

import common;
import interfaces;

struct GGXSampler : ISampler
{
    float3 N;
    float3 V;
    float  roughness;

    float3 sampleDirection(float2 u)
    {
        float a  = max(roughness * roughness, 1e-4);
        float a2 = a * a;

        float cosTheta = sqrt((1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y));
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi      = 2.0 * PI * u.x;

        float3 T, B;
        buildBasis(N, T, B);
        float3 H = normalize(
            T * cos(phi) * sinTheta +
            B * sin(phi) * sinTheta +
            N * cosTheta
        );

        return reflect(-V, H);
    }

    float pdf(float3 L)
    {
        float3 H    = normalize(V + L);
        float  NdotH = max(dot(N, H), 0.0);
        float  VdotH = max(dot(V, H), 1e-6);
        if (dot(N, L) <= 0.0 || NdotH <= 0.0)
            return 0.0;

        float a  = max(roughness * roughness, 1e-4);
        float a2 = a * a;
        float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D  = a2 / (PI * d * d);

        return D * NdotH / (4.0 * VdotH);
    }
}
