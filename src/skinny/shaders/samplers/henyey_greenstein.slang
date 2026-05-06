// Henyey-Greenstein phase function sampler for volumetric scattering.
// Operates in tangent space; scatters around wo.

import common;
import interfaces;

struct HenyeyGreensteinSampler : ISampler
{
    float g;

    float3 sample(float3 wo, float2 uv)
    {
        float cosTheta;
        if (abs(g) > 1e-3)
        {
            float t = (1.0 - g * g) / (1.0 - g + 2.0 * g * uv.x);
            cosTheta = (1.0 + g * g - t * t) / (2.0 * g);
        }
        else
        {
            cosTheta = 1.0 - 2.0 * uv.x;
        }
        cosTheta = clamp(cosTheta, -1.0, 1.0);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float phi      = 2.0 * PI * uv.y;

        float3 T, B;
        buildBasis(wo, T, B);

        return normalize(
            T * cos(phi) * sinTheta +
            B * sin(phi) * sinTheta +
            wo * cosTheta
        );
    }

    float pdf(float3 wi, float3 wo)
    {
        float cosTheta = dot(normalize(wi), wo);
        float g2  = g * g;
        float num = 1.0 - g2;
        float den = pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
        return num / (4.0 * PI * den);
    }
}
