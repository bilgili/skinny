// Cosine-weighted hemisphere sampler (Lambertian diffuse).

import common;
import interfaces;

struct LambertSampler : ISampler
{
    float3 N;

    float3 sampleDirection(float2 u)
    {
        return sampleCosineHemisphere(u, N);
    }

    float pdf(float3 L)
    {
        return cosineHemispherePdf(L, N);
    }
}
