// Cosine-weighted hemisphere sampler (Lambertian diffuse).
// Operates in tangent space (N = +Z).

import common;
import interfaces;

struct LambertSampler : ISampler
{
    float3 sample(float3 wo, float2 uv)
    {
        return sampleCosineHemisphereTS(uv);
    }

    float pdf(float3 wi, float3 wo)
    {
        return cosineHemispherePdfTS(wi);
    }
}
