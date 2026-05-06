// Uniform sphere sampler — isotropic direction distribution.
// Commonly used as the MIS companion for importance-sampled estimators.

import common;
import interfaces;

struct UniformSphereSampler : ISampler
{
    float3 sample(float3 wo, float2 uv)
    {
        return sampleUniformSphere(uv);
    }

    float pdf(float3 wi, float3 wo)
    {
        return uniformSpherePdf();
    }
}
