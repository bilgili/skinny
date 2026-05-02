// Uniform sphere sampler — isotropic direction distribution.
// Commonly used as the MIS companion for importance-sampled estimators.

import common;
import interfaces;

struct UniformSphereSampler : ISampler
{
    float3 sampleDirection(float2 u)
    {
        return sampleUniformSphere(u);
    }

    float pdf(float3 L)
    {
        return uniformSpherePdf();
    }
}
