// Debug material — visualizes surface normals as color (0.5 + 0.5*N).
// Smoke test for the one-file material registration contract.

import common;
import interfaces;

struct DebugNormalMaterial : IMaterial
{
    HitInfo hit;

    float3 evalRadiance(Ray ray, inout RNG rng)
    {
        return 0.5 + 0.5 * hit.normal;
    }
}

DebugNormalMaterial loadDebugNormal(HitInfo h)
{
    DebugNormalMaterial m;
    m.hit = h;
    return m;
}
