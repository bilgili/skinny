// Debug material — visualizes surface normals as color (0.5 + 0.5*N).

import common;
import interfaces;

struct DebugNormalMaterial : IMaterial
{
    BSDFSample sample(float3 wo, inout RNG rng)
    {
        BSDFSample s;
        s.valid    = false;
        s.emission = float3(0.0);
        return s;
    }

    BSDFEval evaluate(float3 wo, float3 wi)
    {
        BSDFEval e;
        e.response = float3(0.0);
        e.pdf      = 0.0;
        return e;
    }
}
