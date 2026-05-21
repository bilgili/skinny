// Debug material — visualizes surface normals as color (0.5 + 0.5*N).

import common;
import interfaces;

struct DebugNormalMaterial : IMaterial
{
    BSDFSample sample(float3 wo, inout RNG rng)
    {
        BSDFSample s;
        s.valid       = false;
        s.transmitted = false;
        s.wi          = float3(0.0);
        s.weight      = float3(0.0);
        s.response    = float3(0.0);
        s.pdf         = 0.0;
        s.emission    = float3(0.0);
        return s;
    }

    BSDFSample evaluate(float3 wo, float3 wi)
    {
        BSDFSample e;
        e.valid       = false;
        e.transmitted = false;
        e.wi          = float3(0.0);
        e.weight      = float3(0.0);
        e.response    = float3(0.0);
        e.pdf         = 0.0;
        e.emission    = float3(0.0);
        return e;
    }
}
