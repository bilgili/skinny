// Debug-normal bounce evaluation — the `default` branch of
// integrators/path.slang::evaluateBounce, standalone. Trivial normal
// visualisation; imports nothing heavy, so it packs into any group for free.

import common;        // RNG, Ray
import interfaces;    // BounceResult, BSDFSample, HitInfo

BounceResult evaluateDebugBounce(HitInfo h, Ray r, uint bounce, inout RNG rng)
{
    BounceResult br;
    br.directLight            = float3(0.0);
    br.fullRadiance           = float3(0.0);
    br.bsdfSample.valid       = false;
    br.bsdfSample.transmitted = false;
    br.bsdfSample.pdf         = 0.0;
    br.bsdfSample.emission    = float3(0.0);
    br.bsdfSample.wi          = float3(0.0);
    br.bsdfSample.weight      = float3(0.0);

    br.fullRadiance = 0.5 + 0.5 * h.normal;
    return br;
}
