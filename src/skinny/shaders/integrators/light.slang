// Light-tracer / volume-connection integrator (fc.integratorMode == 2).
// Skin path activates volume.slang's light-connection branch via
// fc.integratorMode >= 2u; flat path falls back to MIS behavior. Phase 3
// delegates to material_eval — the dedicated light-subpath construction
// will land alongside the BDPT estimator generalization in a later phase.

import common;
import interfaces;
import material_eval;

struct LightTracer : IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        return evalMaterial(firstHit, ray, rng);
    }
}
