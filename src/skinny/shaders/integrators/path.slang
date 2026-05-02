// Unidirectional path tracer (fc.integratorMode == 0). Surface terms
// without MIS arms; environment via importance-sampled BSDFs only.
// Phase 3 delegates to material_eval — current skin estimators read
// fc.integratorMode internally to gate MIS branches, so the single
// switch in main_pass already steers behavior. Phase 4 will lift MIS
// policy into the integrator (no longer reading fc inside estimators).

import common;
import interfaces;
import material_eval;

struct PathTracer : IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        return evalMaterial(firstHit, ray, rng);
    }
}
