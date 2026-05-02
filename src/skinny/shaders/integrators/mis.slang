// MIS path tracer (fc.integratorMode == 1). Combines BSDF and uniform
// environment samples via the power heuristic on the IBL specular and
// IBL diffuse estimators. Phase 3 delegates to material_eval; the MIS
// arms are activated by the existing fc.integratorMode != 0 branches in
// estimators/skin_ibl_specular.slang and estimators/skin_ibl_diffuse.slang.
// Phase 4 will move that policy into this integrator and drop the
// estimator-side branches.

import common;
import interfaces;
import material_eval;

struct MISPathTracer : IIntegrator
{
    float3 estimateRadiance(Ray ray, HitInfo firstHit, inout RNG rng)
    {
        return evalMaterial(firstHit, ray, rng);
    }
}
