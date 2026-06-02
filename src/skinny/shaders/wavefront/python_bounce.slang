// Python-material bounce evaluation — the MATERIAL_TYPE_PYTHON branch of
// integrators/path.slang::evaluateBounce, standalone. Imports only the generated
// python_materials_dispatcher (+ NEE), NOT the skin BSSRDF, so a python-only
// wavefront shade kernel stays off the skin estimator. The dispatcher is
// monolithic over all python materials (per-single-material isolation would need
// a subset dispatcher — deferred), so this evaluator covers the whole python set.

import common;        // RNG, Ray, buildBasis, worldToTangent, tangentToWorld
import interfaces;    // BounceResult, BSDFSample, HitInfo
import bindings;      // pythonMaterialId
import nee;           // allLightsNEE
import python_materials_dispatcher;    // PythonMaterial, loadPythonMaterial

BounceResult evaluatePythonBounce(HitInfo h, Ray r, uint bounce, inout RNG rng)
{
    float3 V = -r.direction;

    BounceResult br;
    br.directLight            = float3(0.0);
    br.fullRadiance           = float3(0.0);
    br.bsdfSample.valid       = false;
    br.bsdfSample.transmitted = false;
    br.bsdfSample.pdf         = 0.0;
    br.bsdfSample.emission    = float3(0.0);
    br.bsdfSample.wi          = float3(0.0);
    br.bsdfSample.weight      = float3(0.0);

    uint pyId = pythonMaterialId(h.materialId);
    PythonMaterial mat = loadPythonMaterial(h, pyId);

    float3 N = mat.data.N;
    float3 T, B;
    if (h.hasTangent)
    {
        T = normalize(h.tangent - N * dot(h.tangent, N));
        B = cross(N, T);
    }
    else
    {
        buildBasis(N, T, B);
    }
    float3 wo = worldToTangent(V, T, B, N);

    br.directLight = allLightsNEE(mat, h, N, T, B, wo, rng);

    BSDFSample bs = mat.sample(wo, rng);
    if (bs.valid)
        bs.wi = tangentToWorld(bs.wi, T, B, N);
    br.bsdfSample = bs;
    return br;
}
