// Flat-material bounce evaluation — the MATERIAL_TYPE_FLAT branch of
// integrators/path.slang::evaluateBounce, standalone. Imports only the flat
// material + the material-free NEE module, so a flat-only wavefront shade
// kernel does NOT pull in the skin BSSRDF or python-material dispatcher (the
// heavy imports that bloat the catch-all shade kernel and trip MoltenVK's
// Metal compile). Covers flat + MaterialX-graph materials (graphs overlay the
// flat std_surface path via loadFlatMaterial).

import common;        // RNG, Ray, buildBasis, worldToTangent, tangentToWorld
import interfaces;    // BounceResult, BSDFSample, HitInfo
import nee;           // allLightsNEE (material-free)
import materials.flat.flat_material;  // FlatMaterial, loadFlatMaterial
import sampling.proposal;             // sampleBounceDirection (BSDF-only ⇒ mat.sample)

BounceResult evaluateFlatBounce(HitInfo h, Ray r, uint bounce, inout RNG rng)
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

    FlatMaterial mat = loadFlatMaterial(h);
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

    ProposalContext pctx;
    pctx.woT = wo; pctx.N = N; pctx.T = T; pctx.B = B;
    pctx.position = h.position; pctx.materialId = h.materialId;
    BSDFSample bs = sampleBounceDirection(mat, pctx, rng);
    if (bs.valid)
        bs.wi = tangentToWorld(bs.wi, T, B, N);
    br.bsdfSample = bs;
    return br;
}
