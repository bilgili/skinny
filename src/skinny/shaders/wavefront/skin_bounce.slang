// Skin-material bounce evaluation — the MATERIAL_TYPE_SKIN branch of
// integrators/path.slang::evaluateBounce, standalone. Imports only the skin
// material (BSSRDF §1–§6) + bindings, so a skin-only wavefront shade kernel does
// NOT pull in the python-material dispatcher or the flat material (the heavy
// imports that bloat the combined catch-all and trip MoltenVK's Metal compile).

import common;        // RNG, Ray
import interfaces;    // BounceResult, BSDFSample, HitInfo
import bindings;      // materialTypes
import materials.skin.skin_material;   // evalSkinRadiance

BounceResult evaluateSkinBounce(HitInfo h, Ray r, uint bounce, inout RNG rng)
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

    uint matFlags    = materialTypes[h.materialId];
    uint scatterBits = (matFlags >> 8) & 0x3u;
    br.fullRadiance  = evalSkinRadiance(h, r, rng, scatterBits);
    return br;
}
