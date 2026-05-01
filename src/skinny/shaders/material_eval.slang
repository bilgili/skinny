// Unified material dispatch — routes each hit to the appropriate evaluator
// so the main entry point is material-agnostic.
//
// Per-material flags packed into materialTypes[i]:
//   bits  0-7 : material type code (MATERIAL_TYPE_SKIN | MATERIAL_TYPE_FLAT)
//   bits  8-9 : scatter mode for skin slots — bit 0 = BSSRDF, bit 1 = volume
//   bits 10-31: reserved
//
// `bindings.slang` defines `materialTypes[]`; the encoding below is the
// agreed contract with `_pack_material_types_buffer()` on the renderer.

import common;
import bindings;
import flat_material;
import skin_material;

float3 evalMaterial(HitInfo hit, Ray ray, inout RNG rng)
{
    uint matFlags    = materialTypes[hit.materialId];
    uint matType     = matFlags & 0xFFu;
    uint scatterBits = (matFlags >> 8) & 0x3u;

    if (matType == MATERIAL_TYPE_FLAT)
        return evalFlatMaterial(hit, ray, rng);

    return evalSkinMaterial(
        hit, ray, rng, fc, lightDirection, lightRadiance, scatterBits
    );
}
