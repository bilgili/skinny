// Pluggable directional-proposal seam at the BSDF bounce.
//
// The integrator draws its next direction from a runtime-selected *mixture*
// of proposals (fc.proposalMask + fc.proposalAlpha) via one-sample MIS,
// dividing throughput by the full mixture pdf. Proposals return exact
// solid-angle pdfs so the estimator stays unbiased.
//
// Shipped proposals:
//   bit0 PROPOSAL_BSDF — the material's own importance sampler (always on).
//   bit1 PROPOSAL_ENV  — importance-sample the environment distribution
//                        (reuses the env CDFs; no new GPU state).
//
// With only the BSDF bit active (the default) the mixture collapses to the
// material's native sample(), so the rendered image is bit-identical to the
// pre-seam renderer. Correctness rules baked in here:
//   - Delta / invalid lobes pass through unmixed (can't MIS a singular lobe
//     with a continuous proposal).
//   - Surface emission is technique-independent — always taken from the
//     material, even when the env technique is picked, so emissive surfaces
//     don't lose energy.
//   - The bounce mixPdf and NEE's companion pdf (mixtureProposalPdf) use the
//     SAME α_b·bsdfPdf + α_e·envPdf for a given wi, so direct + indirect
//     lighting stay MIS-consistent (the NEE-coupling rule).

import common;       // RNG, worldToTangent / tangentToWorld
import interfaces;   // IMaterial, BSDFSample
import bindings;     // fc (proposalMask / proposalAlpha / envIntensity)
import environment;  // sampleEnvDir, envPdf, EnvSample

// Proposal slot bits — must match the host registry (sampling/proposals.py).
static const uint PROPOSAL_BSDF = 0x1u;
static const uint PROPOSAL_ENV  = 0x2u;

// Shading condition handed to every proposal. The tangent frame (T, B, N)
// lets a world-space proposal (env) convert to/from the bounce's tangent
// space (N = +Z); woT is the outgoing direction already in tangent space.
struct ProposalContext
{
    float3 woT;        // outgoing dir, tangent space (N = +Z)
    float3 N;          // world-space shading normal
    float3 T;          // world-space tangent
    float3 B;          // world-space bitangent
    float3 position;   // world-space shading point
    uint   materialId;
};

// Draw the next bounce direction from the active proposal mixture. Returns a
// BSDFSample (tangent-space wi; caller converts to world). The single-proposal
// fast path returns the material's native sample verbatim for bit-exact
// parity; the multi-proposal path recomputes weight = f·cos / Σ_k α_k·pdf_k.
BSDFSample sampleBounceDirection<TM : IMaterial>(TM mat, ProposalContext c, inout RNG rng)
{
    // Always draw the BSDF candidate first: it carries surface emission
    // (technique-independent) and is the BSDF proposal's sample. Drawing it
    // unconditionally keeps the BSDF-only RNG stream identical to pre-seam.
    BSDFSample cand = mat.sample(c.woT, rng);

    // Fast path — BSDF-only (default): bit-identical to the pre-seam renderer.
    if (fc.proposalMask == PROPOSAL_BSDF)
        return cand;

    // A singular (delta) lobe or invalid sample can't MIS-mix with a
    // continuous proposal — take the material's native sample (emission kept).
    if (!cand.valid || cand.pdf <= 0.0)
        return cand;

    float aB = fc.proposalAlpha.x;   // BSDF selection weight
    float aE = fc.proposalAlpha.y;   // env selection weight

    // One-sample MIS: pick a technique ∝ α. BSDF reuses `cand.wi`; env draws a
    // fresh direction from the environment distribution.
    float3 wiT;
    if (rng.next() < aB)
    {
        wiT = cand.wi;
    }
    else
    {
        EnvSample es = sampleEnvDir(rng.next2(), fc.envIntensity);
        if (es.pdf <= 0.0)            // no distribution → zero-contribution
        {
            cand.valid = false;
            return cand;              // emission already added by the bounce loop
        }
        wiT = worldToTangent(es.dir, c.T, c.B, c.N);
        if (wiT.z <= 0.0)            // below the surface horizon → 0 contribution
        {
            cand.valid = false;
            return cand;
        }
    }

    // Full mixture pdf (solid-angle) at the chosen direction. Use the material
    // EVALUATE pdf for the BSDF component so it matches mixtureProposalPdf().
    BSDFSample ev = mat.evaluate(c.woT, wiT);
    float3 worldWi = tangentToWorld(wiT, c.T, c.B, c.N);
    float  mixPdf  = aB * ev.pdf + aE * envPdf(worldWi);
    if (mixPdf <= 0.0)
    {
        cand.valid = false;
        return cand;
    }

    BSDFSample outp;
    outp.valid       = true;
    outp.transmitted = false;          // env + reflective (non-delta) only
    outp.emission    = cand.emission;  // technique-independent self-emission
    outp.wi          = wiT;
    outp.pdf         = mixPdf;
    outp.weight      = ev.response / mixPdf;
    outp.response    = float3(0.0);
    return outp;
}

// Exact solid-angle pdf the active proposal mixture assigns to `wiT` (tangent
// space). NEE's MIS companion weight routes through this so the direct-lighting
// estimate stays coupled to the bounce sampler. BSDF-only ⇒ the material's own
// bsdf pdf (bit-identical to the pre-seam NEE weight).
float mixtureProposalPdf<TM : IMaterial>(TM mat, ProposalContext c, float3 wiT)
{
    float bsdfPdf = mat.evaluate(c.woT, wiT).pdf;
    if (fc.proposalMask == PROPOSAL_BSDF)
        return bsdfPdf;

    float aB = fc.proposalAlpha.x;
    float aE = fc.proposalAlpha.y;
    float ePdf = 0.0;
    if ((fc.proposalMask & PROPOSAL_ENV) != 0u && wiT.z > 0.0)
    {
        float3 worldWi = tangentToWorld(wiT, c.T, c.B, c.N);
        ePdf = envPdf(worldWi);
    }
    return aB * bsdfPdf + aE * ePdf;
}
