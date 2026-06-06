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
import interfaces;   // IMaterial, BSDFSample, WfNeuralSample
import bindings;     // fc (proposalMask / proposalAlpha / envIntensity)
import environment;  // sampleEnvDir, envPdf, EnvSample
import sampling.neural_proposal;  // neuralPdfWorld (inline inverse) + weight buffers

// Proposal slot bits — must match the host registry (sampling/proposals.py).
static const uint PROPOSAL_BSDF   = 0x1u;
static const uint PROPOSAL_ENV    = 0x2u;
static const uint PROPOSAL_NEURAL = 0x4u;   // bit2 — learned spline-flow proposal

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
    // Neural proposal (bit2): the wavefront pre-pass's precomputed per-lane
    // FORWARD sample. neuralValid = false on the megakernel and on non-flat
    // wavefront lanes — those keep the {bsdf,env} subset (weights renormalise,
    // staying per-lane unbiased). The arbitrary-direction INVERSE pdf is always
    // evaluated inline (neuralPdfWorld); only the forward candidate is
    // precomputed. Stamped with the producing network version (baseline 0).
    float3 neuralWi;        // world-space neural-drawn direction
    float  neuralPdf;       // solid-angle pdf at neuralWi (forward)
    uint   networkVersion;
    bool   neuralValid;
};

// Per-lane effective one-sample-MIS selection weights. The neural proposal
// participates only when its bit is set AND a forward candidate exists for this
// lane (neuralValid); otherwise its weight folds away and {bsdf,env} renormalise
// to Σ = 1, keeping each lane's estimator self-consistent + unbiased. With
// {bsdf,env} (no neural) this is identical to the pre-change weights (Σ already
// 1 ⇒ the divide is a no-op).
void proposalWeights(ProposalContext c, out float aB, out float aE, out float aN)
{
    aB = fc.proposalAlpha.x;
    aE = ((fc.proposalMask & PROPOSAL_ENV) != 0u) ? fc.proposalAlpha.y : 0.0;
    bool neuralOn = ((fc.proposalMask & PROPOSAL_NEURAL) != 0u) && c.neuralValid;
    aN = neuralOn ? fc.proposalAlpha.z : 0.0;
    float s = aB + aE + aN;
    if (s > 0.0) { aB /= s; aE /= s; aN /= s; }
}

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

    float aB, aE, aN;
    proposalWeights(c, aB, aE, aN);

    // One-sample MIS: pick a technique ∝ α over [bsdf | env | neural]. BSDF
    // reuses `cand.wi`; env draws a fresh env direction; neural reuses the
    // pre-pass's precomputed world candidate (no inline forward MLP).
    float3 wiT;
    float  xi = rng.next();
    bool   pickedNeural = false;
    if (xi < aB)
    {
        wiT = cand.wi;
    }
    else if (xi < aB + aE)
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
    else
    {
        // Neural technique — precomputed per-lane forward sample (world space).
        pickedNeural = true;
        wiT = worldToTangent(c.neuralWi, c.T, c.B, c.N);
        if (wiT.z <= 0.0)
        {
            cand.valid = false;
            return cand;
        }
    }

    // Full mixture pdf (solid-angle) at the chosen direction. Use the material
    // EVALUATE pdf for the BSDF component so it matches mixtureProposalPdf().
    BSDFSample ev = mat.evaluate(c.woT, wiT);
    float3 worldWi = tangentToWorld(wiT, c.T, c.B, c.N);
    float  mixPdf  = aB * ev.pdf;
    if (aE > 0.0)
        mixPdf += aE * envPdf(worldWi);
    if (aN > 0.0)
    {
        // Neural pdf at the chosen wi: reuse the precomputed forward pdf when
        // neural drew it, else the inline inverse (the only inline MLP eval).
        float pN = pickedNeural
                 ? c.neuralPdf
                 : neuralPdfWorld(c.position, c.N, c.T, c.B,
                                  tangentToWorld(c.woT, c.T, c.B, c.N), worldWi);
        mixPdf += aN * pN;
    }
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

    float aB, aE, aN;
    proposalWeights(c, aB, aE, aN);
    float pdf = aB * bsdfPdf;
    if ((aE > 0.0 || aN > 0.0) && wiT.z > 0.0)
    {
        float3 worldWi = tangentToWorld(wiT, c.T, c.B, c.N);
        if (aE > 0.0)
            pdf += aE * envPdf(worldWi);
        if (aN > 0.0)
            pdf += aN * neuralPdfWorld(c.position, c.N, c.T, c.B,
                                       tangentToWorld(c.woT, c.T, c.B, c.N), worldWi);
    }
    return pdf;
}
