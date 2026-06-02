// Pluggable directional-proposal seam at the BSDF bounce.
//
// The integrator draws its next direction from a runtime-selected *mixture*
// of proposals (fc.proposalMask + fc.proposalAlpha) via one-sample MIS,
// dividing throughput by the full mixture pdf. Proposals return exact
// solid-angle pdfs so the estimator stays unbiased.
//
// This module currently ships the BSDF proposal only — it delegates to the
// material's own importance sampler. With just the BSDF bit active (the
// default) the mixture collapses to the material's native sample(), so the
// rendered image is bit-identical to the pre-seam renderer. The
// environment-importance proposal and the ≥2-proposal mixture math (weight =
// f·cos / Σ_k α_k·pdf_k, with delta-lobe pass-through and NEE coupling) land
// in follow-up changes; the interface + dispatch are shaped here so they drop
// in without a rewrite.

import common;       // RNG
import interfaces;   // IMaterial, BSDFSample
import bindings;     // fc (proposalMask / proposalAlpha)

// Proposal slot bits — must match the host registry (sampling/registry.py).
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

// Draw the next bounce direction from the active proposal mixture.
//
// Returns the material's native BSDFSample (wi/weight/pdf/emission/
// transmitted/valid). The single-proposal fast path returns it verbatim for
// bit-exact parity; the multi-proposal path (env/neural, later) will
// recompute weight = f·cos / Σ_k α_k·pdf_k and store mixPdf in `pdf`.
BSDFSample sampleBounceDirection<TM : IMaterial>(TM mat, ProposalContext c, inout RNG rng)
{
    // Default + only-shipped path: the BSDF proposal delegates to the
    // material's own VNDF / cosine sampler, so the throughput weight is the
    // material's exact value — identical to calling mat.sample() directly.
    return mat.sample(c.woT, rng);
}

// Exact solid-angle pdf the active proposal mixture assigns to `wiT` (tangent
// space). NEE's MIS companion weight will route through this so the direct-
// lighting estimate stays coupled to the bounce sampler. BSDF-only ⇒ the
// material's own bsdf pdf.
float mixtureProposalPdf<TM : IMaterial>(TM mat, ProposalContext c, float3 wiT)
{
    return mat.evaluate(c.woT, wiT).pdf;
}
