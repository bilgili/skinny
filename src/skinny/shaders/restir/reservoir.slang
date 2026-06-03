// ReSTIR reservoir — weighted reservoir sampling (RIS) over a candidate stream.
//
// Pure: no scene bindings, so the core RIS math (selection ∝ weight, the
// contribution weight W) is unit-tested in isolation (tests/test_restir.py via
// the synthetic harness). The light-domain target p̂ and candidate generation
// live in restir/initial.slang, which needs the scene bindings.
//
// RIS recap: candidates x_i are drawn from a source pdf; each carries a weight
// w_i = p̂(x_i)/sourcePdf(x_i) and its target value p̂(x_i). The reservoir keeps
// one survivor y selected ∝ w_i. The final unbiased contribution weight is
//   W = (Σ w_i) / (M · p̂(y))
// and the estimate of ∫f is f(y)·W — unbiased for any M whenever p̂ > 0 where
// f ≠ 0 (Talbot et al. 2005; Bitterli et al. 2020).

import common;   // RNG

// Compact reference to a selected light sample, unified over light types.
//   packed: lightType:2 | lightId:30 ; uv: point-on-light / env-dir param.
struct LightSampleRef
{
    uint   packed;
    float2 uv;
};

// Per-pixel reservoir.
struct Reservoir
{
    LightSampleRef y;     // surviving sample
    float wSum;           // Σ RIS weights seen
    float W;              // contribution weight = wSum / (M · p̂(y))
    uint  M;              // candidate count (capped for temporal reuse)
    float pHat;           // cached target p̂(y)
};

void reservoirReset(out Reservoir r)
{
    r.y = LightSampleRef(0u, float2(0.0));
    r.wSum = 0.0;
    r.W = 0.0;
    r.M = 0u;
    r.pHat = 0.0;
}

// Stream a candidate `x` with RIS weight `w` (= p̂(x)/sourcePdf(x)) and its
// target value `pHat_x`. `rand` ∈ [0,1). Returns true when `x` is selected.
bool reservoirUpdate(inout Reservoir r, LightSampleRef x, float w, float pHat_x, float rand)
{
    r.wSum += w;
    r.M += 1u;
    bool take = (w > 0.0) && (rand * r.wSum < w);
    if (take)
    {
        r.y = x;
        r.pHat = pHat_x;
    }
    return take;
}

// Merge a finalized source reservoir `src` into `dst` as a single
// supercandidate with weight  w = pHatInDst · src.W · src.M, where pHatInDst is
// src.y's target value evaluated in dst's domain. For same-point temporal reuse
// pHatInDst == src.pHat (the geometry is unchanged); for spatial reuse it is
// src.y re-evaluated at the canonical shading point (the reconnection). dst.M
// accumulates src.M. `rand` ∈ [0,1). Returns true if dst adopted src.y.
//
// Combining reservoirs r₁…r_k into a fresh reservoir via reservoirMerge is the
// unbiased multi-reservoir RIS combination (Bitterli 2020 Alg. 4): each source
// contributes its W·M·p̂, the survivor is chosen ∝ those weights, and finalize
// yields the combined contribution weight over ΣM samples.
bool reservoirMerge(inout Reservoir dst, Reservoir src, float pHatInDst, float rand)
{
    float w = pHatInDst * src.W * float(src.M);
    dst.wSum += w;
    dst.M += src.M;
    bool take = (w > 0.0) && (rand * dst.wSum < w);
    if (take)
    {
        dst.y = src.y;
        dst.pHat = pHatInDst;
    }
    return take;
}

// Finalize the contribution weight after the candidate stream (or a merge).
void reservoirFinalize(inout Reservoir r)
{
    float denom = float(r.M) * r.pHat;
    r.W = (denom > 0.0) ? r.wSum / denom : 0.0;
}
