// Test harness for the ReSTIR reservoir core (restir/reservoir.slang).
//
// Synthetic 1D RIS probe: integrand f(x)=x² on [0,1] with a uniform source
// pdf. The RIS estimate f(y)·W is an unbiased estimate of ∫₀¹ f = 1/3 for any
// M and any positive target p̂ — this exercises reservoir selection (y ∝ p̂),
// the wSum/(M·p̂) contribution weight, and the f≠p̂ case. Averaging is done on
// the GPU so the Python test makes a single dispatch.

import common;          // RNG, pcgHash
import restir.reservoir;

float synthF(float x) { return x * x; }                         // integrand, ∫₀¹ = 1/3

// Target p̂: a deliberately imperfect proxy (p̂=x) when perfectTarget==0, or
// the exact integrand (p̂=x²) when perfectTarget!=0 (lower variance, still 1/3).
float synthPHat(float x, uint perfectTarget)
{
    return (perfectTarget != 0u) ? (x * x) : x;
}

// Run `trials` independent RIS estimates of ∫f with M candidates each; return
// the mean (→ 1/3 as trials → ∞, for any M / target).
float test_ris_mean(uint baseSeed, uint m, uint trials, uint perfectTarget)
{
    float acc = 0.0;
    for (uint t = 0u; t < trials; t++)
    {
        RNG rng;
        rng.state = pcgHash(baseSeed + t * 2654435761u);

        Reservoir r;
        reservoirReset(r);
        for (uint i = 0u; i < m; i++)
        {
            float x  = rng.next();                 // candidate ~ uniform
            float ph = synthPHat(x, perfectTarget);
            float w  = ph;                          // p̂(x) / sourcePdf, sourcePdf = 1
            LightSampleRef xr;
            xr.packed = 0u;
            xr.uv = float2(x, 0.0);
            reservoirUpdate(r, xr, w, ph, rng.next());
        }
        reservoirFinalize(r);
        acc += synthF(r.y.uv.x) * r.W;
    }
    return acc / float(trials);
}

// Empirical selection frequency check: fraction of `trials` in which the LAST
// of two candidates (with weights wA, wB) is the survivor — should approach
// wB/(wA+wB). Validates the reservoir replacement probability.
float test_select_frequency(uint baseSeed, uint trials, float wA, float wB)
{
    uint countB = 0u;
    for (uint t = 0u; t < trials; t++)
    {
        RNG rng;
        rng.state = pcgHash(baseSeed + t * 40503u + 1u);
        Reservoir r;
        reservoirReset(r);
        LightSampleRef a; a.packed = 0u; a.uv = float2(0.0, 0.0);
        LightSampleRef b; b.packed = 1u; b.uv = float2(1.0, 0.0);
        reservoirUpdate(r, a, wA, wA, rng.next());
        reservoirUpdate(r, b, wB, wB, rng.next());
        if (r.y.packed == 1u) countB++;
    }
    return float(countB) / float(trials);
}
