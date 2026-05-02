// MIS (multiple importance sampling) helper. Computes power-heuristic
// weights for a primary + companion sampler pair. Not itself an ISampler —
// it's a utility for estimators that draw from two distributions.

import common;
import interfaces;

float misPrimaryWeight<TA : ISampler, TB : ISampler>(
    TA primary, TB companion, float3 L)
{
    return powerHeuristic(primary.pdf(L), companion.pdf(L));
}

float misCompanionWeight<TA : ISampler, TB : ISampler>(
    TA primary, TB companion, float3 L)
{
    return powerHeuristic(companion.pdf(L), primary.pdf(L));
}
