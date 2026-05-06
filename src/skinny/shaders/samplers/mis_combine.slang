// MIS (multiple importance sampling) helper. Computes power-heuristic
// weights for a primary + companion sampler pair. Not itself an ISampler —
// it's a utility for estimators that draw from two distributions.

import common;
import interfaces;

float misPrimaryWeight<TA : ISampler, TB : ISampler>(
    TA primary, TB companion, float3 wi, float3 wo)
{
    return powerHeuristic(primary.pdf(wi, wo), companion.pdf(wi, wo));
}

float misCompanionWeight<TA : ISampler, TB : ISampler>(
    TA primary, TB companion, float3 wi, float3 wo)
{
    return powerHeuristic(companion.pdf(wi, wo), primary.pdf(wi, wo));
}
