// Test harness for the neural-flow Lambert directional chart (change
// directional-flow-parameterization). Exposes the chart pair + the underlying
// concentric square<->disk map so the Python tests can assert the chart is a
// measure-preserving bijection (round-trip identity) and that its true area
// scaling equals the coded constant Jacobian 2*pi (solid-angle normalization).

import sampling.neural_flow;

// ── Lambert square <-> hemisphere (V1) ─────────────────────────

float3 test_v1_square_to_dir(float2 z)
{
    return nf_square_to_hemi(z);
}

float2 test_v1_dir_to_square(float3 w)
{
    return nf_hemi_to_square(w);
}

// ── Shirley concentric square <-> unit disk ────────────────────

float2 test_v1_square_to_disk(float2 z)
{
    return nf_concentric_square_to_disk(z);
}

float2 test_v1_disk_to_square(float2 c)
{
    return nf_disk_to_square(c);
}
