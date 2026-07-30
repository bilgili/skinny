// Auto-generated. Do not edit — written by megakernel_sources.emit_generated_materials().
// Imports each scene MaterialX nodegraph as a Slang module.
// Per-graph modules expose only `evalGraph_<target>` + the
// matching `GraphParams_<target>` / `GraphOutputs_<target>`
// structs; their `internal` helpers stay module-private, so
// duplicate symbol names across graphs do not collide.

import mtlx_std_surface;  // StdSurfaceParams


// Evaluate the per-hit nodegraph and overlay each driven
// std_surface input on `sp`. graphId 0 / 1 reserved (skin /
// flat) — callers gate the call by `materialGraphId(mid) >= 2`.
void evalSceneGraph(uint graphId, uint matId,
                    float3 P, float3 N, float3 T, float2 UV,
                    inout StdSurfaceParams sp)
{
    switch (graphId)
    {
        default:
            sp.base_color = float3(1.0, 0.0, 1.0);
            return;
    }
}

// Returns true and fills `outColor` only when the active graph
// drives std_surface.base_color (marble, wood). Graphs that
// drive only other inputs (brass: specular_roughness + coat_*)
// return false; the caller keeps the SSBO-uploaded constant
// for FlatMaterial.albedo.
bool evalSceneGraphBaseColor(uint graphId, uint matId,
                              float3 P, float3 N, float3 T, float2 UV,
                              out float3 outColor)
{
    outColor = float3(0.0);
    switch (graphId)
    {
        default:
            return false;
    }
}
