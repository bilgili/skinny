// Minimal tool-buffer read/write probe — diagnostic for the 6.1 structural-AOV
// mechanism, NOT part of the renderer. Tiny on purpose: it touches only
// toolBuffer (binding 30), so it compiles in the cheap "foundation" class and
// never drags in main_pass (whose huge cold compile is the MTLCompilerService
// wedge hazard).
//
// Each thread reads toolBuffer[0].x (the armed tool mode). Iff it equals
// TOOL_MODE_STRUCTURAL (4) it writes a deterministic sentinel into
// toolBuffer[16 + i]; otherwise a distinct "mode-not-seen" marker. The host
// poisons the region first, so the read-back splits three outcomes:
//   (i, 2, 3, 4)        → CPU→GPU arm visible AND GPU→CPU read-back works
//   (-1, -1, -1, -1)    → dispatch ran but the armed mode was not visible
//   poison (unchanged)  → the GPU write never reached the host read-back
//
// Entry point is `computeMain`, never `main` (Slang renames `main`→`main_0` on
// the Metal target and pipeline creation fails SLANG_FAIL).
[[vk::binding(30)]] RWStructuredBuffer<float4> toolBuffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= 4096u)
        return;
    if (asuint(toolBuffer[0].x) == 4u)
        toolBuffer[16u + i] = float4(float(i), 2.0, 3.0, 4.0);
    else
        toolBuffer[16u + i] = float4(-1.0, -1.0, -1.0, -1.0);
}
