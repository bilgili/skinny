// Trivial foundation compute kernel: write each thread's global invocation id
// into a storage buffer. Used by the Metal/Vulkan foundation parity test
// (deterministic output = arange(N)) and the windowed present smoke is in the
// host (skinny.metal_context.present_clear), not here.
//
// The entry point is `computeMain`, NOT `main`: Slang's Metal target reserves
// `main` and renames it `main_0`, which makes Metal compute-pipeline creation
// fail with SLANG_FAIL. Keep the non-reserved name for cross-backend builds.

RWStructuredBuffer<uint> result;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    result[tid.x] = tid.x;
}
