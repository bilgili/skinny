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

// Second stage for the multi-pass encoder test (task 1.4): read-modify-write the
// same buffer. Encoded after `computeMain` with a global barrier between, the
// result must be `tid + 1000` — proving stage 2 observed stage 1's writes within
// one command encoder + single submit.
[shader("compute")]
[numthreads(64, 1, 1)]
void addBias(uint3 tid : SV_DispatchThreadID)
{
    result[tid.x] += 1000u;
}
