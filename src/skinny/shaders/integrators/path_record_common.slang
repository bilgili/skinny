// Shared neural training-record primitives — the ONE source of truth for the
// `PathRecord` layout, the append-buffer bindings (36/37), the bounds-safe
// `emitRecord`, and the backward-attribution weight `recordContrib`. Imported by
// both record producers:
//   - integrators/path_record.slang  (the `mainImageRecord` megakernel `.nrec`
//     dump + live drain), and
//   - wavefront/wf_records.slang      (the wavefront-native record emitter).
// Keeping the struct + attribution in one module guarantees the two streams are
// byte-for-byte compatible and compute the identical training weight.
//
// Scalar layout (-fvk-use-scalar-layout): float3 = 12 B tight → 64 B record
// stride. Host mirror: skinny.sampling.path_records (RECORD_STRIDE/RECORD_DTYPE).

static const uint REC_MAX_BOUNCES = 6;   // lockstep with integrators.path MAX_BOUNCES

// One training record — 64 B (see host mirror skinny.sampling.path_records).
struct PathRecord
{
    float3 pos;       // 0..12   world hit position (raw; trainer normalises via header AABB)
    float3 normal;    // 12..24  world shading normal (= HitInfo.normal, == neuralCondition N)
    float3 wo;        // 24..36  world outgoing dir toward the previous vertex (= -rayDir)
    float3 wiLocal;   // 36..48  sampled bounce dir in flow-local (x=T,y=N,z=B) frame, y-up
    float3 contrib;   // 48..60  (L_final - L_k)/beta_in_k = weight_k·Li_k  (RGB training weight)
    uint   depth;     // 60..64  bounce index (0 = primary hit)
};

// Append buffer (binding 36) + counter (binding 37: [0]=atomic count, [1]=capacity).
// `mainImage` never references them → dead-stripped from its entry (stays
// byte-identical); the host binds 1-element dummies until a dump/drain runs.
// Metal slot-cap gate (change metal-wavefront-parity phase 6): the wavefront
// program with the neural proposal un-stubbed (SKINNY_METAL_NEURAL) sits
// exactly at Metal's 31-buffer-slot argument table, and these two globals are
// dead there (wf_records.slang stubs the only wavefront emitter under
// SKINNY_METAL) — compile them out of that build so neuralWeights/Biases/
// Layers fit. The megakernel + Vulkan builds are unaffected.
#if !(defined(SKINNY_METAL) && defined(SKINNY_METAL_NEURAL))
[[vk::binding(36)]] RWStructuredBuffer<PathRecord> recordBuf;
[[vk::binding(37)]] RWStructuredBuffer<uint>       recordCounter;

void emitRecord(PathRecord rec)
{
    uint cap = recordCounter[1];
    uint idx;
    InterlockedAdd(recordCounter[0], 1u, idx);
    if (idx < cap)
        recordBuf[idx] = rec;
}
#endif

// Backward attribution: the radiance gathered from vertex k onward is
// (L_final - L_k) = beta_out_k·Li_k; dividing by beta_in_k folds in this
// vertex's f·cos/pdf so the stored weight ∝ the integrand the proposal should
// match. Clamp ≥ 0 (float noise). Both the megakernel dump and the wavefront
// emitter call THIS — the identical per-vertex training weight.
// DOC:contrib#core start
float3 recordContrib(float3 Lfinal, float3 L_k, float3 beta_in_k)
{
    float3 beta = max(beta_in_k, float3(1e-8));
    return max((Lfinal - L_k) / beta, float3(0.0));
}
// DOC:contrib#core end
