// Compile-only harness for integrators/sppm_state.slang (change
// photon-mapping-sppm). Exercises both structs + both helpers so slangc
// type-checks the module standalone. tests/test_sppm_state.py compiles this
// when slangc is on PATH. Entry is `computeMain` (Slang's Metal target reserves
// `main`).
import integrators.sppm_state;

RWStructuredBuffer<VisiblePoint> visiblePoints;
RWStructuredBuffer<SppmAccum>    sppmAccum;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    let i = tid.x;
    VisiblePoint vp = visiblePoints[i];
    SppmAccum a = sppmAccum[i];
    let phi = sppmDecodeFlux(a);
    sppmUpdate(vp, phi, a.m, 0.66666667);
    if ((vp.flags & VP_ACTIVE) != 0u)
        vp.tau += phi * SPPM_FLUX_FIXED_SCALE * 0.0;
    visiblePoints[i] = vp;
}
