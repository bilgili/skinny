// Shared display tail for the wavefront resolve kernels. The staged backends
// write the linear running-mean into accumBuffer (binding 2); this mirrors
// main_pass.slang's post-accumulation display path — s=1 light-tracer splat
// composite, exposure, tonemap, sRGB, HUD overlay — and writes the tonemapped
// result into the offscreen/swapchain output image (binding 1). Without this
// the wavefront would populate only the linear accum image and the displayed /
// render_headless() output would be black.
//
// The focus-plane + furnace over-energy overlays in main_pass are
// editor/visualiser extras that need the primary ray / scene hit (not kept
// per-pixel in the wavefront) and are intentionally omitted here. The HUD and
// transform-gizmo overlays only need screen-space buffers (binding 3 / 22), so
// they ARE composited so the editor gizmo is visible in wavefront mode too.

import common;    // GizmoSegment (overlay line-segment type)
import tonemap;   // applyTonemap, linearToSRGB
import bindings;  // fc, outputBuffer (1), hudMask (3), gizmoSegments (22), lightSplatBuffer

void wfWriteDisplay(uint2 pixel, float3 accum)
{
#if !(defined(SKINNY_METAL) && defined(SKINNY_METAL_RECORDS))
    // s = 1 light-tracer (bdpt) composite. lightSplatBuffer accumulates
    // fixed-point Q22.10 radiance over all frames; divide by frames-so-far for
    // the running mean. Zero for the path integrator (buffer cleared on reset).
    // Compiled out of the Metal records build (slot cap, metal-record-drain):
    // records are path-only, where this term is identically zero.
    uint base = (pixel.y * fc.width + pixel.x) * 3u;
    float3 splatSum = float3(lightSplatBuffer[base + 0u],
                             lightSplatBuffer[base + 1u],
                             lightSplatBuffer[base + 2u]) * (1.0 / 1024.0);
    accum += splatSum / float(fc.accumFrame + 1u);
#endif

    float3 exposed = accum * exp2(fc.exposure);
    float3 display = linearToSRGB(applyTonemap(exposed, fc.tonemapMode));

    // HUD overlay (binding 3): white text, dark panel blend — matches main_pass.
    float hud = hudMask[pixel];
    if (hud > 0.75)
        display = float3(1.0);
    else if (hud > 0.0)
        display = lerp(display, float3(0.02, 0.02, 0.03), hud);

#if !(defined(SKINNY_METAL) && defined(SKINNY_METAL_RECORDS))
    // Transform-gizmo overlay (binding 22): same screen-space 2D line composite
    // as main_pass.slang so the editor gizmo draws in wavefront mode. The CPU
    // rebuilds + uploads the segment list (and fc.numGizmoSegments) every frame
    // regardless of execution mode, so the loop is a no-op when none is active.
    // Compiled out of the Metal records build (slot cap, metal-record-drain):
    // no gizmo overlay on online-training frames.
    float2 px = float2(float(pixel.x) + 0.5, float(pixel.y) + 0.5);
    for (uint gi = 0u; gi < fc.numGizmoSegments; ++gi)
    {
        GizmoSegment seg = gizmoSegments[gi];
        float2 ab = seg.b - seg.a;
        float ab2 = max(dot(ab, ab), 1e-6);
        float t = clamp(dot(px - seg.a, ab) / ab2, 0.0, 1.0);
        float2 closest = seg.a + ab * t;
        float d = length(px - closest);
        float aa = clamp(seg.width + 1.0 - d, 0.0, 1.0);
        display = lerp(display, seg.color, aa);
    }
#endif

    outputBuffer[pixel] = float4(display, 1.0);
}
