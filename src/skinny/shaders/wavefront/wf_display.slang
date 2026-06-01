// Shared display tail for the wavefront resolve kernels. The staged backends
// write the linear running-mean into accumBuffer (binding 2); this mirrors
// main_pass.slang's post-accumulation display path — s=1 light-tracer splat
// composite, exposure, tonemap, sRGB, HUD overlay — and writes the tonemapped
// result into the offscreen/swapchain output image (binding 1). Without this
// the wavefront would populate only the linear accum image and the displayed /
// render_headless() output would be black.
//
// The focus-plane + furnace over-energy + gizmo overlays in main_pass are
// editor/visualiser extras (need the primary ray / hit / gizmo buffer) and are
// intentionally omitted here.

import tonemap;   // applyTonemap, linearToSRGB
import bindings;  // fc, outputBuffer (1), hudMask (3), lightSplatBuffer

void wfWriteDisplay(uint2 pixel, float3 accum)
{
    // s = 1 light-tracer (bdpt) composite. lightSplatBuffer accumulates
    // fixed-point Q22.10 radiance over all frames; divide by frames-so-far for
    // the running mean. Zero for the path integrator (buffer cleared on reset).
    uint base = (pixel.y * fc.width + pixel.x) * 3u;
    float3 splatSum = float3(lightSplatBuffer[base + 0u],
                             lightSplatBuffer[base + 1u],
                             lightSplatBuffer[base + 2u]) * (1.0 / 1024.0);
    accum += splatSum / float(fc.accumFrame + 1u);

    float3 exposed = accum * exp2(fc.exposure);
    float3 display = linearToSRGB(applyTonemap(exposed, fc.tonemapMode));

    // HUD overlay (binding 3): white text, dark panel blend — matches main_pass.
    float hud = hudMask[pixel];
    if (hud > 0.75)
        display = float3(1.0);
    else if (hud > 0.0)
        display = lerp(display, float3(0.02, 0.02, 0.03), hud);

    outputBuffer[pixel] = float4(display, 1.0);
}
