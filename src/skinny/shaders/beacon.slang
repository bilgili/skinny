// beacon.slang — GPU kernel-beacon write (change metal-kernel-beacon).
//
// Compiled ONLY under SKINNY_METAL_TRACE. With the gate off the whole file is
// empty and no entry imports it, so the Metal argument table and the Vulkan
// SPIR-V stay byte-identical to production. Under the gate, every compute entry
// calls beaconStore(KID_<entry>) as its first statement, writing its static
// kernel id into the UMA beacon buffer so a wedge confirms which kernel started
// on the device.
//
// The 256-byte cell layout mirrors host `metal_beacon.py`:
//   offset 8 = kernel_id (u32), 12 = phase (u32), 16 = trip (u32).
// This change writes kernel_id with phase 0 and trip 0; the reserved 3-arg
// overload carries the per-loop phase/trip a follow-up will drive.
#if defined(SKINNY_METAL_TRACE)

import bindings;   // gKernelBeacon (declared under SKINNY_METAL_TRACE)

// Static kernel ids — MUST equal host metal_beacon.KERNEL_NAMES. Naming each
// constant makes a shader typo an undefined-symbol compile error, never a
// silent wrong id.
static const uint KID_mainImage       = 1;
static const uint KID_wfPathGenerate  = 2;
static const uint KID_wfPathIntersect = 3;
static const uint KID_wfBuildArgs     = 4;
static const uint KID_wfScatter       = 5;
static const uint KID_wfPathShadeFlat = 6;
static const uint KID_wfPathResolve   = 7;
static const uint KID_wfPathShade     = 8;
static const uint KID_wfSppmEye              = 9;
static const uint KID_wfSppmGridCount        = 10;
static const uint KID_wfSppmGridScanBlock    = 11;
static const uint KID_wfSppmGridScanBlockSums = 12;
static const uint KID_wfSppmGridScanAdd      = 13;
static const uint KID_wfSppmGridScatter      = 14;
static const uint KID_wfSppmPhotonTrace      = 15;
static const uint KID_wfSppmUpdate           = 16;
static const uint KID_wfMltBootstrap = 17;
static const uint KID_wfMltInit      = 18;
static const uint KID_wfMltMutate    = 19;
static const uint KID_wfMltResolve   = 20;
static const uint KID_wfBdptClassify    = 21;
static const uint KID_wfBdptBuildArgs   = 22;
static const uint KID_wfBdptScatter     = 23;
static const uint KID_wfBdptConnectNee  = 24;
static const uint KID_wfBdptConnectFull = 25;
static const uint KID_wfBdptResolve     = 26;
static const uint KID_wfBdptGenEye      = 27;
static const uint KID_wfBdptWalkClassify = 28;
static const uint KID_wfBdptBounceEye   = 29;
static const uint KID_wfBdptWalk        = 30;
static const uint KID_wfBdptLightTail   = 31;
static const uint KID_wfBdptGenLight    = 32;
static const uint KID_wfBdptBounceLight = 33;
static const uint KID_wfBdptSplat       = 34;
static const uint KID_wfNeuralProposal = 35;
static const uint KID_wfIndirectPaint  = 36;
static const uint KID_previewMain = 37;
static const uint KID_clearImage  = 38;
static const uint KID_clearDepth  = 39;
static const uint KID_depthLines  = 40;
static const uint KID_colorLines  = 41;
static const uint KID_blendTris   = 42;

// Write the kernel id (phase 0, trip 0) into the UMA beacon buffer.
void beaconStore(uint kernelId)
{
    gKernelBeacon.Store(8u, kernelId);
    gKernelBeacon.Store(12u, 0u);
    gKernelBeacon.Store(16u, 0u);
}

// Reserved for the follow-up: per-loop phase + trip count.
void beaconStore(uint kernelId, uint phase, uint trip)
{
    gKernelBeacon.Store(8u, kernelId);
    gKernelBeacon.Store(12u, phase);
    gKernelBeacon.Store(16u, trip);
}

#endif  // SKINNY_METAL_TRACE
