// Common types and utilities shared across all shader modules.
//
// References:
//   [1] O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically
//       Good Algorithms for Random Number Generation", HMC-CS-2014-0905, 2014.
//   [2] Veach, "Robust Monte Carlo Methods for Light Transport Simulation",
//       PhD thesis, Stanford University 1997.
//   [3] Walter, Marschner, Li, Torrance,
//       "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007.

static const float PI = 3.14159265358979;

// Radiance / throughput carrier width. Spectral build (-DSKINNY_SPECTRAL):
// float4 (4 hero wavelengths). RGB build: float3. Defined here — not only in
// spectrum.slang — so the carrier structs in interfaces.slang (BSDFSample /
// BounceResult) type-check in BOTH builds without importing the spectral module,
// keeping the RGB SPIR-V byte-identical. spectrum.slang (compiled only in the
// spectral variant) imports common for this alias rather than redefining it.
#if defined(SKINNY_SPECTRAL)
typealias Spectrum = float4;
static const int SPECTRUM_N = 4;
#else
typealias Spectrum = float3;
static const int SPECTRUM_N = 3;
#endif

struct Ray
{
    float3 origin;
    float3 direction;
};

struct HitInfo
{
    float  t;              // ray parameter at intersection (world-space units)
    float  backT;          // ray parameter at back-face exit (-1 = no back hit)
    float3 position;       // world-space hit position
    float3 positionObject; // object-space (BLAS-local) hit position — for 3D procedural textures
    float3 normal;         // world-space shading normal
    float3 tangent;        // surface tangent aligned with +U; bitangent = cross(N, T)
    float2 uv;
    bool   hit;
    bool   hasTangent;     // true when uv comes from an asset (not the fallback sphere map)
    uint   materialId;     // index into the active material set
    uint   instanceId;     // winning TLAS instance index (structural AOV / parity)
};

// Per-material flat-shading record. Used by main_pass.slang when a hit's
// materialTypes[id] == MATERIAL_TYPE_FLAT. Layout matches
// renderer.pack_flat_material exactly: 128 B / record, scalar-layout.
// float4-packed so the Vulkan scalar (`-fvk-use-scalar-layout`) and Metal MSL
// StructuredBuffer layouts agree: each `float3` + its trailing scalar is one
// float4 (.xyz + .w), 16 B on both targets. Scalar-only runs already match. The
// property getters keep every accessor site unchanged; the CPU byte layout
// (documented offsets) is byte-identical to the old packing. See mesh_head.slang
// for the same pattern + the rationale.
struct FlatMaterialParams
{
    float4 _diffuseColorRoughness;    // 0..16   xyz=diffuseColor, w=roughness
    float  metallic;                  // 16..20
    float  specular;                  // 20..24
    float  opacity;                   // 24..28
    uint   diffuseTextureIdx;         // 28..32; 0xFFFFFFFF = use constant
    uint   roughnessTextureIdx;       // 32..36; sentinel = use constant
    uint   metallicTextureIdx;        // 36..40; sentinel = use constant
    uint   normalTextureIdx;          // 40..44; sentinel = use mesh normal
    uint   emissiveTextureIdx;        // 44..48; sentinel = use emissive const
    float4 _emissiveColorIor;         // 48..64  xyz=emissiveColor, w=ior (1=vacuum)
    float  coat;                      // 64..68; clear coat weight (0..1)
    float  coatRoughness;             // 68..72
    float  coatIOR;                   // 72..76
    uint   opacityTextureIdx;         // 76..80; sentinel = use constant
    float4 _coatColorOpacityThreshold;// 80..96  xyz=coatColor, w=opacityThreshold
    // _normalScaleChannelMask: xyz = UsdUVTexture inputs:scale.xyz (OpenGL conv
    // (2,2,2)/(-1,-1,-1)); w = channelMask (per-input 4-bit channel selector,
    // _CHANNEL_SHIFT diffuse(0)/roughness(4)/metallic(8)/opacity(12)/emissive(16)).
    float4 _normalScaleChannelMask;   // 96..112 xyz=normalScale, w=channelMask
    float4 _normalBiasPad;            // 112..128 xyz=normalBias, w=_pad0
    // Stage-2 rich inputs (flat-lobes-rich-inputs). float4-wrapped so the MSL
    // and SPIR-V layouts stay byte-identical (no Metal repack). Defaults
    // reproduce the prior behavior: transmissionColor falls back to diffuseColor
    // in the packer, specularColor = white, diffuseRoughness = 0 ⇒ Lambert.
    float4 _transmissionColorDiffuseRough; // 128..144 xyz=transmissionColor, w=diffuseRoughness
    float4 _specularColorPad;              // 144..160 xyz=specularColor, w=_pad1
    // Subsurface medium (pbrt-subsurface-volumetric), packed here to avoid a new
    // SSBO (Metal 31-buffer cap). Only read when materialType == SUBSURFACE; zero
    // for every other flat material. σ in mm⁻¹ (renderer converts to scene units).
    // eta (boundary IOR) reuses `ior` (_emissiveColorIor.w). float4-wrapped for
    // Metal/Vulkan byte parity.
    float4 _mediumSigmaA_g;                // 160..176 xyz=σ_a, w=g
    float4 _mediumSigmaS_kind;             // 176..192 xyz=σ_s, w=asfloat(mediumKind)
    // World→uvw affine for the heterogeneous density grid
    // (nanovdb-volume-rendering): rows 0..2 of the math (column-vector) matrix
    // taking a world-space point to normalized [0,1]³ texture coords of the
    // volumeDensity texture (binding 26) — uvw[r] = dot(row[r].xyz, p) + row[r].w.
    // The CPU folds prim-xform⁻¹, the pbrt→USD axis flip, the grid's index→world
    // map, index_min, and the +0.5 voxel-center offset into these 12 floats.
    // Identity rows for every non-volume material (bytes only appended, never
    // shifted — the 0..192 prefix is byte-identical to before this change).
    float4 _worldToUvw0;                   // 192..208
    float4 _worldToUvw1;                   // 208..224
    float4 _worldToUvw2;                   // 224..240
    // Procedural cloud medium (pbrt-cloud-procedural-medium): pbrt CloudMedium
    // params, read only when mediumKind == MEDIUM_CLOUD (zeros elsewhere —
    // bytes appended, never shifted; the 0..240 prefix is byte-identical).
    // For the cloud the worldToUvw rows map world → medium-local [0,1]³
    // (identical semantics to the grid's texel space; no texture involved).
    float4 _cloudDensityWispinessFrequency; // 240..256 x=density y=wispiness z=frequency w=pad

    property float3 diffuseColor      { get { return _diffuseColorRoughness.xyz; } }
    property float  roughness         { get { return _diffuseColorRoughness.w; } }
    property float3 emissiveColor     { get { return _emissiveColorIor.xyz; } }
    property float  ior               { get { return _emissiveColorIor.w; } }
    property float3 coatColor         { get { return _coatColorOpacityThreshold.xyz; } }
    property float  opacityThreshold  { get { return _coatColorOpacityThreshold.w; } }
    property float3 normalScale       { get { return _normalScaleChannelMask.xyz; } }
    property uint   channelMask       { get { return asuint(_normalScaleChannelMask.w); } }
    property float3 normalBias        { get { return _normalBiasPad.xyz; } }
    // Named-glass Cauchy dispersion coefficient B (µm²) for a dispersive dielectric
    // (change spectral-rendering, Group 6.4): n(λ) = ior + B/λ_µm². 0 = constant-IOR
    // (non-dispersive). Rides the previously-unused _normalBiasPad.w (_pad0); only
    // the spectral dispersion path reads it, RGB build ignores it (byte-identical
    // CPU packing: the packer wrote 0 there before).
    property float  glassCauchyB      { get { return _normalBiasPad.w; } }
    property float3 transmissionColor { get { return _transmissionColorDiffuseRough.xyz; } }
    property float  diffuseRoughness  { get { return _transmissionColorDiffuseRough.w; } }
    property float3 specularColor     { get { return _specularColorPad.xyz; } }
    // Named-conductor identity (change spectral-rendering, Group 6.2): 0 = none
    // (RGB Schlick F0), 1..4 = au/ag/al/cu. Stored in the previously-unused
    // _specularColorPad.w (_pad1) — no struct growth. Only the spectral conductor
    // Fresnel reads it; the RGB build ignores it (byte-identical CPU packing: the
    // packer wrote 0 there before).
    property uint conductorMetalId    { get { return asuint(_specularColorPad.w); } }
    property float3 mediumSigmaA      { get { return _mediumSigmaA_g.xyz; } }
    property float  mediumG           { get { return _mediumSigmaA_g.w; } }
    property float3 mediumSigmaS      { get { return _mediumSigmaS_kind.xyz; } }
    property uint   mediumKind        { get { return asuint(_mediumSigmaS_kind.w); } }
    property float4 worldToUvw0       { get { return _worldToUvw0; } }
    property float4 worldToUvw1       { get { return _worldToUvw1; } }
    property float4 worldToUvw2       { get { return _worldToUvw2; } }
    property float3 cloudParams       { get { return _cloudDensityWispinessFrequency.xyz; } }
};

// Participating medium resolved at a hit (homogeneous subsurface interior or a
// heterogeneous free-standing grid). The random walk reads the medium ONLY
// through this + the density seam (densityAt / mediumMajorant), so adding a
// kind is purely additive. `worldToUvw0..2` are the folded world→texture rows
// for MEDIUM_NANOVDB (identity for HOMOGENEOUS, where densityAt never reads
// them) — they replaced the formerly reserved `gridHandle`.
struct Medium
{
    float3 sigmaA;       // mm⁻¹ (scene scale + grid value max folded at pack time)
    float3 sigmaS;       // mm⁻¹
    float  g;            // Henyey-Greenstein anisotropy
    float  eta;          // boundary relative IOR (1 = index-matched pass-through)
    uint   kind;         // MEDIUM_HOMOGENEOUS | MEDIUM_NANOVDB | MEDIUM_CLOUD
    float4 worldToUvw0;  // uvw.x = dot(.xyz, p) + .w  (MEDIUM_NANOVDB: texel
    float4 worldToUvw1;  // uvw.y   space; MEDIUM_CLOUD: medium-local [0,1]³)
    float4 worldToUvw2;  // uvw.z
    float3 cloudParams;  // MEDIUM_CLOUD: x=density y=wispiness z=frequency
};

// Gen-reflected skin material UBO — layout produced by MaterialXGenSlang
// running on M_skinny_skin_default. 164 scalar-layout bytes / 27 fields.
// The renderer fills this each frame via materialx_runtime.pack_material_values
// using SkinParameters values mapped through Renderer._mtlx_skin_overrides.
// Compiled with -fvk-use-scalar-layout: float3 has 4-byte alignment (no
// 16-byte promotion), so the struct packs tightly.
struct MtlxSkinParams
{
    float  layer_top_melanin;             // +0
    float  layer_top_thickness;           // +4
    float3 layer_top_scattering_coeff;    // +8
    float  layer_top_anisotropy;          // +20
    float  layer_top_ior;                 // +24
    float  layer_middle_hemoglobin;       // +28
    float  layer_middle_blood_oxygenation;// +32
    float  layer_middle_thickness;        // +36
    float3 layer_middle_scattering_coeff; // +40
    float  layer_middle_anisotropy;       // +52
    float  layer_middle_ior;              // +56
    float4 layer_middle_pigment;          // +60
    float  layer_bottom_thickness;        // +76
    float3 layer_bottom_absorption;       // +80
    float3 layer_bottom_scattering;       // +92
    float  layer_bottom_anisotropy;       // +104
    float  layer_bottom_ior;              // +108
    float4 skin_bsdf_pigment;             // +112
    float  skin_bsdf_weight;              // +128
    float  skin_bsdf_roughness;           // +132
    float  skin_bsdf_ior;                 // +136
    float  skin_bsdf_pore_density;        // +140
    float  skin_bsdf_pore_depth;          // +144
    float  skin_bsdf_hair_density;        // +148
    float  skin_bsdf_hair_tilt;           // +152
    float  skin_surface_opacity;          // +156
    uint   skin_surface_thin_walled;      // +160
};

// float4-packed so the CPU scalar bytes are read identically on Metal. Under
// the Vulkan scalar layout a `float3` is 4-aligned (12 B), but MSL 16-aligns and
// pads it to 16 B — so a `float3 v0; float pad;` pair, which the CPU packs as
// v0@0/pad@12/next@16, would be read by Metal as v0@0/pad@16/next@32 (every later
// field shifted). Wrapping each `float3 + scalar` into one `float4` makes the
// layout 16-aligned on BOTH backends with byte-identical CPU packing; read-only
// `property` getters keep every accessor site unchanged. (Same recipe as
// BvhNode/MeshVertex/FlatMaterialParams.)
struct EmissiveTriangle
{
    float4 _v0;              // xyz = v0,  w = cw  (cumulative power CDF, inclusive)
    float4 _v1;              // xyz = v1,  w = pSel (per-triangle selection prob p_i)
    float4 _v2;              // xyz = v2
    float4 _emissionArea;    // xyz = emission, w = area

    property float3 v0       { get { return _v0.xyz; } }
    property float3 v1       { get { return _v1.xyz; } }
    property float3 v2       { get { return _v2.xyz; } }
    property float3 emission { get { return _emissionArea.xyz; } }
    property float  area     { get { return _emissionArea.w; } }
    // Power-weighted NEE selection (change emissive-mesh-nee). The host packs a
    // normalized cumulative-power CDF inline into the otherwise-unused .w lanes
    // (no separate buffer → no extra Metal argument-table slot). `cw` is the
    // inclusive cumulative weight Σ_{j≤i} w_j / Σw (cw of the last triangle = 1),
    // used for binary-search selection; `pSel` = w_i / Σw is the per-triangle
    // selection probability consumed as the light's selectionPdf.
    property float  cw       { get { return _v0.w; } }
    property float  pSel     { get { return _v1.w; } }
};

// Distant (directional) light record. `direction` points TOWARD the light
// (= -L for a surface normal n; same convention as the legacy uniform).
// Mirrors UsdLux.DistantLight: radiance is color × intensity × 2^exposure.
struct DistantLight
{
    float4 _direction;       // xyz = direction, w = spectral SPD slot (-1 = none)
    float4 _radiance;        // xyz = radiance

    property float3 direction { get { return _direction.xyz; } }
    property float3 radiance  { get { return _radiance.xyz; } }
    // Authored illuminant SPD slot (spectral mode, Group 6.3); < 0 ⇒ upsample the
    // RGB radiance instead. Ignored by the RGB build (the lane packs 0.0 there).
    property float  spdIndex  { get { return _direction.w; } }
};

struct SphereLight
{
    float4 _positionRadius;  // xyz = position, w = radius
    float4 _radiance;        // xyz = radiance

    property float3 position { get { return _positionRadius.xyz; } }
    property float  radius   { get { return _positionRadius.w; } }
    property float3 radiance  { get { return _radiance.xyz; } }
};

struct GizmoSegment
{
    float2 a;       // screen-space pixel coords (origin at bottom-left)
    float2 b;
    float3 color;   // linear RGB
    float  width;   // half-width in pixels
};

struct Camera
{
    float4x4 viewInverse;
    float4x4 projInverse;
    float4x4 view;          // forward view matrix (world → camera space)
    float4x4 proj;          // forward projection matrix (camera → clip)
    float3   position;
    float    fov;
};

struct FrameConstants
{
    Camera camera;
    uint   frameIndex;
    uint   accumFrame;      // running count for progressive accumulation (0 = reset)
    float  time;
    uint   width;
    uint   height;
    uint   numDistantLights;  // active count in distantLights buffer (binding 25); 0 = IBL only
    uint   useMesh;         // always 1 — triangle mesh pipeline (mesh_head.slang)
    float  tattooDensity;   // 0 = no ink; 1 = full pigment load at alpha-1 texels
    float  envIntensity;    // scalar on every sampleEnvironment() result
    uint   furnaceMode;     // 1 = energy-conservation test: unit sphere + white env + pink on violation
    float  mmPerUnit;       // world-scale factor: 1 world unit = mmPerUnit millimetres. Used to
                            // reconcile σ (mm⁻¹) with ray distances (world units) in the volume march.
    uint   detailFlags;     // bit0 = master enable; bit1 = normal avail; bit2 = roughness avail; bit3 = disp avail; bit4 = normals pre-baked into vertex data
    float  normalMapStrength;    // 0..1 scale on tangent-space XY before rebuilding +Z
    float  displacementScaleMM;  // millimetres per (disp - 0.5); 0 disables displacement offset
    uint   numInstances;         // size of the TLAS instance list; 0 disables mesh ray traversal
    uint   numSphereLights;      // active count in sphereLights buffer (binding 17)
    uint   numEmissiveTriangles; // active count in emissiveTriangles buffer (binding 18)
    uint   integratorType;       // 0 = path tracer, 1 = BDPT (flat first-hit), 2 = SPPM (wavefront-only)
    uint   numGizmoSegments;     // active count in gizmoSegments buffer (binding 22)
    uint   numLensElements;      // active count in lensElements buffer (binding 23); 0 = pinhole
    float  filmDistance;         // world-space distance from film plane to rear lens surface
    float  rearZ;                // world-space z of rear surface in lens-local frame
    float  rearAperture;         // world-space half-aperture (radius) of the rear element
    float  frontZ;               // world-space z of front (scene-side) surface; used to anchor
                                 // the lens so its front element sits at fc.camera.position
    float  filmHalfH;            // half the sensor height in world units (verticalAperture/2);
                                 // governs framing in the lens path (camera.fov is ignored there)
    float  emissiveTotalPower;   // Σ(area · Rec709-luminance(emission)) over all emissive
                                 // triangles. The path tracer's BSDF-hit MIS weight reads it to
                                 // reconstruct the NEE solid-angle pdf at a light hit without the
                                 // triangle index (area cancels): pdfLightSA = lum·d² / (Σ·cosLight).
                                 // Reuses the retired `irisZ` slot ⇒ no UBO byte-layout change.
    uint   numPupilBounds;       // active count in exitPupilBounds buffer (binding 24)
    float  filmDiagRadiusW;      // half of the film-plane diagonal in world units; bin index =
                                 //   floor(length(pFilm.xy) / filmDiagRadiusW · numPupilBounds)
    uint   focusOverlay;         // 0 = off; 1 = composite the translucent focal-plane visualiser
    float3 focusPlaneOrigin;     // world-space point on the focal plane (camera + forward·focusDist)
    float3 focusPlaneNormal;     // unit world-space normal of the focal plane
    float2 zoomMin;              // viewport sub-rect lower-left in [0, 1]² normalised pixel coords
    float2 zoomMax;              // viewport sub-rect upper-right; (0,0)→(1,1) = no zoom
    uint   lensVignetteDebug;    // 1 = render the per-pixel lens-ray success rate as grayscale
                                 //     instead of the integrator's radiance (black corners ⇒
                                 //     vignetting, full white ⇒ all rays survive)
    // BXDF visualizer scene-pick. When pickArmed != 0 the main pass writes
    // the HitInfo of the pixel matching pickPixel into toolBuffer (binding
    // 30). One-shot: the CPU clears pickArmed after a single read so the
    // shader stops writing on subsequent frames.
    uint2  pickPixel;
    uint   pickArmed;
    // Display-side exposure (EV stops, applied as 2^EV multiplier in the
    // main pass before tonemapping) and tonemap operator selector. These
    // are post-process knobs: changing them does not reset accumulation.
    //   0 = ACES filmic (Narkowicz fit)
    //   1 = Reinhard (x / (1 + x))
    //   2 = Hable / Uncharted 2 filmic
    //   3 = Linear (clamp only)
    float  exposure;
    uint   tonemapMode;
    // ── Pluggable scene-sampling seam (sampling/proposal.slang) ──
    // proposalMask: bit k = directional proposal k active at the bounce
    //   (bit0 = BSDF, bit1 = environment-importance). reuseMode: 0 = identity
    //   NEE/indirect (ReSTIR later). proposalAlpha: normalized one-sample-MIS
    //   selection weights, one per slot (Σ = 1). Scalar layout — these append
    //   tightly after tonemapMode; the Python packer in renderer._pack_uniforms
    //   writes them in this exact order.
    uint   proposalMask;
    uint   reuseMode;
    float4 proposalAlpha;
    // Per-lobe sampler selection for the flat/std_surface BSDF — 8 bits per
    // lobe: coat | spec<<8 | diff<<16 (0 = native; see flat_lobes.slang
    // FLAT_SAMPLER_*). flat_material.slang unpacks via flatLobeSamplerId. Scalar
    // tail — the Python packer in renderer._pack_uniforms appends it after
    // proposalAlpha in this exact order. No new descriptor binding.
    uint   flatLobeSamplers;
    // ── Neural directional proposal (sampling/neural_proposal.slang) ──
    // Scene AABB (world space) for the condition's position normalisation, plus
    // the active frozen-net version (baseline 0 — the per-sample networkVersion
    // hook for online training). Scalar tail: _pack_uniforms appends these after
    // flatLobeSamplers in this exact order. Read only when the NEURAL proposal
    // bit is active, so the default {bsdf} path stays bit-identical.
    float3 sceneBoundsMin;
    float3 sceneBoundsExtent;
    uint   neuralNetworkVersion;
    // ── Wavefront-native path-record emission (wavefront/wf_records.slang) ──
    // 0 = off (default render is byte-identical: no per-lane record stack
    // writes, no record appends). Non-zero while online neural training drives
    // the wavefront record drain — the wavefront shade/terminate kernels then
    // push a per-lane vertex stack and splat PathRecords to bindings 36/37 at
    // path termination. Scalar tail: _pack_uniforms appends it last in this
    // exact order. Read only by the wavefront path kernels.
    uint   recordMode;
    // ── Improper (mirrored) pbrt camera (change pbrt-mirrored-camera-flip) ──
    // 1 ⇒ the camera-to-world transform was orientation-reversing (negative
    // determinant, e.g. a `Scale -1 1 1` before LookAt). zoomedNDC negates
    // ndc.x to reproduce pbrt's horizontal screen mirror; sampleWi mirrors the
    // inverse pixel map to keep BDPT/light-tracing consistent. 0 ⇒ the render is
    // byte-identical (ndc.x untouched). Scalar tail: _pack_uniforms appends it
    // last in this exact order.
    uint   cameraMirror;
    // ── SPPM (change photon-mapping-sppm) ── per-pass photon-mapping params,
    // read only by the wavefront SPPM kernels (integratorType == 2). Scalar tail
    // appended last by _pack_uniforms in this exact order.
    float  sppmInitialRadius;   // initial per-pixel search radius r_0
    float  sppmCellSize;        // spatial-hash cell size (== sppmInitialRadius)
    uint3  sppmGridRes;         // grid resolution per axis (sceneBoundsExtent/cellSize)
    uint   sppmPhotonsEmitted;  // photons emitted per pass (the 1/N estimator divisor)
    float  sppmGlossyContinueRoughness;  // SPPM eye-walk: continue (don't store a VP) through a metallic lobe whose roughness is below this (change sppm-glossy-final-gather); 0 = delta-only (PM-1)
    // ── Film per-sample radiance clamp (change film-maxcomponent-clamp) ──
    // pbrt `Film "float maxcomponentvalue"`: each sample's radiance is scaled
    // before accumulation so its largest RGB component does not exceed this
    // threshold (hue-preserving), matching pbrt's RGBFilm firefly suppression.
    // 0 = disabled (the clamp is a no-op and the render is byte-identical).
    // Scalar tail appended last by _pack_uniforms in this exact order.
    float  filmMaxComponent;
    // ── SPPM photon-emission group pmf (change sppm-power-proportional-photon-groups) ──
    // Selection probability per emission group (emissive triangles, sphere
    // lights, distant lights, environment), proportional to each group's
    // emitted power and normalized host-side (renderer._sppm_photon_group_pmf;
    // uniform-over-present fallback when total power is 0/non-finite). Zeros
    // when the integrator is not SPPM. sppmEmitPhoton walks these as a CDF and
    // divides each branch's flux by its own entry — replacing the uniform 1/G
    // whose flux disparity produced sparse huge env splats. Scalar tail: the
    // Python packer appends them after filmMaxComponent in this exact order.
    float  sppmGroupPmfE;
    float  sppmGroupPmfS;
    float  sppmGroupPmfD;
    float  sppmGroupPmfEnv;
    // ── Metal megakernel watchdog tiling (change metal-megakernel-watchdog-tiling) ──
    // Screen-space Y offset of the row band this megakernel command buffer covers.
    // Under `SKINNY_METAL` the megakernel frame is committed as a sequence of row
    // bands (one command buffer each) so no single buffer exceeds the macOS GPU
    // watchdog budget; `mainImage`/`mainImageRecord` add this to the dispatch
    // thread's y so the band addresses the right pixels. The field itself is
    // `#if defined(SKINNY_METAL)`-gated so the Vulkan `FrameConstants` struct is
    // unchanged and the SPIR-V stays byte-identical; the Python `_pack_uniforms`
    // still appends a trailing u32 (0) on both backends, which lands harmlessly in
    // the oversized Vulkan UBO and at the reflected offset on Metal.
#if defined(SKINNY_METAL)
    uint   tileOriginY;
#endif
};

/// Per-sample radiance clamp matching pbrt `RGBFilm::AddSample`. When
/// `maxc > 0` and the sample's largest RGB component `m` exceeds it, scale the
/// whole triple by `maxc / m` (hue-preserving). `maxc <= 0` ⇒ unchanged, so the
/// default render path is byte-identical. The `!(m > maxc)` guard also leaves a
/// NaN sample untouched (it is sanitised by the accumulator as before).
float3 clampSampleRadiance(float3 L, float maxc)
{
    if (maxc <= 0.0)
        return L;
    float m = max(L.x, max(L.y, L.z));
    if (!(m > maxc))
        return L;
    return L * (maxc / m);
}

/// Remap an integer pixel coordinate to NDC ∈ [-1, +1]² accounting for
/// the current zoom-rect. When the zoom rect is the full frame this
/// reduces to the canonical `pixel/dim · 2 − 1` mapping; otherwise the
/// sub-rect is stretched to fill the output. Use this everywhere an
/// integrator needs a primary-ray NDC.
float2 zoomedNDC(FrameConstants fc, uint2 pixel, float2 jitter)
{
    float2 uv = (float2(pixel) + 0.5 + jitter) / float2(fc.width, fc.height);
    float2 sub = lerp(fc.zoomMin, fc.zoomMax, uv);
    float2 ndc = sub * 2.0 - 1.0;
    ndc.y = -ndc.y;
    // Improper (mirrored) pbrt camera: reproduce pbrt's horizontal screen mirror
    // (change pbrt-mirrored-camera-flip). Single chokepoint — every primary-ray
    // generator (pinhole + thick_lens, megakernel + wavefront) calls zoomedNDC.
    if (fc.cameraMirror != 0)
        ndc.x = -ndc.x;
    return ndc;
}

/// One spherical interface in a thick-lens stack (PBRT § 6.4).
/// Packed as float4 (radius, thickness, ior, halfAperture) — see
/// renderer.py::_sync_lens_buffer. Matches PBRT-v3's
/// `LensElementInterface` exactly: `thickness` is the axial distance
/// to the *next surface toward the film* (the rearmost element's
/// thickness is the film distance). `ior` is the medium *between this
/// surface and the next one toward the film*. `radius == 0` means a
/// planar interface (the aperture stop).
struct LensSurface
{
    float radius;
    float thickness;
    float ior;
    float halfAperture;
};

static const uint INTEGRATOR_PATH = 0u;
static const uint INTEGRATOR_BDPT = 1u;
static const uint INTEGRATOR_SPPM = 2u;  // Stochastic Progressive Photon Mapping (wavefront-only)

// ── Random number generation (PCG) ──────────────────────────────

/// PCG-32 hash — one step of the PCG random-number generator [1].
///
///   state = input · 747796405 + 2891336453            (LCG advance)
///   word  = ((state >> ((state >> 28) + 4)) ^ state) · 277803737   (output XSH-RR)
///   hash  = (word >> 22) ^ word
///
/// Produces a statistically high-quality 32-bit hash with period 2³².
uint pcgHash(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct RNG
{
    uint state;

    [mutating]
    float next()
    {
        state = pcgHash(state);
        return float(state) / float(0xFFFFFFFFu);
    }

    [mutating]
    float2 next2()
    {
        return float2(next(), next());
    }
};

RNG createRNG(uint2 pixel, uint frame)
{
    RNG rng;
    rng.state = pcgHash(pixel.x + pixel.y * 65536u + frame * 16777259u);
    return rng;
}

// ── Sampling helpers ────────────────────────────────────────────

/// Cosine-weighted hemisphere sampling via Malley's method.
///
///   φ = 2π·u₁
///   cosθ = √u₂          → projection of uniform disk sample onto hemisphere
///   sinθ = √(1 − u₂)
///
/// Probability density:  p(ω) = cosθ / π
/// Used as the diffuse-BRDF importance sampler: weight = f_d · cosθ / p = 1/π · π = 1.
float3 sampleCosineHemisphere(float2 u, float3 normal)
{
    float phi      = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);

    // Build orthonormal basis around normal
    float3 tangent   = abs(normal.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 bitangent = normalize(cross(normal, tangent));
    tangent          = cross(bitangent, normal);

    return normalize(
        tangent * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        normal * cosTheta
    );
}

/// PDF of cosine-weighted hemisphere sampling.
///
///   p(ω) = (n · ω) / π
///
/// where  n · ω = max(dot(normal, dir), 0)
float cosineHemispherePdf(float3 dir, float3 normal)
{
    float nDotW = max(dot(normalize(dir), normalize(normal)), 0.0);
    return nDotW / PI;
}

/// Uniform sphere sampling via equal-area mapping.
///
///   z   = 1 − 2·u₁              (uniform in [−1, 1])
///   r   = √(1 − z²)
///   φ   = 2π·u₂
///   dir = (r·cosφ, z, r·sinφ)
///
/// PDF: p(ω) = 1 / (4π)
float3 sampleUniformSphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * u.y;
    return float3(r * cos(phi), z, r * sin(phi));
}

/// PDF of uniform sphere sampling.
///
///   p(ω) = 1 / (4π)
float uniformSpherePdf()
{
    return 1.0 / (4.0 * PI);
}

/// GGX Smith G₁ masking term (single-direction) [3].
///
///   G₁(v, α) = 2·(n·v) / ((n·v) + √(α² + (1 − α²)·(n·v)²))
///
/// where  α  = roughness²  (Disney remap)
///        v  = view / light direction
///        n·v = dot(shading normal, v)
/// The separable product G₁(v)·G₁(l) approximates the joint Smith G₂.
float ggxSmithG1(float NdotV, float roughness)
{
    float a = roughness * roughness;
    float k = a / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

/// Power heuristic for multiple importance sampling (MIS) [2].
///
///   w(pₐ, p_b) = pₐ² / (pₐ² + p_b²)       (β = 2)
///
/// Combines two sampling strategies A and B so their sum is an unbiased
/// estimator of the integrand with near-optimal variance reduction.
/// The β = 2 exponent minimises expected squared error (Veach §9.2).
float powerHeuristic(float pdfA, float pdfB)
{
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / max(a2 + b2, 1e-12);
}

void buildBasis(float3 n, out float3 t, out float3 b)
{
    t = abs(n.x) > 0.9 ? float3(0, 1, 0) : float3(1, 0, 0);
    b = normalize(cross(n, t));
    t = cross(b, n);
}

float3 worldToTangent(float3 dir, float3 T, float3 B, float3 N)
{
    return float3(dot(dir, T), dot(dir, B), dot(dir, N));
}

float3 tangentToWorld(float3 dir, float3 T, float3 B, float3 N)
{
    return T * dir.x + B * dir.y + N * dir.z;
}

float3 sampleCosineHemisphereTS(float2 u)
{
    float phi      = 2.0 * PI * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0 - u.y);
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float cosineHemispherePdfTS(float3 wi)
{
    return max(wi.z, 0.0) / PI;
}

/// Uniform disk sampling via concentric mapping.
///
///   r   = √u₁              (area-preserving radial CDF inversion)
///   φ   = 2π·u₂
///   p   = (r·cosφ, r·sinφ)
///
/// PDF: p(x, y) = 1 / π  (uniform over unit disk)
float2 sampleUniformDisk(float2 u)
{
    float r = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

// Camera ray generation lives in the ICamera implementations under
// `cameras/` (pinhole.slang and thick_lens.slang). main_pass.slang
// tag-switches on `fc.numLensElements` and dispatches through the
// interface; common.slang stays free of the lens-buffer binding.
