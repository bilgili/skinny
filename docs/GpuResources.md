# Skinny — GPU Resources, Bindings, and Byte Layouts

This document is the authoritative map of what the GPU sees: the descriptor
binding map, the resource inventory that owns every allocation and binding,
the host-mirrored byte layouts, the shader build-variant key, and the
`FrameConstants` layout.

Change a binding, a struct field, or a build define, and the answer is here.
For the renderer overview see [Architecture.md](Architecture.md).

---

## Descriptor Binding Map

The numbers below are authoritative and `bindings.slang` declares the same ones.
On the **host** side each is stated once more, next to the allocation it belongs
to, in `src/skinny/gpu_resources.py` — see
[GPU resource inventory](#gpu-resource-inventory-gpu_resourcespy-change-renderer-gpu-resource-set).
That module does not derive these numbers; it relocates them so a resource's
binding, its size and its destruction sit together.

| Binding | Type | Content | Owner |
|---------|------|---------|-------|
| 0 | UBO | FrameConstants + SkinParams + light uniforms | `bindings.slang` |
| 1 | RWTexture2D | Offscreen output (RGBA8) — always `_offscreen_output`, on every path; the windowed path blits it to the acquired swapchain image rather than rebinding | `bindings.slang` |
| 2 | RWTexture2D | HDR accumulation (RGBA32F) | `bindings.slang` |
| 3 | RWTexture2D | HUD alpha mask (R8) — filled + copied by `Renderer._sync_hud_overlay`, which every render path calls so no path composites a previous frame's mask | `bindings.slang` |
| 4 | Sampler2D | HDR environment map (1024×512) | `environment.slang` |
| 5 | StructuredBuffer | Mesh vertices (32 B each) | `mesh_head.slang` |
| 6 | StructuredBuffer | Mesh indices (uint32) | `mesh_head.slang` |
| 7 | StructuredBuffer | BVH nodes (32 B each) | `mesh_head.slang` |
| 8 | Sampler2D | Tattoo map (512×512 RGBA) | `materials/skin/skin_shading.slang` |
| 9 | Sampler2D | Normal detail map (2048²) | `materials/skin/skin_shading.slang` |
| 10 | Sampler2D | Roughness detail map (2048²) | `materials/skin/skin_shading.slang` |
| 11 | Sampler2D | Displacement detail map (2048²) | `materials/skin/skin_shading.slang` |
| 12 | StructuredBuffer | TLAS instances (144 B each) | `mesh_head.slang` |
| 13 | StructuredBuffer | FlatMaterialParams (256 B each, scalar layout — `transmissionColor`@128, `diffuseRoughness`@140, `specularColor`@144 for the Stage-2 rich-input lobes; the subsurface/volume medium is packed inline at `σ_a`@160, `g`@172, `σ_s`@176, `mediumKind`@188 — `MEDIUM_HOMOGENEOUS`/`MEDIUM_NANOVDB`/`MEDIUM_CLOUD`, boundary `eta` reuses `ior`@60 — the world→uvw affine rows at 192..240 (`nanovdb-volume-rendering`), and the procedural-cloud density/wispiness/frequency float4 at 240..256 (`pbrt-cloud-procedural-medium`) — so neither medium kind needs a new buffer under Metal's 31-buffer cap, read via `resolveMedium(matId)`) | `bindings.slang` |
| 14 | Sampler2D[128] | Bindless material textures (PARTIALLY_BOUND) | `bindings.slang` |
| 15 | StructuredBuffer | MtlxSkinParams (164 B each, scalar layout) | `materials/skin/skin_shading.slang` |
| 16 | StructuredBuffer | Material type code + scatter + furnace + graph slot + python id (uint32 each) | `bindings.slang` |
| 17 | StructuredBuffer | SphereLight (32 B each) | `scene_lights.slang` |
| 18 | StructuredBuffer | EmissiveTriangle (64 B each); **dynamically sized** to the actual emissive-triangle count (no 256 cap — grows + rebinds like `material_capacity`). The power-weighted NEE selection CDF is packed **inline** in each record's spare `.w` lanes (`cw` = cumulative-power CDF, `pSel` = per-triangle prob) — no separate buffer / Metal slot (change `emissive-mesh-nee`) | `scene_lights.slang` |
| 19 | StructuredBuffer | StdSurfaceParams (256 B each) — raster `preview_pass` only; the path-traced / BDPT flat BSDF uses the `flat_lobes` model, not `evalStdSurfaceBSDF` | `bindings.slang` |
| 20 | StructuredBuffer | DistantLight (analytic distant lights) | `scene_lights.slang` |
| 21 | RWStructuredBuffer | BDPT light-splat buffer (Q22.10 uint per R/G/B) | `bindings.slang` |
| 22 | StructuredBuffer | Transform-gizmo line segments | `gizmo.py` |
| 23 | StructuredBuffer | Lens elements (thick-lens stack, float4) | `cameras/thick_lens.slang` |
| 24 | StructuredBuffer | Per-radius exit-pupil bounds (float4) | `cameras/thick_lens.slang` |
| 25 | ByteAddressBuffer | **Combined** MaterialX nodegraph params `graphParamsCombined` — ONE matId-major byte buffer shared by every scene graph, read `Load<GraphParams_X>(matId * GRAPH_PARAM_STRIDE)` (scalar layout, identical Metal/SPIR-V). Replaces the former one-`StructuredBuffer`-per-graph at 25..25+N−1, so graph count no longer grows the Metal argument table (change `combine-graph-param-buffers`). **Packing invariant:** `Load<GraphParams_X>` reads the *emitted* struct — only the uniforms referenced by the graph body — laid out **contiguously from 0**, so `generate_for_compute` re-compacts the kept `uniform_block` offsets dense-from-0 before `pack_uniform_block` writes them. Leaking the gen's full-block offsets (which carry a hole wherever an unused uniform was dropped) skews every field by that gap — an `<image>` graph then misreads `uv_scale` as `(0,1)` and collapses the U coord (`tests/test_materialx_graph.py::test_graph_uniform_offsets_are_dense`) | `generated_materials.slang` |
| 26 | Sampler3D&lt;float&gt; (Vulkan) / Texture3D&lt;float&gt; (Metal) | Heterogeneous-medium density grid `volumeDensity` — ONE R16F 3D texture, normalized to [0,1] (value-max folded into the packed σ so the texel is the density multiplier). Sampled by `densityAt`'s `MEDIUM_NANOVDB` case through the folded world→uvw affine in `FlatMaterialParams`. Always bound (1×1×1 zero fallback when no volume). On Metal splits into `Texture3D` + `SamplerState volumeDensitySampler` at **binding 44** (design D8); a texture, not a buffer, so the 31-slot buffer table is unaffected (change `nanovdb-volume-rendering`) | `materials/subsurface/medium.slang` |
| 30 | RWStructuredBuffer | Tool readback (float4) — scene pick / BXDF / BSSRDF probe. *Metal slot-cap gate:* compiled out of the neural-active wavefront build (`SKINNY_METAL && SKINNY_METAL_NEURAL`), where it is dead, to fit 33–35 under Metal's 31-buffer argument table | `bindings.slang` |
| 31 | StructuredBuffer | Env importance-sampling distribution `envDistCdf` — **one** buffer = marginal CDF (`ENV_H+1` floats) then conditional CDF (`H×(W+1)` floats) at element offset `ENV_COND_CDF_BASE = ENV_H+1`. Folds the former 31/32 pair into one to free a Metal buffer slot for the neural + online-training build (change `combine-graph-param-buffers`); binding 32 retired | `environment.slang` |
| 33 | StructuredBuffer | Neural-proposal flat Linear weights (`NF_WT`, row-major — `float` by default, `half` in the fp16 precision modes) | `sampling/neural_proposal.slang` |
| 34 | StructuredBuffer | Neural-proposal flat Linear biases (`NF_WT` — `float`/`half`) | `sampling/neural_proposal.slang` |
| 35 | StructuredBuffer | Neural-proposal per-Linear-layer headers (`NfLayerHeader`: weightOffset, biasOffset, inDim, outDim — precision/size-agnostic) | `sampling/neural_proposal.slang` |
| 36 | RWStructuredBuffer | Neural training-record append buffer (`PathRecord`, 64 B) — written by the `mainImageRecord` dump entry **and** the wavefront path integrator (when `fc.recordMode` is set). *Metal slot-cap gate:* compiled out of the neural-active wavefront build (with 30/37, see binding 30) | `integrators/path_record_common.slang` |
| 37 | RWStructuredBuffer | Record append counter (`uint[2]` = `[count, capacity]`) — same Metal slot-cap gate as 36 | `integrators/path_record_common.slang` |
| 45 | StructuredBuffer&lt;float&gt; | **Spectral upsampling — scale grid `spectralScale`** — the Jakob-Hanika RGB→spectrum sigmoid-coefficient table's RES node array (`SPECTRAL_TABLE_RES` = 64 floats). **Spectral-build-only** (`#if defined(SKINNY_SPECTRAL)`); absent from the RGB SPIR-V. Uploaded by `renderer.py` only when `--spectral` is active; on Metal binds by name (`spectralScale`), so the `vk::binding` index is inert there (change `spectral-rendering`) | `bindings.slang` |
| 46 | StructuredBuffer&lt;float&gt; | **Spectral upsampling — coefficient grid `spectralData`** — flat `[3][res][res][res][3]` sigmoid-coefficient table (2,359,296 floats at res 64). Spectral-build-only, see binding 45 | `bindings.slang` |
| 47 | StructuredBuffer&lt;float&gt; | **CIE D65 SPD `spectralD65`** — the reference illuminant SPD normalized to unit luminance (`SPECTRAL_D65_COUNT` = 95 floats), consumed by `upsampleIlluminant`. Spectral-build-only, see binding 45 | `bindings.slang` |
| 48 | StructuredBuffer&lt;float&gt; | **Named-conductor eta/k `spectralMetals`** (Group 6.2) — au/ag/al/cu (ids 1..4), each `[eta(95) \| k(95)]` on the 360–830/5 nm grid (stride 190 floats). Sampled at the 4 hero λ by `namedMetalEtaK` for exact complex-index Fresnel. Spectral-build-only, see binding 45 | `bindings.slang` |
| 49 | StructuredBuffer&lt;float&gt; | **Per-emissive-triangle blackbody `spectralEmitters`** (Group 6.1) — `(temperature_K, scale)` per emissive triangle (2 floats), **parallel-indexed to the emissive-triangle buffer (binding 18)**; a blackbody area light carries `(T>0, blackbody_scale(T, emission))`, a plain-RGB emitter `(0,0)`. NEE substitutes `planckSpectrum(sw,T)·scale` for the RGB illuminant upsample. Spectral-build-only, see binding 45 | `bindings.slang` |
| 50 | StructuredBuffer&lt;float&gt; | **Per-distant-light illuminant SPD `spectralLightSpd`** (Group 6.3) — 95 floats/light on the 360–830/5 nm grid (host-scaled to the light's RGB luminance), indexed by the `DistantLight._direction.w` slot (−1 = none → RGB upsample). Fixed `DISTANT_LIGHT_CAPACITY` (16) slots. Spectral-build-only, see binding 45 | `bindings.slang` |
| 51 | StructuredBuffer&lt;float&gt; | **Per-material blackbody `spectralMatEmission`** (Group 6.1 follow-up) — `(temperature_K, scale)` per flat material, **indexed by materialId**. Lets a camera-visible / BSDF-hit blackbody emitter use the exact Planck SPD (matching the NEE path's binding-49 lookup) instead of the RGB upsample. Grown/rebound with the flat-material buffer. Spectral-build-only, see binding 45 | `bindings.slang` |
| 52 | RWStructuredBuffer | **MLT primary-sample vectors `mltPrimarySamples`** — the per-chain PSS state `X` (`MltPrimarySample` = value/backup + lastMod/modBackup, 16 B), `nChains × dims_per_chain`. The PSS `RNG` override in `common.slang` reads/mutates it. **MLT-build-only** (`#if defined(SKINNY_MLT)`, change `mlt-integrator`); absent from the default RGB/megakernel SPIR-V. On Metal binds by name (`vk::binding` index inert) | `common.slang` |
| 53 | RWStructuredBuffer | **MLT chain metadata `mltChainMeta`** (`MltChainMeta`, `nChains`) — per-chain `{depth, cCurrent, pCurrent, LCurrent (rgb), rngState, iteration counters}`; the accept/reject bookkeeping. MLT-build-only, see binding 52 | `wavefront/wavefront_mlt.slang` |
| 54 | RWStructuredBuffer | **MLT current-state records `mltCurrentRecords`** (`MltRecord`, `nChains × MLT_RECORD_SLOTS`, `MLT_RECORD_SLOTS = BDPT_MAX_VERTS + 1`) — the accepted chain's captured contributions (eye value + ≤ `BDPT_MAX_VERTS` light-tracer splats) restored on reject and re-splatted per mutation. MLT-build-only, see binding 52 | `wavefront/wavefront_mlt.slang` |
| 55 | RWStructuredBuffer&lt;float&gt; | **MLT bootstrap weights `mltBootstrapWeights`** (`nBootstrap`) — each bootstrap L-evaluation writes its scalar contribution `c` here; the host reads it back once per accumulation reset to build the CDF, `b = (1/N)·Σc`, and resample chain seeds proportional to weight. MLT-build-only, see binding 52 | `wavefront/wavefront_mlt.slang` |
| 56 | RWStructuredBuffer&lt;uint&gt; | **MLT chain seeds `mltChainSeeds`** (`nChains`) — the resampled `bootstrapIndex` per chain (host-uploaded after the bootstrap readback); `wfMltInit` replays each seed to reconstruct the chain's initial current state. MLT-build-only, see binding 52 | `wavefront/wavefront_mlt.slang` |
| 57 | RWStructuredBuffer | **MLT proposal records `mltProposalRecords`** (`MltRecord`, `nChains × MLT_RECORD_SLOTS`) — device-memory scratch for the proposed eye contribution and light-tracer splats. Keeping these records out of a thread-local array is required by the spectral Metal live-state budget. MLT-build-only, see binding 52 | `wavefront/wavefront_mlt.slang` |

The table is the **Vulkan** layout. On the **Metal** target (gated
`#if defined(SKINNY_METAL)`, Vulkan SPIR-V byte-unchanged) the combined
`Sampler2D` slots split into a `Texture2D` + a `SamplerState`, because slang-rhi's
Metal backend cannot bind a combined `Sampler2D` and the 128-texture pool exceeds
Apple's compute argument limits: binding 14 becomes `Texture2D[120]` sampled
through a shared `commonSampler` at **binding 38**, and the five discrete maps
(env 4, tattoo 8, normal 9, roughness 10, displacement 11) keep their texture slot
but gain a per-map `SamplerState` at **bindings 39–43** (5 + `commonSampler` =
6 ≤ 16). The buffer/image slots (0–37) are identical on both backends.

The heterogeneous-volume density grid (`volumeDensity`, binding 26) adds a sixth
discrete `Texture3D` + a `SamplerState` at **binding 44** (change
`nanovdb-volume-rendering`). To stay under Apple's **128-texture** compute-argument
limit the bindless flat-material pool is trimmed **`Texture2D[120]` → `[119]`**
(`BINDLESS_TEXTURE_CAPACITY`): 119 pool + 5 discrete 2D maps + the 3D grid +
output/accum/hudMask exactly fills the table. Before the trim the pool filled it to
128 and the added 3D texture silently failed every Metal pipeline (all-black
frames).

On **Vulkan**, binding 26 must also be declared in the shared set-0 layout
(`ComputePipeline._create_descriptor_set_layout`, `vk_compute.py`) as a combined
image sampler — the megakernel SPIR-V references `volumeDensity` unconditionally
(the medium walk is compiled in), so an undeclared binding is undefined behaviour
on desktop Vulkan and a hard `SPIR-V to MSL conversion error: nullptr` pipeline
build failure on MoltenVK (`VUID-VkComputePipelineCreateInfo-layout-07988`). This
layout is shared by the megakernel and every wavefront stage pipeline (via
`scene_bindings_only`), so one entry covers all of them. The hostless audit
`tests/test_vk_binding_layout.py` asserts every Vulkan-branch `[[vk::binding(N)]]`
in `bindings.slang` has a matching layout entry, so a new shared scene binding
cannot ship without its declaration (change `fix-vulkan-volume-density-binding`).

**Spectral bindings 45–51** are compiled in **only** the spectral megakernel
variant (`#if defined(SKINNY_SPECTRAL)`, change `spectral-rendering`) and are
absent from the default RGB SPIR-V, so they never enter an RGB build's set-0
layout. `renderer.py` uploads them only when `--spectral` is active: the three
upsampling `StructuredBuffer<float>`s (45/46/47 — the Jakob-Hanika scale grid,
the sigmoid-coefficient grid, and the unit-luminance CIE D65 SPD), plus four
exact-source buffers — named-conductor eta/k (48, `spectralMetals`, Group 6.2),
per-emissive-triangle blackbody `(T, scale)` (49, `spectralEmitters`, Group 6.1,
parallel-indexed to binding 18), per-distant-light illuminant SPD (50,
`spectralLightSpd`, Group 6.3, indexed by `DistantLight._direction.w`), and
per-material blackbody `(T, scale)` (51, `spectralMatEmission`, indexed by
materialId — the exact-Planck visible/BSDF-hit emission companion to 49). On
Metal they all bind by name so the `vk::binding` index is inert. The table resolution
(`SPECTRAL_TABLE_RES` = 64) and D65/grid length (`SPECTRAL_D65_COUNT` = 95) ride
as **compile-time constants**, not `FrameConstants` fields, so the RGB UBO
packing is unchanged.

**MLT bindings 52–57** are compiled in **only** the MLT wavefront variant
(`#if defined(SKINNY_MLT)`, change `mlt-integrator`) and are absent from the
default RGB SPIR-V (the megakernel `.spv` stays byte-identical), so they never
enter a non-MLT build's set-0 layout. They hold the per-chain PSSMLT state:
`mltPrimarySamples` (52, the primary-sample `X` vectors read by the PSS `RNG`
override), `mltChainMeta` (53, per-chain accept/reject bookkeeping),
`mltCurrentRecords` (54, the accepted chain's captured eye + light-tracer
contributions), `mltBootstrapWeights` (55, the bootstrap `c` weights read back
for the CDF/`b`-normalization), and `mltChainSeeds` (56, the resampled
`bootstrapIndex` per chain), plus `mltProposalRecords` (57, device-memory
proposal scratch for the eye value and light splats). Sized by `nChains` (not `stream_size`) via
`mlt_buffer_sizes` in `wavefront_layout.py` (SPPM `sppm_buffer_sizes`
precedent). On Metal they all bind by name so the `vk::binding` index is inert.
The full state and algorithm reference is
[MetropolisLightTransport.md](MetropolisLightTransport.md).

**One host declaration owns rows 52–57** (change `mlt-binding-declaration`):
`wavefront_layout.MLT_CHAIN_BUFFERS` states, per buffer, its `mlt_buffer_sizes`
key **together with** its Vulkan binding and its Metal shader-global name — the
same pairing the shader states in one `[[vk::binding(N)]] … <name>` line. Every
consumer derives from it and none may restate it: `vk_compute`'s set-0 layout
entries, `gpu_resources.MLT_BINDINGS` (creation-time dummy writes + pool count),
`WavefrontMltPass`'s descriptor writes, and `MetalWavefrontMltPass`'s
bind-by-name map. The rule exists because the failure mode here is a
**transposition**, not an omission: an omission is a `KeyError` or a MoltenVK
layout error, but a valid binding paired with the wrong buffer allocates, binds
and dispatches, and silently diverges one backend's image — which MLT's 0.15
self-consistency tolerance would charge to Markov correlation.
`tests/test_mlt_binding_declaration.py` gates the declaration against the
shader's own pairing (count-checked first, so it cannot pass vacuously) and
fails the build if any consumer re-introduces a binding number or a Metal
global name of its own.

`commonSampler` is created **repeat/repeat** to match the Vulkan per-slot
samplers (the `TexturePool` default is `wrap_s = wrap_t = "repeat"`). One shared
sampler cannot honour per-texture USD `wrapS`/`wrapT`, so repeat/repeat is the
correct default for the tiling material pool — clamp-V (the equirect env-map
default) would clamp a `tiledimage` sampled past v=1 (e.g. a `uvtiling=4`
material) to the edge row on Metal while Vulkan tiles it.

Binding **25** (`GRAPH_BINDING_BASE`) is the single combined MaterialX nodegraph
param buffer — one byte buffer for any number of graphs (was one
`StructuredBuffer` per graph at 25..25+N−1). On Metal, buffer argument-table
indices are assigned by kernel-parameter order, not vk::binding, so the only
deterministic way to keep the neural + online-training wavefront kernel under the
31-slot cap is to **reduce the bound-buffer count**: collapsing the per-graph
buffers to one (graph count no longer grows the table) and folding the two env
CDFs into `envDistCdf` (binding 31) together free the slots that buffer. The
neural-proposal weight buffers sit at **33+**, above the graph buffer (25), the
tool buffer (30), and the env CDF (31). All three
are **always bound** — the renderer seeds them with a full-sized all-zero ("dummy")
net so the inline flow inverse referenced by `sampling/proposal.slang` has valid
descriptors on every pipeline, **including the megakernel** (which never sets the
neural bit); real per-scene weights overwrite them when the neural proposal is
activated. The full reference is in
[Wavefront.md § Neural directional proposal](Wavefront.md#proposal-seam-neural-directional-proposal-proposal-bit2-wavefront-only).

The network **size and precision are build-time configurable** (study change
`neural-precision-size-study`): bindings 33/34 keep their slots but their **element
type follows `NF_WT`** — `float` in the default fp32 mode, `half` in the
fp16-storage / fp16-compute modes (the host casts the fp32 NFW1 file to half at
upload, halving the GPU footprint). Their byte size tracks the configured
`(layers, bins, hidden)`. The header buffer (35) is precision- and size-agnostic.
No binding slot moves; the shader's `NF_WT`/`NF_CT` aliases + `NF_LAYERS/BINS/HIDDEN`
`#ifndef` defaults reproduce the shipped net byte-for-byte when no override is
given. See [Wavefront.md § Neural size & precision](Wavefront.md#neural-size--precision-tuning-neural-precision-size-study).

A fourth **fp8-storage** mode (`NeuralPrecision.FP8_STORAGE`, change
`neural-trainer-backends`) compiles with `-D NF_FP8=1 -D NF_WT=uint`: bindings
33/34 carry e4m3 (OCP E4M3FN) weights packed 4-per-`uint` (a **quarter** of the
fp32 footprint), and `neural_flow.slang nf_fetch` decodes each byte to float in
the scalar GEMM (`nf_decode_e4m3`). The decode is plain integer math + `exp2`, so
it needs **no device feature** — the most portable precision (Vulkan / Metal /
MoltenVK alike). `NF_CT` stays `float`; fp8 *compute* is out of scope (would need
a cooperative-matrix rewrite). The host encode is `neural_weights.f32_to_e4m3`,
mirrored bit-for-bit by the shader decode.

Under **`--neural-handoff interop`** (online training, change
`neural-online-training`) bindings 33/34/35 are allocated as **externally-shared
memory** on Vulkan (`VK_KHR_external_memory`, **dedicated allocation** — required
for the CUDA import on NVIDIA) so the CUDA trainer can write freshly-baked
weights (33) and biases (34) straight into them with no CPU round-trip — the
slots and element types are unchanged, only the buffers' memory backing differs.
A companion exportable **timeline semaphore** (`VK_KHR_timeline_semaphore` +
`VK_KHR_external_semaphore_win32`/`_fd`) orders the CUDA write against the Vulkan
read so a frame never tears. On the native **Metal** backend the same flag
allocates bindings 33/34 as **UMA shared-storage** buffers instead (change
`metal-neural-interop`; binding 35 is immutable after build and stays
device-local): the publisher stages published bytes host-side and the
frame-boundary swap writes them in place on the render thread after the frame's
device drain — no exported handles, no semaphore, no NFW1 round-trip. The
default `--neural-handoff file` keeps them as ordinary device-local buffers the
host re-uploads on a hot-reload. See
[Online neural training](OnlineTraining.md#online-neural-training).

Bindings **36/37** back the per-vertex training-record stream
(`PathRecord`/`emitRecord`, shared in `integrators/path_record_common.slang`).
Two producers append to them: the offline dump via a second megakernel entry
`mainImageRecord` (`Renderer.dump_path_records` → a `.nrec` file), and — for the
**live online-training drain** — the wavefront path integrator itself, which
emits the same records during the normal render whenever `fc.recordMode` is set
(`wavefront/wf_records.slang`; a per-lane vertex stack in the path pass's own
set-1 bindings 9/10 carries the snapshots, splatted at termination). `mainImage`
never references 36/37 (dead-stripped → byte-identical), so they are seeded with
1-element dummies and only reallocated to per-frame capacity during a dump or
while the wavefront drain is armed. `Renderer.drain_path_records_to_replay` is
source-selectable (`_record_source`: `auto` → wavefront for the wavefront path
integrator, else the megakernel dispatch): the wavefront source needs **no**
megakernel dispatch — removing the ~400 s-compile / 2 s-TDR seam that loses the
device on NVIDIA/Windows — and reads the buffers the render already filled via
the shared `records_from_buffer` reader. The drain runs on both backends
(change `metal-record-drain`): Vulkan rebinds descriptor 36 to the drain
target with the `[count, capacity]` counter in 37; Metal routes a merged
header+records byte-address buffer through the bind-by-name dict (capacity at
byte 0, atomic count at byte 60, packed 64-byte records from byte 64 — the
same record bytes as Vulkan), resetting only the 4-byte count word per frame.
The megakernel record source is refused on Metal with a clear error. See
[Online neural training](OnlineTraining.md#online-neural-training).

Light uniforms (part of UBO, not separate bindings):
- `lightDirection` (float3) — analytic directional light toward-light vector
- `lightRadiance` (float3) — analytic directional light colour × intensity

`ProceduralParams` (formerly binding 20) was removed; procedural flat colour is
now derived inside `flat_shading.slang` without a dedicated buffer.

### Wavefront pass-local descriptor sets (set 1)

The wavefront passes bind the scene set above as **set 0** and add a pass-local
**set 1** for their stream state (these are NOT part of the scene set):

| Set 1 binding | Owner | Content |
|---|---|---|
| 0 | `WavefrontPathPass` / `RestirDiPass` | `WavefrontPathState[]` (per-lane path state) |
| 1 | `WavefrontPathPass` / `RestirDiPass` | `HitInfo[]` (per-lane primary/bounce hit) |
| 2–7 | `WavefrontPathPass` | counting-sort queues (lane-slot / counts / offsets / queue / cursor / indirect args) |
| 8 | `WavefrontPathPass` | `WfNeuralSample[]` (per-lane neural forward sample `{wi, pdf, version, valid}`, 32 B) — written by the neural pre-pass, read by the flat shade |

The **neural pre-pass** (`WavefrontNeuralProposalPass`) binds set 0 verbatim and a
3-binding set 1 of its own (0 path-state, 1 hit, 2 the `wfNeural` output buffer above);
see [Wavefront.md § Neural directional proposal](Wavefront.md#proposal-seam-neural-directional-proposal-proposal-bit2-wavefront-only).

**ReSTIR DI** (`RestirDiPass`) uses its own set-1 layout — it shares bindings 0–1
(path-state + hit, over the same buffers as the path pass) and adds three
ReSTIR-owned per-pixel buffers:

| Set 1 binding | Owner | Content |
|---|---|---|
| 2 | `RestirDiPass` | `Reservoir[]` A (ping-pong; fill writes, spatial reads) |
| 3 | `RestirDiPass` | `Reservoir[]` B (ping-pong; spatial writes, resolve reads; persists across frames for temporal) |
| 4 | `RestirDiPass` | G-buffer `{pos, normal}[]` (spatial-neighbour domain check; per-neighbour material is re-loaded from `wfHits` for the GRIS p̂ re-eval) |

`RestirPC` push constant (36 B scalar): `streamSize, flags (bit0 spatial / bit1
temporal / bit2 biased), mLight, spatialK, spatialRadius, normalThresh,
depthThresh, mCap, mBsdf`.

The full ReSTIR DI reference — pipeline stages, equations, the equation→shader
map, design choices, and GUI controls — is in **[ReSTIR.md](ReSTIR.md)**.

**SPPM** (`WavefrontSppmPass`, `INTEGRATOR_SPPM = 2`) uses its **own** set-1
layout — it does **not** share the path pass's stream state. Four **typed**
structured buffers (no `ByteAddressBuffer` fold — a SPPM kernel sits ~15/31 Metal
slots, so no `SKINNY_METAL_SPPM` gate is needed), each sized by **num_pixels**
(the persistent per-pixel estimator, not `stream_size`):

| Set 1 binding | Owner | Content |
|---|---|---|
| 0 | `WavefrontSppmPass` | `sppmVisiblePoints` — `VisiblePoint[]`, the persistent per-pixel estimator (eye geometry + evaluated flat BSDF + per-pass direct + τ/r/N) |
| 1 | `WavefrontSppmPass` | `sppmAccum` — `SppmAccum[]`, per-pass fixed-point atomic flux accumulator (cleared each pass) |
| 2 | `WavefrontSppmPass` | `sppmGrid` — `uint[]`, four sub-ranges over `numCells`: `gridCount` \| `gridOffset` \| `gridCursor` \| `sortedIdx` |
| 3 | `WavefrontSppmPass` | `sppmScanScratch` — `uint[]`, `ceil(numCells/256)` block sums for the parallel prefix scan |

SPPM's 12-byte push constant reuses the path pass's `{streamBase, shadeSlot
(unused), streamSize}` tile layout. The `FrameConstants` SPPM tail
(`sppmInitialRadius`, `sppmCellSize`, `sppmGridRes`, `sppmPhotonsEmitted`) is read
only when `integratorType == 2`. The full SPPM reference — pipeline, equations,
buffer layout, pbrt mapping, deferred phases — is in
**[PhotonMapping.md](PhotonMapping.md)**.

---

## GPU resource inventory (`gpu_resources.py`, change `renderer-gpu-resource-set`)

Every GPU resource the renderer owns is declared **once**. That single
`ResourceDecl` carries its allocation inputs, its binding identity on *both*
backends (Vulkan descriptor binding number, Metal shader-global name, either
optionally absent), and its destruction. `SceneResourceSet` is the list plus the
code that walks it.

Before this, the same inventory was described four times in `renderer.py`,
thousands of lines apart — `_init_gpu` allocated by attribute name,
`_create_descriptors` + five `_rebind_*_descriptors` wrote the Vulkan sets,
`_build_metal_binds` re-stated the identical mapping as Metal names, and
`cleanup` destroyed by attribute name again. Adding a resource meant editing all
four and remembering the last one.

**What it owns:** `_init_gpu`'s allocations, `_create_descriptors`, the five
`_rebind_*_descriptors` methods, `_rewrite_size_dependent_descriptors`,
`_ensure_mesh_buffer_capacity`'s realloc, `_build_metal_binds`' resource table,
and `cleanup`'s destroy list.

**Interface** — small on purpose:

| Call | Does |
|------|------|
| `SceneResourceSet(ctx, gpu, sizes, spectral=…)` | allocates every active declaration, in declaration order |
| `.bind_vulkan(sets, mlt_bindings=…)` | writes every binding, in the recorded write order |
| `.metal_binds()` | the shader-global → resource dict, from the same declarations |
| `.regrow(*attrs)` | reallocates at the sizes the declarations now yield, after a capacity on `.sizes` was bumped, and rebinds |
| `.replace({attr: args})` / `.resize(w, h)` | reallocates at explicit sizes / the viewport-sized set, and rebinds |
| `.adopt(attr, res)` | transfers an externally built resource into a declared slot |
| `.write_binding(n, res, sets)` | one write for a resource whose lifetime is *not* the renderer's (the graph-param buffer, the record-drain buffer) |
| `.close()` | destroys every allocated resource, in the recorded teardown order |

**One backend branch (design D3).** It lives at the binding step: the Vulkan
adapter emits descriptor writes, the Metal adapter fills the name table. The
five per-method `is_metal` / `descriptor_sets is None` early-returns that used
to open each rebind helper are gone — the Metal adapter simply *has no
descriptor step*, rather than each method opting out.

**Three orders, each recorded once.** They genuinely differ and all three are
preserved verbatim:

- `DECLARATIONS` — allocation order.
- `VULKAN_WRITE_SEQUENCE` — descriptor-write order. Neither allocation nor
  binding order: 26 is written between 11 and 12, 20 after 24, and 1/30/31 last.
- `DESTROY_SEQUENCE` — teardown order.

**Growth sites state capacities, never byte sizes.** `regrow` re-evaluates the
declaration's own `cap * STRIDE + slack`, so the arithmetic cannot drift between
the initial allocation and a later grow, and the rebind is part of the same call.
This is what closes the bug class where binding 49 (`spectralEmitters`) kept
pointing at a freed buffer because only 18 was rewritten.

**Renderer access.** Resource attributes are read-only properties forwarding to
the set, so the ~120 `self.<resource>` reads inside `renderer.py` are unchanged
(design D4). Assignment is refused: `self.vertex_buffer = X` would reallocate
without rebinding — exactly the split this module removes. To replace a
resource, declare the new size through the set. Tests that fake a resource on a
`__new__`-constructed renderer use `SceneResourceSet.stub(attr=…)`.

**Gates.** `tests/fixtures/gpu_resource_inventory.json` is the pre-change
inventory *captured from the live renderer* on Vulkan RGB, Metal RGB and Metal
spectral — recorded reality, never a transcription, the same discipline as
`shader-variant-key-module`. `tests/test_gpu_resources.py` pins the
declarations, all three orders, both bind adapters and alloc/destroy pairing to
it, and fails the build if a `VkWriteDescriptorSet` reappears in `renderer.py`
or a deleted rebind helper comes back. If a declaration legitimately changes,
**re-capture** the fixture; do not edit the expectation to match the code.

**Not owned here:** the descriptor *pool* and *sets* themselves (Vulkan objects,
created in `_create_descriptors`), the seeding uploads (they call renderer
methods), the combined MaterialX graph-param buffer and the record-drain buffer
(scene-graph / per-frame lifetimes — they route their writes through
`write_binding` but the set does not allocate or destroy them).

---

## Byte-layout ownership (`slang_layout.py`, change `reflection-owned-byte-layouts`)

The byte layouts the host mirrors from a Slang struct have **one owner**:
`src/skinny/slang_layout.py` (the structs listed below — `SkinParameters`'
std140 UBO, `INSTANCE_STRIDE` and the light-buffer records remain
single-authored at their packers and are a documented follow-up). It parses the authoritative `.slang` declaration
and computes both dialects the renderer speaks — **scalar** (Vulkan,
`-fvk-use-scalar-layout`: offsets are a pure running sum, so the declared field
order *is* the reflection equivalent, hostlessly) and **MSL** (Metal: `float3` /
`uint3` pad to 16 B, struct rounds up to its largest member alignment). Packers
and allocators are consumers; no hand-maintained field/offset table remains.

| Owned struct | Source | Consumer |
|---|---|---|
| `FrameConstants` (+ `Camera`) | `common.slang` | `_FC_SCALAR_FIELDS*`, `_TILE_ORIGIN_Y_OFFSET`, `_VK_UNIFORM_BUFFER_BYTES` bound, `_pack_uniforms_msl` |
| `FlatMaterialParams` | `common.slang` | `FLAT_MATERIAL_STRIDE`, `FLAT_MATERIAL_FIELDS` + `pack_material_record` (see Material field table below) |
| `StdSurfaceParams` | `mtlx_std_surface.slang` | `STD_SURFACE_STRIDE`, `std_surface_fields()`, `pack_std_surface_params_msl` |
| `WavefrontPathState`, `RecVertex`, `VisiblePoint`, `SppmAccum`, `BDPTVertex`, `WfBdptAux`, MLT chain structs | wavefront/SPPM/BDPT/MLT shaders | `wavefront_layout.py` sizers (public API unchanged) |

**Variants.** Exactly three preprocessor gates are resolvable —
`SKINNY_SPECTRAL`, `SKINNY_MLT`, `SKINNY_METAL` — resolved per query. Anything
else (unknown gate, unknown field type, unrecognised declaration form)
**raises**: an unparseable struct is a test failure, never a guessed offset.

**The `FrameConstants` blob rule.** Declaration order is *not* host blob order.
`tileOriginY` is `SKINNY_METAL`-gated and declared *before* the `SKINNY_MLT`
tail, but `_pack_uniforms` always appends it **last** — so under an MLT pack
`mltSigma` lands at 564, exactly where the Vulkan MLT SPIR-V (which has no
`tileOriginY` at all) expects it, and the trailing word is benign filler inside
the oversized UBO. `fc_scalar_blob()` applies that rule; base blob 568 B, MLT
blob 600 B.

**Drift gates, three layers, none optional:**

1. **Hostless (primary)** — `tests/test_slang_layout.py` pins each owned
   struct's scalar stride (and the MSL stride for the structs the Metal
   allocator sizes against) to a golden, pins the declared `(type, name)` list
   per struct so a same-size swap or same-width retype cannot slip through,
   checks gap/overlap coverage, locks the `fc` blob order, and asserts the
   raise-on-unknown paths (unknown gate/type, attributed field, multi-declarator,
   array member, locals inside a method body). Goldens are *not* derived, so a
   parser change and a shader change moving together still trip a visible
   failure. The existing `wavefront_layout` / `test_struct_layout` /
   `test_sppm_state` / `test_mlt_host` locks stay at full strength, now reading
   the module.
2. **gpu-marked MSL ground truth** — `tests/test_metal_fc_layout.py` (`fc`, RGB
   and MLT) plus the `_reflect_msl_layout` locks in `test_wavefront_state.py`
   and the `StdSurfaceParams` round-trip assert the derived MSL offsets equal
   what Slang's Metal target actually emits. This is what validates the
   `float4x4` / `uint2` / `uint3` / nested-struct rules.
3. **Runtime** — `_pack_uniforms` asserts the derived table covers the blob it
   just produced (both backends); `_pack_uniforms_msl` still packs from **live**
   reflection (the `metal-backend` contract) and cross-checks it against the
   derived layout once per program+variant, so drift in either direction raises
   before an upload.

### Material field table (change `flat-material-field-table`)

The two material records are owned one level deeper: **field by field**, not
only by stride.

`FlatMaterialParams` declares 14 opaque `float4` rows, so the parser derives
each row's offset but not which lane means `roughness` and which means
`metallic`. That map used to be a docstring above a 60-argument positional
`struct.pack`. Transposing two same-typed arguments changed every rendered pixel
and passed every test, because the only guard was a size-equality assert.

**Derive where possible, pin where not.** `slang_layout.FLAT_MATERIAL_FIELDS`
declares one `MaterialField` per field: `(name, row, lane, kind, default, key)`.
The byte offset is DERIVED — the row's parsed offset plus `4 × lane` — so a
shader-side reorder moves the field with no edit to the table. Only the lane
assignment is declared. `StdSurfaceParams` declares named scalars, so its half
of the table is derived outright.

**The table is load-bearing, not documentation.** `material_pack` emits both
records through `slang_layout.pack_material_record(record, {name: value})`, so
the bytes are produced BY the table. That is what makes the permanent
name→offset golden (`tests/fixtures/material_field_offsets.json`) a real
transposition gate: swap two same-typed fields and both offsets move.
`tests/test_material_field_table.py` carries the negative control that performs
the swap and asserts the gate fires — and shows the size-only check still
passing on the same swap.

**One override-key vocabulary.** The strings that ride
`Material.parameter_overrides` from the pbrt authors, through the intake merge,
to the packers have one owner in the same module:

| Set | Contents |
|---|---|
| `FLAT_OVERRIDE_KEYS` | the 30 keys `pack_flat_material` reads — the table's own `key` bindings plus `FLAT_DERIVED_KEYS` (medium, dispersion, named conductor) |
| `STD_SURFACE_OVERRIDE_KEYS` | derived from the `StdSurfaceParams` declaration |
| `PREVIEW_SURFACE_FLAT_KEYS` | flat keys the loader's own folds author under UsdPreviewSurface names |
| `INTAKE_ONLY_KEYS` | recognised bookkeeping no packer reads (`clearcoat`, `subsurface_eta`, `pbrt_medium`, …) |
| `RENDERER_OVERRIDE_KEYS` | read by the renderer, by neither packer (`emissive_spectral`) |

Three tables used to restate parts of this vocabulary behind "keep in sync"
comments — `usd_loader._STD_SURFACE_TO_FLAT` (5 entries),
`mtlx_synthesis._STD_SURFACE_TO_FLAT_PACK` (12) and
`_PREVIEW_SURFACE_FLAT_KEYS`. All three are projections now, and the comments
are assertions. `std_surface_to_flat()` derives its 12 entries as four genuine
renames plus every std-surface key the flat packer reads under the same name
**and the same kind**. The kind guard is load-bearing: std-surface `opacity` is
a `color3` while the flat record's is a `float`, so aliasing them would
advertise an edit `_override_float` discards.

**Unknown keys are refused, within scope.** `parameter_overrides` is a shared
bag: it also carries a MaterialX document's own input names and a Python
material's slangpile inputs, which are DATA and cannot be enumerated. So
`material_pack.check_material_vocabulary` refuses an unknown key on a material
the table owns (plain UsdPreviewSurface / pbrt-imported flat), and only warns on
a material with an `mtlx_target_name` or a `python_module`. Refusal was enabled
only after a report-only survey of all 49 corpus/suite scenes — 1093 materials,
39 distinct keys, zero unknown.

**Merge and derive are ordered once.** `usd_loader._apply_override_derivations`
states the sequence (transmission bridge → subsurface bridge → coat
canonicalisation) in one place, called after the `customData["skinnyOverrides"]`
merge on both intakes. The mtlx-fallback intake used to derive inside
`_load_mtlx_materials`, before any prim's customData was available, and then
re-run `_derive_opacity_from_subsurface` in `_merge_prim_overrides` once the
interior medium had arrived. Merging first removes the second run.

## Shader variant key (`shader_variants.py`, change `shader-variant-key-module`)

A compiled kernel's identity is a point in a variant matrix, and
`src/skinny/shader_variants.py` is its **one owner**. Before it, each backend
re-derived its own define list at its own compile site (11 emission sites across
`vk_compute` / `vk_wavefront` / `metal_compute` / `metal_wavefront`), so
"Vulkan and Metal agree per variant" was convention, not code.

`ShaderVariantKey` is a frozen value over six axes:

| Axis | Values |
|------|--------|
| `target` | `VULKAN` (`slangc` → SPIR-V, flat `-D` tokens) \| `METAL` (in-process SlangPy session, defines dict) |
| `family` | `MEGAKERNEL` \| `WAVEFRONT` \| `WAVEFRONT_FOUNDATION` \| `PREVIEW` \| `DEBUG_RASTER` |
| `spectral` | `SKINNY_SPECTRAL=1` + `_spectral` filename suffix |
| `mlt` | `SKINNY_MLT=1` + `_mlt` tag — wavefront only (this is what keeps the megakernel SPIR-V byte-unchanged by the MLT axis) |
| `metal_neural` / `metal_records` | `SKINNY_METAL_NEURAL` / `SKINNY_METAL_RECORDS` — the Metal argument-table gates. METAL-target **and** wavefront-family only: both defines are live in `bindings.slang` / `path_record_common.slang`, which the megakernel also includes, so a megakernel key carrying one would silently change that kernel's binding table |
| `neural` | a composed `NeuralBuildConfig` — the `NF_*` defines and the `L6B24H96…` cache slug stay owned by `sampling/neural_weights.py` and are **never** re-derived here |

`__post_init__` enforces a `(target, family)` validity table —
`WAVEFRONT_FOUNDATION` is Vulkan-only (Metal compiles every wavefront kernel
through the full CP+METAL+WAVEFRONT session), `DEBUG_RASTER` is Metal-only (the
Vulkan debug viewport is a graphics rasteriser) — plus the axis rules: `mlt`,
a composed `NeuralBuildConfig` and the two Metal gates are wavefront-only (the
gates additionally METAL-only), and `spectral` exists on megakernel + wavefront
only.

**An axis a family cannot carry is refused, never accepted and then dropped.**
That distinction is the whole point: `cache_token()` folds the neural slug and
the spectral/MLT suffixes into the `.spv` filename, so a key that *accepted* an
axis its emission path ignored would name one variant on disk and compile
another. Every arm of `slangc_flags()` therefore emits all four segments (the
ones a family cannot carry are provably empty), with an assertion that every
declared define reached the flag tuple.

Four derivations, all from one shared internal define table:

- `slangc_defines()` → the Vulkan `-D` tokens as **named ordered segments**
  (`base` / `spectral` / `neural` / `mlt`);
- `slangc_flags(key, entry=…, include_paths=…)` → the full `slangc` flag tuple,
  splicing those segments at each family's **recorded historical position**
  relative to `-fvk-use-scalar-layout`. Three distinct orders exist (megakernel
  and preview put the spectral define *after* the flag; the foundation compile
  puts the flag *before* its define; the wavefront full-tree compile puts every
  segment *before* it) and the flag tuple is hashed **positionally** into the
  `build/spv_cache` blake2b key — so a single contiguous block could not
  reproduce the existing cache keys. Keeping the positions is what makes the
  migration a no-flush change.
- `session_defines()` → a fresh complete dict for the SlangPy Metal session.
  Assign it in **one** statement: `SlangCompilerOptions.defines` is
  copy-on-read, so `opts.defines[k] = v` mutates a throwaway.
- `cache_token()` → the `.spv` filename tag (`""` for the default key,
  `_<neural slug>` then `_mlt` then `_spectral`).

The module also owns the **`build/spv_cache` derivation** that `vk_compute`
duplicated across `ComputePipeline` and `PreviewPipeline`: `spv_cache_key()`
(blake2b-128 over entry point + source path + the flag tuple + every `.slang`
under `shaders/` and `mtlx/genslang/`) and `spv_cache_fetch()` (hit → copy the
cached module out, touch it for LRU, and return before `slangc` is invoked).
Both pipelines now delegate. Because the flags are folded in **positionally**,
this is the mechanism that makes the splice positions load-bearing: reorder a
flag tuple and every existing cache entry is orphaned.

Two named constants keep the deliberate divergences reviewable instead of
folklore: `METAL_ONLY_DEFINES` (the defines with no Vulkan counterpart) and
`RECORDED_ASYMMETRIES` — currently one entry, `sppm-neural-defines`: the Metal
SPPM pass compiles with the active `NF_*` defines while the Vulkan SPPM compile
passes none (vacuous at the default config, which emits zero `NF_*` flags;
aligning them is a follow-up change).

`tests/test_shader_variants.py` is the drift gate. It holds **permanent**
goldens transcribed from the pre-refactor tree (flag tuples, defines dicts,
filename tags, and every wavefront kernel's entry + `.spv` out-name), pins the
`spv_cache_key` digest over a fixture tree, asserts cross-backend agreement for
every family valid on both targets, and greps the four backend modules to prove
no compile site hand-assembles a variant define any more.
`tests/test_spv_cache_hit.py` (`gpu`-marked, since it imports the
vulkan-importing `vk_compute`) proves the payoff end to end: a rebuild over an
unchanged tree returns the cached module with `subprocess.run` patched to fail,
and a one-character shader edit falls through to `slangc`. The host sizers in
`wavefront_layout.py` keep their `spectral=` / `msl=` signatures, but the
wavefront passes read those booleans off the same `ShaderVariantKey` their
kernels compiled with, so shader defines and host sizing cannot disagree.

## FrameConstants Layout

Compiled with `-fvk-use-scalar-layout` — float3 has 4-byte alignment. The
authoritative field order/offsets are derived from the declaration by
`slang_layout` (see above); this table is the human-readable gloss.

| Offset | Type | Field |
|--------|------|-------|
| +0 | Camera | viewInverse, projInverse, position, fov |
| ... | uint | frameIndex |
| ... | uint | accumFrame |
| ... | float | time |
| ... | uint | width, height |
| ... | uint | numDistantLights (active count in binding 20; 0 = IBL only) |
| ... | uint | useMesh |
| ... | float | tattooDensity |
| ... | float | envIntensity |
| ... | uint | furnaceMode |
| ... | float | mmPerUnit |
| ... | uint | detailFlags |
| ... | float | normalMapStrength |
| ... | float | displacementScaleMM |
| ... | uint | numInstances |
| ... | uint | numSphereLights |
| ... | uint | numEmissiveTriangles |
| ... | uint | integratorType (0 = path, 1 = BDPT, 2 = SPPM, 3 = MLT) |
| ... | (lens) | numLensElements (0 = pinhole, else thick-lens), film/aperture/pupil + focus-overlay + zoom-rect + vignette-debug fields |
| ... | uint2 | pickPixel; uint pickArmed (one-shot scene pick → toolBuffer) |
| ... | float | exposure (EV stops, 2^EV) |
| ... | uint | tonemapMode (0 ACES, 1 Reinhard, 2 Hable, 3 linear) |
| ... | uint | proposalMask; uint reuseMode; float4 proposalAlpha (scene-sampling seam) |
| ... | uint | flatLobeSamplers — per-lobe flat-BSDF sampler ids, 8 bits/lobe (`coat \| spec<<8 \| diff<<16`; 0 = native). Unpacked by `flat_material.slang`; no new binding |
| ... | float3 | sceneBoundsMin; float3 sceneBoundsExtent — scene AABB for the neural-proposal condition's position normalisation |
| ... | uint | neuralNetworkVersion — active frozen-net version (baseline 0; per-sample network-version hook for future online training) |
| ... | uint | recordMode — 1 while the wavefront training-record drain is active (else 0; default render byte-identical) |
| ... | uint | cameraMirror — 1 for an improper (mirrored) pbrt camera; `zoomedNDC` negates ndc.x (+ `sampleWi` for BDPT) for a horizontal screen mirror, else 0 |
| ... | (sppm) | sppmInitialRadius, sppmCellSize, sppmGridRes, sppmPhotonsEmitted, sppmGlossyContinueRoughness — per-pass SPPM params, read only when `integratorType == 2` |
| ... | float | filmMaxComponent — pbrt `Film "maxcomponentvalue"` per-sample radiance clamp; each sample is scaled so `max(r,g,b) ≤ filmMaxComponent` (hue-preserving) before accumulation, matching pbrt `RGBFilm::AddSample`. 0 = disabled (no-op; render byte-identical). Set from the imported pbrt film by `usd_loader` |
| ... | float ×4 | sppmGroupPmfE/S/D/Env — SPPM photon-emission group selection pmf (change `sppm-power-proportional-photon-groups`): P(emissive/sphere/distant/env), proportional to each group's emitted power, normalized host-side (`renderer._sppm_photon_group_pmf`; uniform-over-present fallback). Zeros when the integrator is not SPPM |

`cameraType` was removed — camera selection is implied by `numLensElements`
(0 ⇒ pinhole). `exposure` and `tonemapMode` are post-process knobs and do not
reset accumulation. The scalar tail (`sceneBoundsMin` / `sceneBoundsExtent` /
`neuralNetworkVersion` / `recordMode` / `cameraMirror`, the SPPM per-pass fields,
`filmMaxComponent`, and the four `sppmGroupPmf*` floats) brings the scalar UBO
blob to **564 B** (568 B with the trailing `tileOriginY` u32); the
neural/record/sppm fields are read only when their feature is active, and
`cameraMirror` / `filmMaxComponent` default 0, so the default `{bsdf}` path stays
bit-identical. `filmMaxComponent` is part of the accumulation state-hash (changing
it resets accumulation). The Vulkan UBO is allocated with headroom
(`_VK_UNIFORM_BUFFER_BYTES`, currently 768 B) because `UniformBuffer.upload`
memmoves `min(len, size)` and would otherwise silently truncate the blob's
tail.

---
