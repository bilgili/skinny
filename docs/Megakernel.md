# Skinny — Megakernel Execution Mode

The **megakernel** is Skinny's default GPU execution mode: the entire render —
camera ray generation, scene traversal, material shading, the full bounce loop,
accumulation, tonemap, and overlay compositing — runs in **one monolithic
compute dispatch** of `shaders/main_pass.slang`, one thread per pixel.

It is one of two execution modes (the other is the
[wavefront](Wavefront.md) mode). Both run on the Vulkan backend; the mode is
selected once at startup with `--execution-mode megakernel|wavefront`
(`cli_common.py:85`) or `SKINNY_BACKEND`-adjacent config and is **fixed for the
session** — there is no runtime cycler, and it is deliberately excluded from
`_current_state_hash` (`renderer.py:1346-1348`, `6965`). The mode constants are
`EXECUTION_MEGAKERNEL = 0` / `EXECUTION_WAVEFRONT = 1` (`params.py:64-65`).

> **Terminology note.** "Megakernel" and "wavefront" are *execution modes*, not
> graphics backends. There is currently no native Metal path in the tree; both
> modes dispatch through Vulkan (MoltenVK on macOS). The `--backend metal`
> language in some older docs refers to MoltenVK-via-Vulkan, not a separate API.

For *why* the wavefront mode exists alongside this one, see
[Why two execution modes](#why-two-execution-modes) below and the matching
section in [Wavefront.md](Wavefront.md).

---

## 1. Dispatch

| Aspect | Value | Reference |
|--------|-------|-----------|
| Shader entry | `mainImage` | `main_pass.slang:377-437` |
| Workgroup | `[numthreads(8,8,1)]` = 64 threads, **one thread per pixel** | `main_pass.slang`, `WORKGROUP_SIZE=8` `renderer.py:65` |
| Dispatch grid | `ceil(W/8) × ceil(H/8) × 1` | `renderer.py:7159-7173` |
| Frequency | one `vkCmdDispatch` per frame | `render()` `renderer.py:7065` |
| Parameters | single UBO (FrameConstants + SkinParams + lights), **no push constants** | `_pack_uniforms()` `renderer.py:6693+`, upload `7094` |

The pipeline is built only in megakernel mode — gated by
`megakernel = self.execution_mode_index == EXECUTION_MEGAKERNEL`
(`renderer.py:2661`); `ComputePipeline(... entry_module="main_pass",
entry_point="mainImage" ...)` compiles `main_pass.slang` → SPIR-V
(`renderer.py:2695-2723`). In wavefront mode `main_pass` is **not** compiled (a
`scene_bindings_only` build supplies just the set-0 layout, `renderer.py:2728`).

Per-frame: `vkCmdBindPipeline` → `vkCmdBindDescriptorSets(descriptor_sets[f])`
→ `vkCmdDispatch(groups_x, groups_y, 1)` → blit storage image to swapchain
(windowed) or copy to readback (headless, `render_headless` mirrors this at
`renderer.py:7367`). The same pipeline is reused synchronously for the
BXDF/BSSRDF visualiser, keyed by `toolBuffer[0]` (`renderer.py:5609-5619`,
`main_pass.slang:386-396`).

---

## 2. Per-pixel flow (all inline in one invocation)

`main_pass.slang` imports every subsystem into the single kernel
(`main_pass.slang:14-32`): `common, bindings, interfaces, scene_trace,
environment, integrators.path, integrators.bdpt, cameras.{pinhole,thick_lens},
generated_materials, materials.flat.*, materials.skin.*, tonemap`. The body
`runFrame<TC:ICamera>` (`main_pass.slang:36-187`) executes in order:

![Megakernel per-pixel flow: one thread generates a ray, traces the scene, runtime-selects BDPT/PathTracer/env, sanitises, accumulates (+ light-splat), tonemaps, composites overlays, and writes outputBuffer — all in one dispatch.](diagrams/megakernel_per_pixel.svg)

Step-by-step with line refs:

| Step | Reference |
|------|-----------|
| `cam.generateRay(pixel, rng, lensWeight)` | `cameras/{pinhole,thick_lens}.slang` |
| `traceScene(fc, ray) → HitInfo` | `scene_trace.slang` |
| optional pick-buffer write | `main_pass.slang:46-54` |
| integrator select (BDPT if flat first-hit / else Path / env on miss) | `main_pass.slang:58-76` |
| lens throughput weight; NaN/Inf/neg sanitise | `main_pass.slang:81-88` |
| progressive accumulation (running mean) | `main_pass.slang:90-102` |
| + BDPT light-splat composite (Q22.10) | `main_pass.slang:104-118` |
| exposure (2^EV) → tonemap → sRGB | `main_pass.slang:120-121` |
| focus-plane / furnace / HUD / gizmo overlays | `main_pass.slang:129-184` |
| `outputBuffer[pixel] = display` | `main_pass.slang:186` |

The integrator and camera are **monomorphised** structs constructed in
registers (`PathTracer` `main_pass.slang:70-71`, `BDPTIntegrator<TC>` `62-66`);
`runFrame` is templated on `ICamera` so `ThickLensCamera` (when
`fc.numLensElements > 0`) vs `PinholeCamera` specialise at the call site
(`main_pass.slang:427-436`).

---

## 3. The whole path is one register-resident loop

The defining property: **one thread traces an entire light path with an
in-register `for`-loop over bounces.** In `PathTracer.estimateRadiance`
(`integrators/path.slang:124+`): `throughput`, `radiance`, the current `Ray`,
`HitInfo`, and the MIS `prevBsdfPdf` all live in registers across
`for (uint bounce = 0; bounce < MAX_BOUNCES; bounce++)` (`path.slang:148`,
`MAX_BOUNCES = 6` `path.slang:28`), with Russian roulette after bounce 0
(`path.slang:176-183`) and a cutout-transparency skip loop capped at 32
(`path.slang:141,225`).

BDPT is likewise single-thread but builds eye/light subpaths into **register
arrays** `BDPTVertex eye[BDPT_MAX_VERTS]` / `lit[...]` (`bdpt.slang:522-523`),
then connects (`connectGeneric`/`connectT1`). Light-tracer s=1 vertices are
atomic-splatted to the global `lightSplatBuffer` (binding 21, Q22.10
fixed-point, `bdpt.slang:758,801-803`).

No cross-thread path state exists — each pixel is independent. The only
persistent GPU buffers are output targets and tables: `accumBuffer`,
`outputBuffer`, `lightSplatBuffer`, `toolBuffer`, `gizmoSegments`, `hudMask`,
the UBO, material tables, lights, bindless textures, env CDFs (descriptor set 0).

---

## 4. Selection: compile-time vs runtime

| Decision | Mechanism | Reference |
|----------|-----------|-----------|
| Integrator (path/BDPT) | **runtime** branch on `fc.integratorType` (BDPT only if flat first-hit) | `main_pass.slang:58-72` |
| Camera (pinhole/thick-lens) | **compile-time** monomorphisation, runtime tag pick | `main_pass.slang:427-436` |
| Material (flat/skin/debug/python) | **runtime** tag-switch on `materialTypes[id] & 0xFF` | `evaluateBounce()` `path.slang` |
| MaterialX nodegraphs | **compile-time** codegen into `generated_materials` at pipeline build | `vk_compute.py:538` |
| `MAX_BOUNCES`, `BDPT_MAX_VERTS`, proposal seam | **compile-time** constants/imports | `path.slang:28`, `bdpt.slang` |

Adding or removing a MaterialX graph triggers a `slangc` recompile; everything
else branches at runtime inside the one kernel.

---

## 5. Progressive accumulation

- **GPU:** running mean in `accumBuffer` keyed by `fc.accumFrame` — `n=accumFrame;
  n==0` writes the sample, else `accum=(prev*n + sample)/(n+1)`
  (`main_pass.slang:90-102`). `accumBuffer` is persistent across frames.
- **CPU:** `update()` computes `_current_state_hash()` (`renderer.py:6947-6983`)
  over camera/lights/env/model/integrator/proposal-seam/playback-time/MaterialX
  overrides. Hash changed → `accum_frame = 0` + zero the light-splat buffer; else
  `accum_frame += 1` (`renderer.py:7047-7057`). `execution_mode_index` is
  excluded (fixed for the session, `renderer.py:6965`). `frameIndex` (RNG seed,
  `createRNG(pixel, fc.frameIndex)` `main_pass.slang:401`) advances every frame
  so each accumulated sample draws fresh noise.

---

## 6. Design trade-offs

**Strengths**

- **One dispatch.** No inter-stage buffers, queues, sort, or compaction; one
  pipeline, one descriptor set. Accumulation is trivial (no path compaction).
- **Everything composites inline** — overlays, tonemap, BDPT splat — so megakernel
  and wavefront produce bit-comparable pixels (`tonemap` is shared,
  `main_pass.slang:29-32`). This is why it is the default mode (`renderer.py:1096`).
- **Low CPU overhead** — minimal command recording per frame.

**Costs (inherent to the design)**

- **Register pressure.** Skin BSSRDF + GGX + the full MaterialX graph set + BDPT
  subpath register arrays (`BDPTVertex eye[BDPT_MAX_VERTS]`) all coexist in one
  kernel, capping occupancy.
- **Divergence.** Runtime per-pixel branching on material type, integrator, and
  camera model, plus per-bounce cutout/RR loops, makes warps diverge — every lane
  pays for the most expensive branch any lane in the warp takes.
- **Compile size.** The fat kernel (skin + python + all integrators) is large; on
  MoltenVK it can trip the Metal compiler's kernel-size limit — the direct
  motivation for the wavefront mode.
- `MAX_BOUNCES = 6` is hard-capped partly to bound this kernel's cost.

---

## Why two execution modes

The megakernel is simplest and produces the reference image, but its single fat
kernel suffers warp divergence (mixed materials in a warp serialise) and high
register pressure, and on MoltenVK the monolithic `main_pass` can exceed the
Metal compiler's kernel-size limit. The **[wavefront](Wavefront.md)** mode
splits the same estimator into many small per-stage / per-material kernels
connected by GPU queues: same math, better material coherence, and each kernel
small enough to compile. The trade is memory bandwidth (path state round-trips
through VRAM between bounces) and CPU-side command-recording overhead.

Both modes are unbiased estimators of the **same** integral and are
A/B-verified to match (the wavefront kernels are verbatim relocations of the
megakernel's per-bounce body and call the same `evaluateBounce` / MIS /
proposal-seam code). Pick megakernel for simplicity and the reference path;
pick wavefront for coherence-sensitive scenes and to dodge the MoltenVK
compile limit.

See [Wavefront.md § Megakernel vs wavefront](Wavefront.md#megakernel-vs-wavefront)
for the side-by-side table.

---

## Key files

| File | Role |
|------|------|
| `shaders/main_pass.slang` | entry `mainImage`, per-pixel body, overlays, tool branches |
| `shaders/integrators/path.slang` | register bounce loop, `MAX_BOUNCES=6` |
| `shaders/integrators/bdpt.slang` | subpath register arrays, light splat |
| `renderer.py:2661,2695-2723` | mode gate + megakernel pipeline build |
| `renderer.py:7159-7173` / `7367` | windowed / headless dispatch |
| `renderer.py:6693+` | UBO packing |
| `renderer.py:6947-6983`, `7047-7057` | state hash + accumulation reset |
| `params.py:64-65,79` | execution-mode constants + capability gate |
| `cli_common.py:85` | `--execution-mode` flag |
