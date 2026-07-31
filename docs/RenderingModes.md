# Skinny — Rendering Modes

This document covers what the renderer can be told to do at run time: which GPU
backend runs, at what resolution, which combinations of integrator, execution
mode, proposal, and reuse are in the envelope, and how each integrator samples.

`src/skinny/render_envelope.py` is the source of truth for "does combination X
actually run?". The compatibility matrix below documents that predicate; it is
not a second copy that the code mirrors.

For installation and the command lines that reach these modes see
[README.md](../README.md).

---

## Rendering Modes

### GPU backend (`--backend`)

`--backend {auto,metal,vulkan}` (env `SKINNY_BACKEND`, persisted on the
interactive front-ends) selects the GPU backend for the session, exposed
identically by every front-end from one shared definition. `auto` (the default)
resolves to native **Metal** on a Metal-capable Apple-Silicon host and falls back
to **Vulkan** everywhere else (precedence: `--backend` flag > `SKINNY_BACKEND`
> persisted > `auto`). The native **Metal** backend runs the full renderer at
parity with Vulkan: the megakernel (geometry, shaded color, windowed present)
**and** the wavefront execution mode — staged path + BDPT integrators, ReSTIR DI
reuse, and the neural directional proposal (change `metal-wavefront-parity`).
Both backends compile from the same Slang sources: Metal compiles in-process via
SlangPy (slang-rhi, no MoltenVK), and `vulkan` forces the production path
everywhere (MoltenVK under Vulkan on macOS). An explicit `--backend metal` on a
host with no Metal device fails with a clear message rather than degrading. See
[docs/Backends.md § Backend selection](Backends.md#backend-selection)
and [docs/Wavefront.md § Metal wavefront backend](Wavefront.md#metal-wavefront-backend).

### Render resolution (`--width` / `--height`)

`--width` and `--height` (env `SKINNY_WIDTH` / `SKINNY_HEIGHT`) set the
render-area pixel size, exposed from the same shared definition by the
interactive front-ends `skinny` and `skinny-gui`. Both default to **640×480**
(precedence: flag > env > default); a non-positive value is rejected at startup.
On `skinny` (windowed) they size the GLFW window and the GPU render target
together; on `skinny-gui` they size the offscreen render area (the pixels the
renderer computes) while the Qt window and dock layout keep their own size. The
headless `skinny-render` keeps its own `--width` / `--height` (default
`1024×1024`, offline-output size); `skinny-web` does not expose these flags.

### Compatibility matrix

What runs where — backend × execution mode × integrator × neural × interop.

These tables **document** `src/skinny/render_envelope.py`, which is the source of
truth: one predicate answers "does combo X run, and if not, why", and the parity
matrix, every CLI startup refusal, and the renderer's spectral scene gate all
derive their answer from it. A hostless doc-sync check asserts the key envelope
facts below still match the predicate, so an envelope change that skips the docs
fails a test.

**Backend × feature parity** — `--backend metal` is at full parity with
`--backend vulkan` for the renderer; non-renderer GPU work differs.

| Feature | Vulkan | Metal (native, Apple-Silicon) |
|---------|--------|-------------------------------|
| Megakernel execution | ✅ | ✅ |
| Wavefront execution (path / BDPT / ReSTIR DI) | ✅ | ✅ |
| SPPM integrator (wavefront, flat materials) | ✅ | ✅ (`MetalWavefrontSppmPass`; caustic parity matches Vulkan) |
| MLT integrator — PSSMLT over BDPT (wavefront, flat materials, RGB + spectral) | ✅ (`WavefrontMltPass`) | ✅ (`MetalWavefrontMltPass`; equivalent to Vulkan within the film-splat quantum, **not** bit-identical — see [docs/MetropolisLightTransport.md](MetropolisLightTransport.md#cross-backend-equivalence)) |
| pbrt `subsurface` (volumetric interior random walk) | ✅ | ✅ (megakernel + wavefront, all integrators; under wavefront BDPT/SPPM every non-flat first hit — subsurface/skin/volume/python — falls back to the path tracer, parity with the megakernel, with the heavy multi-bounce cases bounded per eye tile on Metal; lights from a single distant light + the environment) |
| Heterogeneous volumes — NanoVDB `MakeNamedMedium` (path integrator, megakernel + wavefront) | ✅ | ✅ (`disney-cloud` / `bunny-cloud`; distant + env NEE; BDPT/SPPM excluded) |
| Procedural `cloud` medium — pbrt `MakeNamedMedium "cloud"` (analytic Perlin-fBm density, path integrator, megakernel + wavefront) | ✅ | ✅ (`clouds`; no grid/texture — `MEDIUM_CLOUD` evaluates pbrt's `CloudMedium::Density` in-shader; BDPT/SPPM excluded) |
| Neural directional proposal (inference) | ✅ | ✅ |
| Spectral rendering (`--spectral`, hero-wavelength) | ✅ (path + bdpt + sppm + mlt, megakernel + wavefront, flat) | ✅ (`spectral-rendering`, `spectral-bdpt-megakernel`, `spectral-wavefront`) |
| MaterialX `standard_surface` / `OpenPBR` / skin | ✅ | ✅ |
| Per-lobe BSDF sampler registry | ✅ | ✅ |
| Material Graph dock preview (`preview_pass.slang`) | ✅ (descriptor sets) | ✅ (`PreviewPipeline`, bind-by-name; `metal-tool-dock-render`) |
| Camera Debug viewport | ✅ (graphics rasteriser) | ✅ (`DebugRasterMetal` compute rasteriser; `metal-tool-dock-render`) |
| UsdSkel GPU skinning + GPU BVH refit | ✅ | CPU fallback (deformation only) |
| H264 encoder pool (web mode) | NVENC / QSV / AMF | VideoToolbox |
| GPU indirect dispatch (wavefront slot counts) | ✅ | CPU readback fallback (slang-rhi gap) |

**Neural directional proposal (`--proposals …,neural`)** — cross-cutting
constraints, independent of GPU backend.

| Aspect | Constraint |
|--------|-----------|
| Execution mode | **Wavefront only.** Megakernel strips the neural bit and falls back to its analytic subset (`bsdf` / `bsdf,env` / `env`). |
| Materials | **Flat only** (`UsdPreviewSurface`, `standard_surface`, `OpenPBR`, Python materials). Skin path is untouched. |
| Backends | Vulkan ✅, native Metal ✅ — same Slang sources, same MIS pdf accounting. |
| Inference dtype | fp32 (default), fp16 (mixed on Metal w/ graceful fp32 fallback), fp8 e4m3 (in-shader decode, portable across Vulkan / Metal / MoltenVK). |

**Spectral rendering (`--spectral`)** — hero-wavelength transport instead of
RGB. **Live** (`SPECTRAL_IMPLEMENTED = True`): the megakernel spectral integrators
(`path_spectral.slang`, and `bdpt_spectral.slang` for `--integrator bdpt`) render
the path/bdpt+megakernel+flat envelope on both backends — per-wavelength NEE, the
pbrt sigmoid/D65 upsampling model, exact named-conductor Fresnel, authored +
blackbody illuminant SPDs, and hero-λ glass dispersion — resolving through the
Wyman CMF to the existing RGBA32F accumulation. The same hero-λ transport also
threads all three **wavefront** integrators — path, BDPT, and SPPM (change
`spectral-wavefront`) — wired + CPU-verified + merged (RGB `.spv` byte-identical,
179+ hostless tests); its GPU self-consistency + prism/pbrt-truth gates are now
measured on Metal across the confirming-suite spectral scenes (white-furnace
closure + full-corpus sweep still pending). An in-envelope `--spectral` run is accepted
on every front-end; out-of-envelope combos are still refused at startup (see the
scope below). See [Spectral.md](Spectral.md).

| Aspect | Scope |
|--------|----------|
| Integrator / execution | **Path, BDPT, or SPPM.** Path/BDPT run under megakernel + wavefront; SPPM is wavefront-only (no megakernel photon pass). Megakernel path/BDPT is GPU-validated; wavefront (all three) is CPU-verified + merged, with the GPU self-consistency + prism/pbrt-truth gates now measured on Metal (suite scenes; white-furnace + full-corpus pending). Out-of-envelope combos are refused at startup. |
| Materials | **Flat only** — a skin/subsurface/heterogeneous-volume scene under `--spectral` is refused. |
| Sampling layers | Spectral **path** supports the analytic `bsdf`, `bsdf,env`, and `env` directional-proposal presets in megakernel + wavefront; the environment proposal reuses the existing env CDF and full mixture-pdf MIS. BDPT/SPPM/MLT keep their native sampling. No neural proposal or ReSTIR reuse (both refused under `--spectral`). |
| Dispersion | Path + BDPT carry hero-λ Cauchy glass dispersion; **SPPM has no dispersion** (v1 limit — it would break the per-pass photon/visible-point wavelength coherence). |
| Samples | 4 hero-rotated wavelengths over 360–830 nm (pbrt visible-λ pdf); CIE film resolve to the existing RGBA32F accumulation. |

**Online training (`--online-training`)** — combinations of
`--neural-trainer` × `--neural-handoff` × host. Loop requires
`--execution-mode wavefront` **and** a neural proposal in the mixture; both
prereqs hard-checked at startup or at the moment a neural proposal is selected
in the GUI.

| Trainer | Vulkan host | Metal host (Apple-Silicon) |
|---------|-------------|----------------------------|
| `cpu` (torch-free numpy oracle) | ✅ on any host | ✅ always available |
| `cuda` (torch) | ✅ when a CUDA GPU is present | n/a (raises) |
| `mlx` (Apple MLX on the Metal GPU, `pip install -e ".[mlx]"`) | n/a (raises) | ✅ — GPU trainer, the recommended Mac default |
| `auto` | → `cuda` if present, else numpy oracle | → `mlx` if the `[mlx]` extra is importable, else numpy oracle |

| Handoff | Vulkan host | Metal host |
|---------|-------------|------------|
| `file` (NFW1 double-buffer) | ✅ any host | ✅ any host — CPU round-trip through disk, portable |
| `shared` (in-process CPU double-buffer) | ✅ any host — RAM, no disk, no extra deps | ✅ any host — RAM, no disk, no extra deps |
| `interop` (GPU-side, no file) | ✅ requires CUDA + `VK_KHR_external_memory` + timeline semaphore (`pip install -e ".[interop]"`) | ✅ unified-memory shared-storage in-place writes, no extra deps |

Train precision is independent of inference precision (training always bakes
fp32 weights; the handoff format is unchanged).

| `--train-precision` | Behavior |
|---------------------|----------|
| `fp32` (default) | always available, every backend / trainer |
| `fp16` | torch autocast on CUDA; float16 compute over fp32 masters on Apple MLX (runtime fall-back to fp32 on a non-finite step); **falls back to fp32** on the numpy oracle |

**Supported Mac (no CUDA) online-training combo:**
`--execution-mode wavefront --proposals bsdf,neural --online-training
--neural-trainer mlx --neural-handoff {file|shared|interop}` — fully single-device
on Apple Silicon, training on the Metal GPU via Apple MLX; `interop` keeps the
weight handoff GPU-side (UMA write-in-place), `shared` hands weights across in
RAM (no disk, no extra deps), and `file` works identically with a CPU round-trip
through disk. Swap `mlx` for `cpu` to use the torch-free numpy oracle instead.

### Sampling

Four integrators selectable via `--integrator {path,bdpt,sppm,mlt}` across the
front-ends:

| Strategy | Description |
|----------|-------------|
| Path tracing (`path`, default) | Unidirectional with MIS; each estimator pairs a primary sampler with a companion via power heuristic |
| BDPT (`bdpt`) | Bidirectional path tracer with light-tracer splatting for caustics; 4-vertex subpaths, connections evaluate the real `standard_surface` BSDF, env importance sampling matched to the path tracer |
| SPPM (`sppm`) | **Stochastic Progressive Photon Mapping** — caustic-efficient eye/grid/photon/update pipeline; **wavefront-only**, **flat materials only**, on both Vulkan and native Metal (caustic parity matches across backends). Runs under wavefront — `--integrator sppm` **auto-selects** `--execution-mode wavefront` (see below), so it needs no second flag; an explicit `--execution-mode megakernel` is refused. One SPPM pass == one accumulation frame; the per-pixel estimator (radius / count / flux) persists across frames. The initial search radius (default ≈ 0.1 % of the scene bbox diagonal) and photons/pass (default one per pixel) are set by the pbrt `sppm` importer; `--sppm-glossy-roughness` (float; SPPM + wavefront only; default tuned ≈ 0.6 in perceptual/USD roughness, reaching pbrt-imported polished metals; `0` = delta-only PM-1 behaviour) is the glossy / near-specular eye-walk continuation threshold, so glossy metals reconstruct sharp reflections (a glossy metal reflecting only the environment is MIS-weighted on escape). See [docs/PhotonMapping.md](PhotonMapping.md). |
| MLT (`mlt`) | **Metropolis Light Transport** — Kelemen primary-sample-space Metropolis (PSSMLT) driving the existing wavefront BDPT estimator (all strategy families, existing MIS weights), so `E[MLT] = E[skinny BDPT]` by construction. Full-sample chains (Kelemen 2002 / Mitsuba PSSMLT), **not** pbrt's per-depth strategy decomposition — skinny's environment transport is deliberately not strategy-partitioned, so a per-depth split would drop env transport per stratum. Thousands of GPU-parallel Markov chains (default 16384) advance one mutation per frame; each frame runs a bootstrap b-normalization at accumulation reset, then mutate (propose → dual splat of proposal and current state by acceptance, uint fixed-point, **never clamped**) → resolve (fold splats × `b/mpp_actual`, film-averaged like SPPM). **Wavefront-only**, **flat materials only**, on both Vulkan (`WavefrontMltPass`) and native Metal (`MetalWavefrontMltPass`). The two backends are **not** bit-identical to each other: each is bit-reproducible with itself, and they agree to within an integer count of film-splat quanta (relMSE 5.3e-10 RGB / 5.2e-08 spectral on `int_caustic` at 128×128, 512 spp — see [Cross-backend equivalence](MetropolisLightTransport.md#cross-backend-equivalence)). Runs under wavefront — `--integrator mlt` **auto-selects** `--execution-mode wavefront` (see below); an explicit `--execution-mode megakernel` is refused. pbrt imports `Integrator "mlt"` (`mutationsperpixel` / `largestepprobability` / `sigma` / `chains` / `bootstrapsamples` / `maxdepth`; pbrt defaults 100 / 0.3 / 0.01 / 1000 / 100000 / 5). MCMC images **"swim"** early as the chains explore, then the progressive film average stabilizes like SPPM. RGB and spectral targets are supported; neural proposals, ReSTIR, online training, and non-flat scenes are refused at startup (recorded parity skips — no path-fallback inside a Markov chain). See [docs/MetropolisLightTransport.md](MetropolisLightTransport.md). |

**Execution mode follows the integrator.** `--execution-mode
{auto,megakernel,wavefront}` (env `SKINNY_EXECUTION_MODE`, default `auto`,
fixed for the session) picks the GPU execution backend. `auto` (the default)
**derives the mode from the startup integrator** — `path` → `megakernel`,
`bdpt` → `megakernel`, `sppm` → `wavefront`, `mlt` → `wavefront` — mirroring
`--backend auto`, so a plain `--integrator sppm` or `--integrator mlt` just
works. An explicit `megakernel`/`wavefront` (flag or env) overrides the derived
default and pins the mode; the impossible combos, `sppm` + explicit
`megakernel` and `mlt` + explicit `megakernel`, are refused at startup. (In a
megakernel-fixed session, cycling the runtime integrator to a wavefront-only
integrator — `sppm` or `mlt` — falls back to the megakernel path tracer, same
safe wart SPPM has today.)

**Per-lobe BSDF samplers.** The flat / `standard_surface` BSDF draws each lobe
(`coat`, `spec`, `diffuse`) from a runtime-selectable importance sampler. Native
is the 2023 spherical-cap VNDF (coat/spec) / cosine (diffuse); the registry also
ships the Heitz-2018 basis-form VNDF (coat/spec) and uniform-hemisphere
(diffuse). Select per lobe in the GUI or on the command line:
`--lobe-samplers coat=basis,spec=basis,diff=uniform` (env
`SKINNY_LOBE_SAMPLERS`). Every strategy shares one pdf between `sample()` and
`evaluate()`, so switching is unbiased — only the noise / variance changes.

**Directional-proposal mixture.** The bounce direction is drawn from a
runtime-selectable mixture of proposals via one-sample MIS:
`--proposals {bsdf,bsdf+env,env,bsdf+neural,neural}` (env `SKINNY_PROPOSALS`) on
`skinny` / `skinny-web` / `skinny-render`; on `skinny-gui` the **Proposals**
combobox owns the selection at runtime (no `--proposals` flag there). Persisted
either way. `bsdf` (default) is the material's own importance sampler —
bit-identical to the classic renderer; `bsdf,env` MIS-mixes an
environment-importance proposal (lower variance on IBL); `env` is env-only;
`bsdf,neural` MIS-mixes a learned, position-conditioned **neural spline-flow**
proposal (frozen, offline-trained per scene; **wavefront-only**, flat materials —
the megakernel strips the neural bit and falls back to its analytic subset). All
proposals report exact solid-angle pdfs, so every mixture is unbiased — only the
variance changes. The neural network's **size and precision are build-time
configurable** (`NeuralBuildConfig`; mixed fp16 on Apple-Silicon Metal with
graceful fp32 fallback) — the default reproduces the shipped net byte-for-byte;
see [docs/Wavefront.md § Neural size & precision](Wavefront.md#neural-size--precision-tuning-neural-precision-size-study).
`--encoding {E0,E1,E3}` (env `SKINNY_ENCODING`, persisted) selects the
conditioner's positional encoding (axis 2): `E0` (default) feeds the raw
condition — byte-identical to the shipped net; `E1` applies a NeRF-γ feature map
to every condition scalar; `E3` is `E1` plus the raw condition appended. It is
Jacobian-free (only the conditioner input changes — `|J|` and the pdf path are
unchanged) and must match the loaded network's encoding — a first-layer-width
mismatch is refused, not rendered mis-conditioned. See
[docs/NeuralGuiding.md § Condition encoding](NeuralGuiding.md#1-condition-encoding).

**Online neural training.** The neural proposal can be trained *continuously*
while the scene animates, so the net adapts instead of staying frozen on an
offline bake. `--neural-handoff {file,shared,interop}` (env `SKINNY_NEURAL_HANDOFF`,
also GUI/persisted) selects how freshly-trained weights are handed from the
async trainer back to the renderer: `file` (default) double-buffers through an
NFW1 file the renderer hot-reloads — a CPU round-trip through disk that works on
**any** platform; `shared` is an in-process CPU double-buffer held in RAM — the
same byte-faithful round-trip without the disk write, no CUDA / unified-memory
device, on **any** platform; `interop` publishes weights + biases GPU-side with no file round-trip,
resolved per backend: on **Vulkan**, CUDA writes the exported weight buffers via
`VK_KHR_external_memory` + an exported timeline semaphore (needs the `interop`
extra, `pip install -e ".[interop]"`); on the native **Metal** backend the
unified-memory shared-storage weight buffers are written in place at the frame
boundary (no extra dependency). It raises a clear error naming the `file`
fallback on hosts with neither path. The renderer swaps in
new weights only at a frame boundary and bumps the per-sample network version, so
an async swap raises variance only, never bias. See
[docs/OnlineTraining.md § Online neural training](OnlineTraining.md#online-neural-training).

`--neural-trainer {cpu,cuda,mlx,auto}` (env `SKINNY_NEURAL_TRAINER`, also
persisted) selects the **training-compute** backend with precedence
`cuda > mlx > cpu`: `auto` (default) uses torch on CUDA when available, else
Apple MLX on an Apple-Silicon Metal host (when the `[mlx]` extra is importable),
else the torch-free **numpy reference oracle**; `cpu` forces the numpy reference
(always available — a torch-free Mac trains for real); `cuda` forces torch on
CUDA (raises if absent); `mlx` forces Apple MLX on the Metal GPU (`pip install
-e ".[mlx]"`; raises off an Apple-Silicon Metal host).
`--train-precision {fp32,fp16}` (env `SKINNY_TRAIN_PRECISION`, persisted)
sets the optimizer precision independently of inference precision — `fp16` uses
torch autocast on CUDA, float16 compute over fp32 masters on Apple MLX (with a
runtime fall-back to fp32 if a step goes non-finite), and fp32 elsewhere.
Training always bakes
fp32 weights, so the handoff format is unchanged. The inference precision adds an
**fp8-storage** (e4m3) mode — quarter-size weights decoded in-shader with no
device feature, portable across Vulkan/Metal/MoltenVK; see
[docs/NeuralGuiding.md § Training backends & the precision matrix](NeuralGuiding.md#training-backends--the-precision-matrix).

`--online-training` (flag, env `SKINNY_ONLINE_TRAINING`, also persisted) is the
switch that actually **starts** the loop on the interactive front-ends (`skinny`
and `skinny-gui`) — the `--neural-handoff` / `--neural-trainer` /
`--train-precision` flags above only *configure* it. It has two prerequisites:
`--execution-mode wavefront` **and** a neural proposal active in the mixture.
It also requires **`--integrator path`**: BDPT does not consume the neural
proposal (it samples directions with native BSDF sampling on every backend), so
`--integrator bdpt` with a neural proposal or `--online-training` is rejected —
the CLI front-ends error-and-exit at startup, the GUI shows the online-training
status as `REFUSED`. The wavefront prerequisite is fixed for the session, so a
non-wavefront mode is refused with a clear one-line message at startup — never a
silent no-op. The
neural-proposal prerequisite is runtime-selectable, so on `skinny-gui` you can
launch with `--online-training` and *then* pick a neural proposal in the
**Proposals** combobox — the loop is armed and starts the moment a neural
proposal becomes active (no `--proposals` flag on the GUI). An unsupported
backend/handoff combo (`--neural-trainer mlx`, or `--neural-handoff interop`
with neither CUDA nor Metal UMA) surfaces its own error. Off by default:
without the flag the renderer is
byte-identical to before. The supported **Mac** combo (no CUDA) is
`--neural-trainer mlx` (or `auto`, which picks MLX there when the `[mlx]` extra
is installed — `cpu`/numpy otherwise) with `--neural-handoff file`; e.g.

```bash
# skinny-gui: launch armed, then select a neural proposal in the combobox.
skinny-gui --execution-mode wavefront \
  --online-training --neural-trainer cpu --neural-handoff file

# skinny (GLFW): the neural proposal is a CLI flag.
skinny --execution-mode wavefront --proposals bsdf,neural \
  --online-training --neural-trainer cpu --neural-handoff file
```

Training runs on a dedicated background thread, so a slow cycle (the numpy
oracle is ~seconds) never stalls the viewport; the renderer drains GPU path
records each frame and the frame-end swap promotes new weights. The whole loop
runs on either GPU backend — on the native Metal backend the wavefront render
emits the records natively (no megakernel) and `--neural-handoff interop`
publishes weights through unified memory, so online training is fully
single-device on Apple Silicon.

Every front-end prints a `[skinny] configuration` matrix at startup (and
reprints it when a selection flips approval) showing each axis's requested vs
resolved value and the online-training row's `OFF`/`REFUSED`/`WAITING`/`APPROVED`
status with its reason — so it's clear at a glance what's selected and whether
training will run. When training actually starts you get a one-time `[neural]
online training ACTIVE …` line, and on stop/exit a `[neural] online training
STOPPED: ran … cycles … steps … final loss=…` summary; `skinny-gui` also shows
the live state in its status bar. See
[docs/NeuralGuiding.md § Running online training](NeuralGuiding.md#running-online-training).

### Furnace Mode

Swaps the scene to a unit sphere under unit-white radiance. Pixels exceeding
energy conservation tolerance are tinted pink. Supports global and
per-material furnace probes.
