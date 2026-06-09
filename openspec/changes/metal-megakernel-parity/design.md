## Context

P1 (`add-metal-backend-foundation`, on `main`) delivered a native Metal device
(`metal_context.py`), a minimal `metal_compute.py` (`StorageBuffer`,
`StorageImage`, `ComputePipeline` — enough for a trivial dispatch + present), and
the `--backend` selection seam (`backend_select.py`). It proved an in-process
Slang→Metal compile + dispatch is bit-identical to Vulkan for a trivial kernel,
and that a windowed present does not hang **provided no per-field `ShaderCursor`
scalar write straddles an open encoder** (the D4 fence-hang discipline: set
uniforms via `set_data` byte blobs).

The full renderer is still Vulkan-only. `renderer.py` imports the resource
classes directly from `vk_compute` (one top-level `from skinny.vk_compute import
…` plus ~9 inline imports) and constructs every GPU resource on `self.ctx`,
passing `ctx` as the first constructor argument throughout. The megakernel path
compiles `main_pass.slang` (`mainImage`) to SPIR-V, reflects the descriptor set
layout (bindings 0–24, 30–32), uploads a `std140` uniform block
(`FrameConstants` + `SkinParams` + `Light`) via `UniformBuffer`, and binds a
bindless `SampledImage` pool + a shared sampler. P2 makes that same path run on a
`MetalContext`, at structural parity with the Vulkan output, and flips
`auto`→Metal on Apple Silicon.

Constraints carried from P1: SlangPy is the only Metal conduit (no raw PyObjC);
the entry point must not be named `main` (`mainImage` is safe); Slang's Metal
target pads `float3` to 16 B, so the `std140` uniform blob does not match the MSL
struct; Metal capability flags stay `false` (the megakernel needs no external
memory/semaphores/fp16).

## Goals / Non-Goals

**Goals:**
- The megakernel path (`main_pass.slang` `mainImage`) renders the head on a
  `MetalContext` and reaches **structural parity** with the Vulkan megakernel:
  exact on the deterministic structural outputs (geometry/instance IDs, hit/miss,
  depth) and within a perceptual tolerance on the converged shaded color.
- `metal_compute.py` reaches public-API parity with the `vk_compute.py` classes
  the renderer consumes, so resource construction is backend-agnostic at the call
  sites.
- An MSL-correct uniform-pack path (`_pack_uniforms_msl`) driven by reflected MSL
  field offsets, uploaded via `set_data` byte blobs only.
- `select_backend("auto")` → Metal on Apple Silicon (else Vulkan); the real
  front-ends launch the renderer on Metal instead of refusing; `--backend vulkan`
  stays byte-identical everywhere.

**Non-Goals:**
- ReSTIR DI (P3), neural inference (P4), wavefront execution (P5), the MLX↔Metal
  zero-copy weight handoff (P6), fp16 on Metal — all deferred. The existing
  capability gates already fold ReSTIR/wavefront back to the megakernel on Metal.
- Porting the renderer onto the unused `gfx/` Backend ABC, or salvaging the dead
  `metal` branch (`e1300e7`) renderer.
- Exact byte parity Metal vs Vulkan for shaded color (different GPU
  transcendental/GGX ULPs make it unachievable; see D5).
- Renaming `Renderer(vk_ctx=)` → `ctx=` (cosmetic; deferred from P1).

## Decisions

### D1: One resource-module resolution point, not per-site branches
`metal_compute.py` mirrors the **public API** of `vk_compute.py` exactly (same
class names, same constructor signatures, same upload-helper names). The renderer
resolves the active resource module **once** from the context —
`self._gpu = backend_select.resource_module(self.ctx)` (returns `vk_compute` or
`metal_compute`, keyed on `ctx.is_metal`) — and references resource classes
through `self._gpu.*`. The ~9 inline `from skinny.vk_compute import …` sites
resolve through the same helper.
*Alternatives rejected:* (a) `if ctx.is_metal:` at every construction site —
duplicates the branch ~30×, drifts; (b) a third abstract resource layer both
backends subclass — a large refactor with no user-visible value mid-port (the P1
D1 rationale); (c) making each resource class self-dispatch on its `ctx` arg —
the classes are backend-specific by construction (raw `vk.` vs SlangPy), so the
dispatch belongs above them, not inside.

### D2: In-process Slang→Metal compile of `main_pass.slang`, mirror the Vulkan reflection
The Metal `ComputePipeline` compiles `main_pass.slang` (`mainImage`) in-process
via SlangPy (`load_module_from_source`/`link_program`/`create_compute_kernel`),
exactly as P1's trivial kernel, and reflects the binding layout from the linked
program so the renderer's existing descriptor-binding map (bindings 0–24, 30–32)
addresses the same logical slots. Resource binding goes through
`dispatch(vars=…)` / `ShaderCursor` **resource** binds (buffers, textures,
samplers) — these are permitted; only per-field **scalar** writes into uniform
structs are banned (D4). The Vulkan path keeps its pre-compiled `main_pass.spv`
unchanged.
*Alternative rejected:* shell `slangc -target metal` → `.metallib` and load it.
Kept as a fallback only (O1 below) — P1 proved the in-process path works.

### D3: MSL-correct uniform packing via reflected offsets, byte-blob upload
`SkinParameters.pack()` keeps the Vulkan `std140` layout untouched. A sibling
`_pack_uniforms_msl` packs `FrameConstants` + `SkinParams` + `Light` to the MSL
struct layout — Slang's Metal target pads `float3`→16 B (16-aligned). **Spike-
confirmed (task 1.2):** `FrameConstants` is **592 B / 16-align MSL vs 512 B /
4-align scalar**; the embedded `Camera` is **288 B vs 272 B**; the delta is
entirely the `float3` fields (16 B vs 12 B). The uniform block is a single global
`uniform FrameConstants fc` at binding 0 — skin/std params are SSBOs, not part of
it. Field offsets are **queried from the compiled module's reflection**
(`program.layout.parameters` → `fc.type_layout.fields[i].offset`) rather than
hand-mirrored, so the blob can never silently drift from the shader. The blob is uploaded with `set_data` only (never
per-field cursor writes — D4). The renderer chooses `_pack_uniforms` vs
`_pack_uniforms_msl` by `ctx.is_metal`.
*Alternative rejected:* a hand-maintained MSL offset table — exactly the drift
class the `SkinParams` docstring warns about; reflection is self-checking.

### D4: `auto`→Metal flip lives in `backend_select`, gated on a constructible device
`select_backend("auto")` returns `"metal"` on Apple-Silicon macOS **iff** the
Metal device constructs, else `"vulkan"` (unchanged elsewhere). The
`METAL_FOUNDATION_NOTICE` refusal path in the four front-ends is removed; they now
build the context via `make_context` and run the renderer. Precedence (explicit >
env > persisted > auto) is unchanged from P1. `--backend vulkan` stays
byte-identical.
*Alternative rejected:* flip `auto` but keep a feature flag gating Metal render —
adds a second switch with no benefit now that render parity is the deliverable.

### D5: Structural-parity acceptance, not byte parity
Parity is verified in two tiers: **exact** equality on the deterministic
structural outputs (geometry/instance IDs, hit/miss mask, depth — integer/
position data with no transcendental divergence), and a **perceptual tolerance**
(e.g. SSIM / relative-error threshold) on the converged shaded color of a fixed
scene. Byte-identical shaded color is unachievable across Metal vs Vulkan (per-GPU
GGX/transcendental ULP differences) and is explicitly not required.
*(User decision, this change.)*

### D7: Extract the generated-Slang emission into a shared, backend-agnostic helper (spike finding)
The compile spike (task 1.1) established that `main_pass.slang` does **not**
compile standalone on either backend: it `import`s three artifacts the renderer
**generates at scene-load** and writes into the shader include path before
compiling — `generated_materials.slang` (the `evalSceneGraph` aggregator),
`generated/*_graph.slang` (per scene MaterialX graph), and
`python_materials_dispatcher.slang` (`PythonMaterial`/`loadPythonMaterial`). On
Vulkan these are emitted inside `vk_compute.ComputePipeline.__init__` before the
`slangc` call. The emission is **pure text generation** (no Vulkan handles), so
P2 extracts it to a module-level `emit_megakernel_sources(shader_dir,
graph_fragments)` that both `vk_compute` and `metal_compute` call before linking.
The checked-in genslang closures (`mtlx/genslang/*.slang`) are resolved by adding
`mtlx/genslang` to the Metal device's `include_paths` (the Metal analogue of the
Vulkan `-I mtlx/genslang`).
*Alternative rejected:* duplicate the emission in `metal_compute` — guarantees
drift the moment the aggregator/dispatcher format changes.
*Spike-confirmed compile recipe (Metal):* `create_device(type=metal,
include_paths=[shaders, mtlx/genslang])`, source prefixed `#define
SKINNY_COMPUTE_PIPELINE 1`, link entry `mainImage`; **no
`-fvk-use-scalar-layout`** (Vulkan-only — Metal uses MSL layout, see D3). No
Metal-target language construct was rejected; `[[vk::binding]]` maps cleanly.

### D6: Capability flags stay `false` on Metal
`supports_external_memory`, `supports_external_semaphore`, `supports_fp16_storage`,
`supports_fp16_compute` remain `false` (P1 D5). The megakernel needs none of them;
ReSTIR/wavefront capability gates already fold back to the megakernel on Metal, so
no new gate is required for P2.

## Risks / Trade-offs

- **MSL uniform-layout drift** (`float3` 16 B padding; 288 B vs 272 B). →
  Reflect offsets from the compiled module (D3), assert the packed length equals
  the reflected struct size in a test; never hand-mirror.
- **`main_pass.slang` may use Vulkan-only constructs that fail on `-target
  metal`.** → Compile-spike the full module on Metal early (O2); isolate any
  offender behind a `__target_switch`/capability path. Fallback: `.metallib`
  shell-out (D2 alternative).
- **A stray per-field `ShaderCursor` scalar write reintroduces the fence hang.** →
  Audit every Metal bind path to `set_data`/resource-binds only; the present smoke
  + a render-loop test catch an un-signalled fence.
- **The ~10 `vk_compute` import sites are easy to miss one of.** → Grep-gate:
  after D1, no `from skinny.vk_compute import` may remain on a Metal-reachable
  path; CI/lint check + the headless Metal render test surfaces a missed site as
  an `AttributeError`.
- **Two resource implementations drift.** → `metal_compute` mirrors the
  `vk_compute` public API; the cross-backend structural-parity test is the
  contract guard; the surface grows test-first.
- **Metal tests need Apple Silicon; CI may lack a GPU/display.** → Structural
  parity is the **headless** test; the windowed present stays a display-gated
  smoke; all Metal tests skip cleanly off Apple Silicon.

## Migration Plan

- Vulkan and every non-macOS host are untouched (`--backend vulkan` byte-identical;
  no SPIR-V change). The only behavior change for users is on Apple Silicon, where
  `auto` now renders through Metal.
- Rollback: `--backend vulkan` (or `SKINNY_BACKEND=vulkan`) restores the prior
  path immediately; the `auto`→Metal flip is a single point in `backend_select`.
- Land order: resource-layer parity (D1) → MSL uniform pack (D3) → Metal
  megakernel pipeline + descriptors + textures (D2) → structural-parity test (D5)
  → flip `auto` + drop the refusal (D4) → docs.

## Open Questions

- **O1 (RESOLVED — in-process, no shell-out):** the full `main_pass.slang` links
  in-process on `-target metal` via SlangPy once the generated artifacts are
  emitted (D7). Recipe: `include_paths=[shaders, mtlx/genslang]` + `#define
  SKINNY_COMPUTE_PIPELINE 1`, entry `mainImage`, no `-fvk-use-scalar-layout`.
- **O2 (UPDATED — one Metal-only construct found):** `[[vk::binding]]`,
  `RWTexture2D`, `StructuredBuffer`, and `IMaterial` generics all compile on the
  Metal target. **But** once all generated modules are present (D7), a full
  `main_pass.slang` Metal link reveals `flat_shading.slang`'s
  `flatMaterialTextures[NonUniformResourceIndex(idx)]` —
  `NonUniformResourceIndex` is **unavailable in the compute stage on `metal`**
  (Slang `error[E36107]`, on both `mainImage` and `mainImageRecord`). It is only a
  divergence hint, so the fix is to make it identity on the Metal target (a `NRI`
  macro/helper) and byte-unchanged on Vulkan. Tracked as task 4.0. No other
  construct rejected so far.
- **O3:** Perceptual-tolerance metric + threshold for D5 — SSIM ≥ 0.99 vs
  relative-MSE < ε on the accumulation (linear-HDR) image; pick against a fixed
  reference scene during test authoring.
- **O4:** Bindless `SampledImage` capacity (`BINDLESS_TEXTURE_CAPACITY`) on Metal —
  does the Metal argument-buffer texture limit accommodate it, or is a smaller cap
  needed? Verify against the device limits during texture binding.
