# Skinny — Host Module Map and Ownership

This document covers the Python host side: the module map, the front-end
bring-up builder, the session snapshot owner, the renderer carve-out pattern, and
the device-free pure core.

For the per-file listing see [ImplementationMap.md](ImplementationMap.md).
For the renderer overview see [Architecture.md](Architecture.md).

---

## Front-end bring-up (`bringup.py`, change `frontend-bringup-builder`)

Turning parsed render flags into a running renderer — resolve the startup
integrator and execution mode, run every refusal guard, select the backend,
build the context, build the `Renderer` — has **one owner**: `src/skinny/bringup.py`.
All four front-ends (`skinny` → `app.py`, `skinny-gui` → `ui/qt/app.py`,
`skinny-render` → `headless.py`, `skinny-web` → `web_app.py`) call it; a new
refusal guard added to the sequence is effective on all four with no
per-front-end edit. Front-ends keep only surface-specific wiring (GLFW window,
Qt threading, web server/session lifecycle, MCP flag plumbing).

**Two stages**, because two front-ends cannot construct the context where they
resolve the flags:

| Stage | Call | What it does |
|---|---|---|
| plan | `plan_bringup(args, prog, persisted=None)` | all resolution + all refusal guards → a frozen `BringupPlan` |
| create | `plan.create(window=…, width=…, height=…, gpu_preference=…, **renderer_kwargs)` | `make_context` + `Renderer(...)`, destroy-on-failure. Runs later, on another thread, no guard re-run |

**Canonical order** (`startup_integrator_name` → `resolve_execution_mode` →
`validate_render_flags` → `reject_sppm_without_wavefront` →
`reject_mlt_unsupported` → `reject_spectral_unsupported` →
`reject_mcp_unsupported` → `resolve_walk` → encoding + neural build config →
`select_backend`). `select_backend` is deliberately **last** — it is the only
step that constructs a GPU device, so every way a launch can be refused is
exhausted before one exists. That includes the two inputs argparse never
validates: `--bdpt-walk` has no `choices=` at all (it accepts a deprecated
alias), and an argparse *default* is never checked against `choices=`, so
`SKINNY_BDPT_WALK` / `SKINNY_ENCODING` both reach the plan unchecked and are
rejected here rather than after a device probe.

Refusal prefixes are asymmetric, deliberately: only the backend-selection
failure is wrapped as `SystemExit(f"{prog}: …")`. The `cli_common` guards print
their own fixed `skinny:` prefix on every front-end (the MCP guard prints none),
exactly as they did when each front-end called them directly — repointing them
would change user-visible output on three front-ends.

The resolve-before-validate order matches the
`resolve_execution_mode` docstring and the `render-cli` requirement that
"validation SHALL run after the execution mode is resolved" — which the two
interactive front-ends previously violated, validating *before* resolving and
compensating with explicit `reject_*` re-checks. That old order was only
refusal-equivalent by accident: it fed the unresolved string `"auto"` into
guards, and `cli_common._envelope_mode` maps `"auto"` to `"wavefront"`, so the
pre-resolution megakernel check was a silent no-op.
`tests/test_bringup.py` transcribes **both** pre-refactor orders and diffs the
canonical one against them across the whole guard matrix (integrator ×
execution mode × spectral × proposals × reuse × online-training ×
persisted-vs-CLI × backend), plus verbatim refusal-message pins.

**The plan/pass-through split.** `BringupPlan` carries only the guard-vetted
fields that are the same everywhere — `backend`, `execution_mode`,
`startup_integrator`, `spectral`, `bdpt_walk`, `encoding`, `neural_config` —
and `create` hands those to `Renderer` itself. Front-end-specific constructor
inputs (`usd_scene_path`, `use_usd_mtlx_plugin`, shader/hdr/tattoo dirs,
`neural_handoff` / `neural_trainer` / `train_precision`) are forwarded verbatim
as `**renderer_kwargs`. Post-construction renderer state stays at the call
sites: `skinny`'s persisted overrides, the web session's and Qt's
integrator/reuse indices and lobe samplers, `headless`'s proposal preset.

**Persistence is the caller's one knob** (`persisted=`), mirroring which
front-ends persist today. `skinny` / `skinny-gui` pass the settings dict — the
persisted integrator feeds startup-integrator resolution (so a persisted `sppm`
under an explicitly-forced `--execution-mode megakernel` is refused, which the
CLI-keyed `validate_render_flags` alone cannot see), the persisted backend feeds
`select_backend`, and the persisted `--encoding` feeds the neural build config.
`skinny-render` / `skinny-web` pass nothing, so resolution stays flags +
environment + `auto` only. The flag > env > persisted > auto precedence itself
is unchanged — it lives inside `select_backend` / `resolve_execution_mode`;
this module only decides *whether* the persisted value is offered.

**Where `create` runs per front-end:**

| Front-end | plan at | create at |
|---|---|---|
| `skinny` | `app.py` `main()`, after `load_settings()` | `main()`, after the GLFW window exists |
| `skinny-gui` | `ui/qt/app.py` `main()`, before `QApplication` | `ui/qt/viewport.py` `_RenderWorker._build_renderer`, on the render thread |
| `skinny-render` | `headless.py` `main()` | `HeadlessRenderer.__init__` |
| `skinny-web` | `web_app.py` `main()` (stored as `_PLAN`) | `SkinnySession.initialize()`, per session, on its background thread |

The Qt plan travels *alongside* `QtRendererConfig` (`MainWindow(plan=…)` →
`RenderViewport(plan=…)` → `_RenderWorker(plan)`) — `render_session.py`
signatures are untouched, and the other front-ends are never routed through
Qt's config object. `HeadlessRenderer` and `_build_renderer` accept
`plan=None` for direct API use (tests, the parity harness, an embedded Qt
surface): their already-resolved kwargs become a plan directly, so there is
still exactly one construction path.

Only the guard *sequence* moved. `cli_common.py` (the guards) and
`backend_select.py` (backend resolution + context construction) are unchanged —
the builder composes them, and their own hostless tests stay authoritative for
the pieces.

---

## Session snapshot (`session_snapshot.py`, change `session-settings-owner`)

The persisted session — what `~/.skinny/settings.json` holds — has **one
owner**. `settings.py` owns the file; `session_snapshot.py` owns the schema.

`skinny` and `skinny-gui` each used to author their own 11-key dict, and
`save_settings` wrote it wholesale. The two key sets intersected in 6, so each
front-end **deleted five of the other's keys** on exit. Two fixes, at two
different boundaries:

1. **`save_settings` merges.** It loads the file, updates the supplied keys, and
   writes. One change protects every writer, including a future one that does not
   know the whole schema.
2. **One declared schema.** The front-ends contribute their own keys; they never
   author the dict.

| Section | Keys | Owner |
|---|---|---|
| shared | `params`, `camera`, `gizmo_mode`, `backend`, `encoding`, `sppm_glossy_roughness`, `neural_handoff`, `neural_trainer`, `train_precision`, `online_training` | the renderer — `capture_shared` / `restore_shared` |
| `GLFW_KEYS` | `vulkan_window` | `skinny` (`app.py::_contributed_session_state`) |
| `QT_KEYS` | `open_docks`, `last_dirs`, `section_states`, `qt_geometry`, `qt_dock_state` | `skinny-gui` (`ui/qt/app.py::MainWindow._contributed_session_state`) |

A contributed section is **opaque**: the module preserves a Qt geometry blob
without interpreting it. `contribute(shared, contributed, owned=…)` **refuses**
any key the caller's own section does not declare — per-front-end, not against
the union, so `skinny` cannot write a Qt-owned key that `skinny-gui` would erase
on its next exit. Declare a new key in `GLFW_KEYS` or `QT_KEYS` first.

`restore_shared` restores parameters, camera and gizmo mode with **independent
fault isolation**: a settings file that breaks one step still returns the others.
`backend` and `encoding` are declared here but restored by
`plan_bringup(persisted=…)` — they are session-fixed bring-up inputs, read before
a renderer exists. This module owns the key; the bring-up builder owns what a
startup value does.

`skinny-web` and `skinny-render` still do not persist. That asymmetry is the
bring-up decision above (`persisted=None`), and it stays.

Three divergences the duplication had grown, each now decided once:

- **Camera restore.** `skinny` raised `max_distance` to fit a restored orbit
  distance; `skinny-gui` clamped the distance to 50 and ignored the cap. The
  module keeps the `skinny` rule, because `max_distance` is the cap authority and
  the clamp destroyed a legitimately persisted wide view.
- **Captured parameter set.** `skinny` captured `build_visible_params`, which
  drops the fallback-light parameters under a scene with authored lighting —
  losing them permanently. Capture uses `build_all_params`; visibility governs
  display and restore, not storage.
- **Persisted flags.** `resolve_persisted_flag` + the `PERSISTED_FLAGS` table are
  the one precedence rule for the flags `cli_common` documents as persisted
  (`--neural-handoff`, `--neural-trainer`, `--train-precision`,
  `--online-training`, `--sppm-glossy-roughness`): an explicit CLI flag or
  environment variable wins, else the persisted value, else the argparse default.
  `skinny-gui` restored only the SPPM threshold before, so the documented
  persistence held on one front-end of two.

`tests/test_session_settings.py` is the gate: both erasure directions, the
corrupt-file case, a capture/restore round trip, the refusal, and each
front-end's contributed key set. The key-set check reads the front-end **source**
(AST), because neither front-end module imports without `vulkan` — an
import-based check would skip on a Metal-only host, and a skip reads as a pass.

---

## Python Modules

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `app.py` | `InputHandler` | GLFW shader-debug entry |
| `ui/qt/app.py` | `MainWindow` | `skinny-gui` entry — dock layout, menu, file open |
| `ui/qt/viewport.py` | `RenderViewport` | Qt widget that blits the renderer's offscreen image |
| `ui/qt/backend.py` | `QtTreeBuilder` | Walks the spec tree, instantiates Qt widgets |
| `ui/spec.py` | `Section`, `DynamicSection`, `Slider`, … | Pure dataclass widget tree |
| `ui/build_app_ui.py` | `AppCallbacks`, `build_main_ui` | Builds shared sidebar tree |
| `ui/panel/backend.py` | `PanelTreeBuilder` | Walks the spec tree, instantiates Panel widgets |
| `web_app.py` | `SkinnySession`, `VideoStreamHandler` | Panel web app, per-session renderer, Tornado video WS |
| `renderer.py` | `Renderer` | GPU resource orchestration, per-frame dispatch |
| `material_pack.py` | `pack_flat_material`, `pack_std_surface_params`, `pack_std_surface_params_msl`, `FLAT_MATERIAL_STRIDE`, `STD_SURFACE_STRIDE`, `MATERIAL_TYPE_*`, `MEDIUM_*` | Flat-material and standard-surface record packing, device-free (see [The device-free pure core](#the-device-free-pure-core-change-renderer-pure-core-extraction)) |
| `camera.py` | `CameraBase`, `OrbitCamera`, `FreeCamera`, `_perspective`, `_look_at` | Camera models and their projection/view math, device-free |
| `film_io.py` | `FilmParameters`, `_write_exr`, `_write_hdr_rgbe` | Film exposure controls and the HDR image writers, device-free |
| `sppm_budget.py` | — | SPPM photon-emission budget and group-selection pmf, device-free |
| `texture_pool.py` | `TexturePool` | Bindless flat-material texture pool; takes the backend's resource module, so it imports no GPU package |
| `skin_params.py` | `SkinParameters` | Layered-skin record and its std140 `pack`, device-free |
| `renderer_helpers.py` | — | Small device-free helpers: instance basis, light colour coercion, spectral proposal token |
| `mlt_chain.py` | — | MLT host chain state, device-free: replay seed (`next_seed`), mutation budget, uniform-tail predicate, and the bootstrap round-trip both backends drive (see [MetropolisLightTransport.md](MetropolisLightTransport.md)) |
| `frame_derive.py` | — | Pure frame-constant derivation consumed by `_pack_uniforms` at its append sites: detail-flag bitfield, lens FOV-framing sensor half-height, exposure/imaging-ratio fold, proposal-mask/reuse capability folding (byte serialization stays in the packer) |
| `vk_context.py` | `VulkanContext`, `SwapchainInfo` | Vulkan 1.3 instance, device, swapchain (+ headless mode) |
| `vk_compute.py` | `ComputePipeline`, `UniformBuffer`, `StorageImage`, `StorageBuffer`, `SampledImage`, `ReadbackBuffer`, `HudOverlay` | Shader compilation (Slang→SPIR-V), GPU resource types |
| `gfx/backend.py` | `Backend`, `BackendCaps` | Backend ABC |
| `gfx/device.py` | `Device` | Device abstraction |
| `gfx/presenter.py` | `Presenter` | Surface/swapchain abstraction |
| `gfx/vulkan/*` | — | Vulkan implementation |
| `scene.py` | `Scene`, `Material`, `MeshInstance`, `LightDir`, `LightSphere`, `LightEnvHDR` | Scene description dataclasses |
| `pbrt/` | `import_pbrt`, `tokenizer`, `parser`, `state`, `transform`, `spectra`, `materials`, `lights`, `camera`, `media`, `emit`, `metrics`, `parity` | pbrt v4 → USD importer (`skinny-import-pbrt`); see [PbrtImport.md](PbrtImport.md) |
| `materialx_runtime.py` | `MaterialLibrary`, `CompiledMaterial`, `UniformField` | MaterialX loading, GenSlang codegen, uniform reflection |
| `mesh.py` | `Mesh`, `MeshSource` | OBJ loading, subdivision, displacement, BVH construction |
| `mesh_cache.py` | — | On-disk BVH cache (zstd-compressed vertex/index/BVH blobs, `~/.skinny/mesh_cache/`) |
| `environment.py` | `Environment` | HDR env map loading (.hdr decoder), built-in presets |
| `params.py` | `ParamSpec`, `AccumStateProvider` | Shared parameter definitions, get/set helpers, persistence; accumulation-reset registry (`ParamSpec.resets_accumulation` + `ACCUM_STATE_PROVIDERS`); `ParamSpec.proxy_default` is the GUI-thread proxy's pre-snapshot placeholder (not the renderer's init value) |
| `choice_tables.py` | `Axis`, `labels`, `tokens`, `index_by_token`, `index_to_token` | Single owner of each enumerated render axis's values/labels/indices — integrator, tonemap, execution mode, reuse, detail-maps, ReSTIR combination, proposal preset; device-free (change `choice-table-owners`, see [The enumerated-axis owner](#the-enumerated-axis-owner-choice_tablespy-change-choice-table-owners)) |
| `hardware.py` | `GpuInfo`, `GpuVendor` | GPU enumeration, vendor detection, encoder selection |
| `video_encoder.py` | `VideoEncoder` | H264/JPEG encoding with hw-aware fallback, Annex B→AVCC |
| `scene_graph.py` | `SceneGraphNode`, `SceneGraphProperty`, `RendererRef` | USD prim hierarchy tree model with typed editable properties |
| `mtlx_graph_view.py` | `NodeGraphView`, `NodeView`, `PortView` | View-model for MaterialX nodegraph editor |
| `bxdf_math.py` | — | CPU BSDF eval + lobe rasterisation |
| `gizmo.py` | `TransformGizmo` | Transform gizmo math (rotate/translate × world/local) + line-list buffer |
| `lens_optics.py` | — | PBRT-v3 thick-lens helpers |
| `debug_viewport.py` | `DebugViewport` | Camera/lens/wireframe debug renderer |
| `presets.py` | `Preset` | 12 built-in skin presets (Fitzpatrick I–VI × Female/Male) |
| `settings.py` | — | Persistent storage at `~/.skinny/` (JSON); `save_settings` merges into the file on disk |
| `session_snapshot.py` | — | The persisted session schema: shared (renderer-owned) + front-end-contributed keys, capture/restore, one camera rule, persisted-flag precedence |
| `tattoos.py` | `Tattoo` | Procedural + image-based tattoo loading |
| `head_textures.py` | `TextureStats` | Detail map loading (normal, roughness, displacement) at 2048² |
| `scene_intake.py` | — | USD stage → `SceneUpdate` value; the one interface the renderer consumes |
| `usd_controls.py` | — | `ControlBinding` → renderer get/set closures (the applying half of the control seam) |
| `usd_loader.py` | — | USD stage → Scene (meshes, lights, cameras, materials, MaterialX fallback) |
| `fetch_hdrs.py` | — | Downloads CC0 HDRIs from Poly Haven |

---

## The enumerated-axis owner (`choice_tables.py`, change `choice-table-owners`)

`render_envelope.py` owns whether a combo is *valid*; `choice_tables.py` owns
what each axis is *called*. Each enumerated render axis is one ordered tuple of
`Axis(label, token)` records — the display label, the CLI/headless/persisted
string token where the axis has one, and the entry's index (its tuple position).
Every other view is a projection: the CLI's `choices` are `tokens(...)`, the
headless `str→index` dicts are `index_by_token(...)`, `render_envelope.INTEGRATORS`
/ `EXECUTION_MODES` and `frame_plan.INTEGRATOR_NAMES` are `tokens(...)` /
`index_to_token(...)`, the renderer's display lists (`integrator_modes`,
`tonemap_modes`, `reuse_modes`, `restir_combination_modes`, `detail_maps_modes`,
`execution_modes`, `_PROPOSAL_PRESETS`) are `labels(...)`, and the GUI-thread
proxy's placeholder names are `labels(...)`. The module is device-free and
imports nothing from `skinny`, so any layer imports it without a cycle. It owns
vocabulary, never validity.

Seven axes are owned: integrator, tonemap, execution mode, reuse, detail-maps,
ReSTIR combination, proposal preset. `DEFAULT_EXECUTION_FOR_INTEGRATOR` is a
projection too — its integrator keys come from the table and its
megakernel/wavefront values from `render_envelope.WAVEFRONT_ONLY_INTEGRATORS`.
The `EXECUTION_MEGAKERNEL`/`EXECUTION_WAVEFRONT` named indices in the four
device-free leaf modules (`params`, `frame_plan`, `frame_derive`, `mlt_chain`)
are each derived from the owner's index — they keep the readable name without a
GPU import cycle, and `tests/test_choice_tables.py` confirms every copy agrees.
`tests/test_choice_tables.py` also carries the AST source gate: scanning every
module under `src/skinny`, a list/tuple/dict literal whose string set equals an
owned axis's membership fails the build (two documented carve-outs — the ambiguous
`On`/`Off` detail-maps pair and `renderer.py`'s record-source token literal).
Adding an axis value edits one tuple.

The wavefront kernel-name and pass-constant owner is a separate follow-up change,
`choice-table-wavefront-owners`.

---

## Renderer carve-out pattern (change `renderer-module-carveout`)

`renderer.py` accumulated a decade of features onto one class. The carve-out
pattern peels a self-contained cluster off `Renderer` without changing a single
rendered pixel, so the pure logic becomes hostless-testable and the
backend-paired orchestration collapses behind the existing seams. The precedent
that proves it works is already in-tree: `wavefront_driver.py` holds the staged
loop once behind a duck-typed recorder, and `mlt_bootstrap.py` holds the pure
resample. Landed stages: `mlt_chain.py` (MLT host chain state),
`frame_derive.py` (frame-constant derivation), the wavefront pass factories
(`vk_wavefront.ensure_pass` / `metal_wavefront.ensure_pass`),
`gpu_resources.py` (the GPU resource inventory — the largest stage so far, 858
lines out), and the seven device-free modules of the **pure core** (see
[below](#the-device-free-pure-core-change-renderer-pure-core-extraction)).

**The five steps** — apply in order, one stage per PR:

1. **Identify the cluster's pure core** — the state→values computation with no
   device, no `self` mutation (seed math, a bitfield, a framing ratio, a
   capability fold). If a value is `f(scalars) → scalar`, it is pure.
2. **Extract it as module-level functions** with the renderer calling them at
   the *unchanged* call sites. No dataclass bundle unless one falls out free —
   per-site pure functions keep the diff mechanical and the byte stream
   provably identical. Side-effectful calls (buffer syncs, warn-once, stashes)
   keep their exact site and order; a computation that also warned returns a
   flag the renderer acts on.
3. **Move backend-paired orchestration behind the existing seams** — the
   `WavefrontRecorder` protocol for stage order, per-backend `build_pass` /
   `ensure_pass` factories for construction. No new abstraction layer, no
   removal of the mandated `is_metal` short-circuits (the `metal-backend` spec
   is preserved verbatim); the goal is *volume in one file*, not the seam's
   existence. Every None-fallback gate (an unbuildable pass → path/env
   fallback) moves verbatim — a dropped gate turns a graceful fallback into a
   crash.
4. **Gate with bit-identity + the parity matrix.** Prefer the strongest check
   the cluster admits: a golden byte-equality snapshot of the packed output
   (stronger than image parity, captured pre-refactor on the same commit,
   green after), or a bit-identical render (same seed ⇒ same image). Then the
   parity matrix must pass with **unchanged** measured values — no baseline or
   tolerance edits — and `git diff --stat` must show no `src/skinny/shaders/`
   change (RGB `.spv` byte-unchanged follows). For a device-bound move with no
   pure core, add key-equality unit tests + a runtime-toggle smoke so the
   rebuild-key paths the matrix doesn't exercise are still covered.
5. **One stage per PR**, independently landable and revertable.

**Follow-on order** (each a future OpenSpec change carrying its own gate):

| Cluster | Why the order | Gate |
|---------|---------------|------|
| Detail maps | Smallest, nearly pure (texture-stat → flags/strengths); a `frame_derive`-shaped extraction | Golden bytes + parity matrix |
| Gizmo overlay | Medium; segment-buffer math is pure, the upload is the seam | Bit-identical overlay render |
| USD live-edit | Largest; threads + async load + scene-graph mutation, so last | Scene-mutation parity + the scene-graph snapshot tests |

Sibling changes own adjacent scope and are **not** part of this pattern's
stages: `reflection-owned-byte-layouts` owns the values→bytes serialization
(`_pack_uniforms` / `_pack_uniforms_msl` offsets + MSL reflection);
`param-registry-accumulation-reset` owns the accumulation state hash. A
carve-out stage must route *around* both.

---

## The device-free pure core (change `renderer-pure-core-extraction`)

`renderer.py` imports `vulkan` at module scope. Every symbol above the
`Renderer` class therefore needed the Vulkan SDK to import — on a machine whose
default backend is Metal. The failure mode was **silent**: a test that could not
import the module skipped, and a skip looks like a pass in the run output. The
packers that produce the bytes the Metal backend uploads could only run where the
Vulkan SDK happened to be installed.

The module-scope core now lives in seven modules that import no GPU package.
Split by subject, not gathered into one container — each module has a consumer
that wants exactly it:

| Module | Owns | Consumer that wanted it |
|--------|------|-------------------------|
| `material_pack.py` | `pack_flat_material`, `pack_std_surface_params`, `pack_std_surface_params_msl`, the material strides, the `MATERIAL_TYPE_*` / `MEDIUM_*` codes, the override readers, the named-conductor id map | the flat-material layout gates, on either backend |
| `camera.py` | `_perspective`, `_look_at`, `_hero_yaw_pitch`, `_orbit_distance_cap`, `CameraBase` / `OrbitCamera` / `FreeCamera` | `debug_viewport.py` |
| `film_io.py` | `FilmParameters`, `_write_exr`, `_write_hdr_rgbe` | the parity harness |
| `sppm_budget.py` | `_sppm_photon_group_pmf`, `_sppm_photon_budget` | the SPPM selection tests |
| `texture_pool.py` | `TexturePool` | a hostless test with a fake resource module |
| `skin_params.py` | `SkinParameters` and its std140 `pack` | the skin path |
| `renderer_helpers.py` | `_instance_local_basis`, `_light_value_to_vec3`, `_spectral_analytic_proposal_token` | the gizmo, the light upload, the spectral gate |

**`renderer` re-exports every moved name**, so no source call site changed.
**Tests may not use that re-export.** A test that imports `skinny.renderer`
still drags in `vulkan`, so it demonstrates nothing about hostlessness; tests
import from the module that owns the symbol. `tests/test_pure_core_modules.py`
holds the gate. It imports each module in a subprocess in which every GPU
package (`vulkan`, `slangpy`) is blocked at the meta path, asserts no GPU
package reaches `sys.modules`, asserts no module imports `renderer` back, and
asserts — from the AST, so the check itself stays hostless — that `renderer`
re-exports every name the seven modules declare. Adding a module to the
device-free side means adding it to `PURE_MODULES` there.

`PURE_MODULES` has two halves (change `frame-plan-split`). `RE_EXPORTED_MODULES`
are the seven above, whose names `renderer` re-exports so pre-split call sites
still resolve. `MODULE_IMPORTED_MODULES` — `frame_derive`, `frame_plan`,
`mlt_chain` — `renderer` consumes as modules (`frame_plan.derive(...)`), so they
carry the same device-free obligation and none of the re-export one. Only the
first half is checked for re-exports.

The move is textual and changed nothing observable: every constant value, every
packed byte for a spread of materials, every camera matrix, both image-writer
outputs and the pool's slot behaviour were captured before and compared after.
Both backends render the same scene to a byte-identical PNG.

`TexturePool` moves even though it *holds* GPU objects, because it never
*imports* one — its constructor takes the backend's resource module. What stays
behind is anything the `Renderer` class body owns, plus the scene-record strides
(instance, sphere/distant light, emissive triangle), which have no consumer
outside `renderer.py` and so would gain a module without gaining a test.

---


## Per-frame path (`frame_plan.py`, change `frame-plan-split`)

Each frame runs three stages. Each stage has one owner.

| Stage | Owner | What it does |
|---|---|---|
| scene sync | `Renderer._sync_scene(dt)` | Advances every piece of scene state the frame reads: USD streaming, playback and animation, live-edit re-read, light recompute, the scene snapshot, the light and environment uploads, the mesh rebake, the tattoo upload. |
| frame plan | `frame_plan.derive(...)` | Derives the frame's decisions as a **value**: execution mode, integrator, step order, dispatch banding, MLT budget, and the optional per-frame work. |
| execute | `Renderer._execute_vulkan_frame(plan, target)` or `Renderer._render_scene_metal(plan)` | Records and submits the plan against a target. |

Before the split, one `update()` + `render()` pair carried roughly 34
responsibilities and `render_headless()` held a near-verbatim copy of the middle
of it. A change to the execution-mode gate had to be made twice.

**The plan holds no device handles.** It names passes, counts, flags and
decisions — never a buffer, a command buffer or a pipeline. `tests/test_frame_plan.py`
derives a plan and asserts its pass sequence with no GPU present, over every
integrator × execution mode × backend capability × target the render envelope
admits. `frame_plan.py` is in the device-free set above.

**Windowed and headless differ only in their target.** Both derive a plan and
call the same execution body. A target supplies exactly three things: where the
output goes, whether a swapchain image is acquired and presented, and whether a
readback follows. `_SwapchainTarget` acquires an image, contributes the
`UNDEFINED → TRANSFER_DST` barrier **to the same barrier call** so the recorded
command stream is unchanged rather than merely equivalent, blits, adds the
`PRESENT_SRC` transition, and presents. `_OffscreenTarget` contributes no extra
barrier, copies into the readback buffer, waits the fence, and returns the bytes.

**Ordering constraints are asserted, not implied.** `plan.steps` lists the
frame's steps in execution order and `frame_plan.ORDERING_INVARIANTS` states each
constraint with its reason; `check_invariants` runs on every derivation. Two
constraints earned their place by being violated during review:

- The pick-result drain precedes the uniform pack, because a satisfied pick that
  disarms after the pack disarms one frame late and fires twice.
- The fence reset sits immediately between `END_CMD` and `SUBMIT`. Anything that
  can raise in between leaves the fence unsignalled, and a caller that catches
  and retries blocks forever in the wait.

`plan.steps` is not yet replayed by the executor — that is
`gpu-backend-adapter`'s recording adapter. Until it lands,
`test_plan_step_order_matches_the_executor_source_order` pins the plan's order to
the shared body's source order, so the plan and the code cannot drift into two
unreconciled authorities.

**The plan is derived after the pick drain, and owns only what it can be
authoritative about.** `poll_pick_result` runs pick callbacks, and
`_on_autofocus_hit` sets `accum_frame = 0` — so a plan derived earlier would hand
the dispatch a `first_frame` that disagrees with the packed `fc.accumFrame`. For
the same reason the frame-end neural weight swap is **not** a plan field: arming
online training is a frame-end decision, and a start-of-frame snapshot would
defer an off-to-on transition by a frame. Those sites read `_online_training`
live.

**Banding is capability-driven.** A dispatch splits into row bands because the
backend's command buffers are watchdog-policed, not because the backend is Metal.
`frame_plan.megakernel_bands` takes `needs_watchdog_tiling`;
`Renderer._needs_watchdog_tiling` is the single place that capability binds to a
backend, and `gpu-backend-adapter` moves that binding into its capability record
without changing anything that consumes it.

**The accumulation reset keeps its owner.** `Renderer.update` decides it from the
`params.py` registry (change `param-registry-accumulation-reset`) and publishes
`accum_frame`; the plan consumes `accum_frame == 0` as `first_frame`, which drives
the SPPM first-frame flag and the MLT reseed.

---

## The backend seam (`gpu_backend.py`, change `gpu-backend-adapter`)

`backend_select.resource_module` documented that `vk_compute` and
`metal_compute` "expose the same public API". They did not, and the renderer
paid for the gap with 34 live `is_metal` branches plus two probes used as if
they were the seam.

`gpu_backend.py` owns what the seam is: the `BackendCapabilities` record read
through `capabilities(ctx)`, the `ONE_SIDED_MEMBERS` / `DIVERGENT_SIGNATURES`
tables, and the AST surface reader the conformance test runs on. The full map is
in [Backends.md § The declared seam](Backends.md#the-declared-seam-gpu_backendpy-change-gpu-backend-adapter);
what belongs here is the ownership rule it establishes.

**Branch on the reason, never on the vendor.** Each capability field names why a
consumer used to diverge — descriptor sets, frame sync objects, external memory,
external semaphore, in-place shared writes, the shared bindless sampler, the
merged record header, indirect dispatch, GPU skinning, the megakernel record
source, reflected record layouts, watchdog tiling, bindless capacity. A field
earns its place only by removing at least one existing branch, so the record
cannot grow into a config blob; a field the two device backends agree on is a
test failure, because it is either dead or misnamed — excluding
`has_indirect_dispatch`, a device probe either backend may report either way.
`has_external_semaphore` is the one field with no *pre-existing* branch behind
it: it encodes the branch that should have existed, since the CUDA handoff
needed both extensions and assumed one implied the other.

**Name the reason, not the binding model.** `has_descriptor_sets` was briefly
made to stand in for "is bind-by-name". That is the same answer on two backends
and the wrong one on a third, and it put a `make_sampler` reach in front of the
recording adapter, which has no such member. Two facts it was hiding — the
argument table splitting a combined `Sampler2D`, and the records compiling under
`SKINNY_METAL_RECORDS` — now own fields. On the renderer the record is
the memoised `self.caps` property, derived from `self.ctx` rather than assigned
in `__init__`, so a renderer built through `__new__` — the hostless test pattern
— still answers it.

**Never probe for a backend by attribute presence.** The removed
`hasattr(ctx, "compute_queue")` is the cautionary case: `MetalContext` sets that
attribute to `None` rather than omitting it, so the probe was *unconditionally
true* at all 7 sites and three wavefront pass factories were protected only by
the descriptor-set check on the next line. The failure was invisible because the
probe never returned `False` anywhere. A named capability read cannot fail that
way, and `tests/test_gpu_backend.py` fails if either probe returns.

**A divergence that survives is declared, not discovered.** One-sided members
and the one divergent signature live in tables beside the interface, each with
its reason, and the conformance test asserts the adapters agree *modulo those
tables* — including that no declaration is stale. This is the same discipline as
`shader_variants.METAL_ONLY_DEFINES` and `gpu_resources`' recorded orders: the
exception is a deliberate edit with a written reason, never a silent gap.

**The third adapter is where testability comes from.** `recording_compute`
records allocations, bindings and dispatches without a device, so dispatch
ordering and binding coverage become plain hostless assertions instead of
questions only a dual-device host could answer. It records, it does not
simulate — radiometry stays the parity matrix's job.
