# Design — renderer-module-carveout

## Context

`renderer.py` (12,114 lines) accumulated a decade of features onto one class.
Three facts shape this design:

- **The pattern already exists in-tree.** `wavefront_driver.py` (463 lines)
  holds the staged path/BDPT/SPPM/MLT stage order once, behind the duck-typed
  `WavefrontRecorder` protocol; each backend supplies a thin adapter
  (`vk_wavefront._VkPathRecorder`, `metal_wavefront`). It imports no GPU
  backend and is unit-testable with a recording stub. `mlt_bootstrap.py`
  (pure numpy `resample_chain_seeds`) proves the pure-core half. This change
  extends those two precedents; it invents nothing new.
- **The `is_metal` seam is a recorded decision.** `metal-backend` spec
  ("Vulkan-only host paths degrade safely on Metal") mandates every renderer
  path either run a Metal-equivalent or short-circuit on `is_metal`, and
  CLAUDE.md documents the `resource_module`/`is_metal` split as intentional.
  The target is *volume in one file* — 117 `is_metal`/`_metal` sites, 13
  `_metal` sibling methods — not the seam's existence.
- **Two siblings own adjacent scope.** `reflection-owned-byte-layouts` owns
  values→bytes serialization (`_pack_uniforms`/`_pack_uniforms_msl`,
  `pack_flat_material`, `pack_std_surface_params*`);
  `param-registry-accumulation-reset` owns `_current_state_hash`. This change
  must carve around both without touching their scope.

What actually remains renderer-resident per cluster (verified against source):

- **MLT**: `_next_mlt_seed` (cross-process-reproducible crc32 over
  `frame_index` — deliberately NOT the state hash, see its docstring),
  `_mlt_uniform_tail_active` (integrator==3 ∧ backend/mode/pass-built
  predicate), `_mlt_iterations_per_frame`, `_mlt_pass_key`,
  `_run_wavefront_mlt_bootstrap` + `_run_wavefront_mlt_bootstrap_metal`
  (identical host round-trip: seed → upload uniforms → bootstrap dispatch →
  weight readback → `resample_chain_seeds` → seed upload → init dispatch →
  set `b` → re-upload uniforms), `_ensure_wavefront_mlt_pass` +
  `_ensure_wavefront_mlt_pass_metal`, `_destroy_wavefront_mlt_pass`.
- **Derivation inside `_pack_uniforms`** (renderer.py:10452–10660): camera
  view/proj inverses; the lens FOV-framing ratio
  (`film_half_h_world *= filmDistance / (focal/mm_per_unit)`, :10559–10566);
  the detail-flag bitfield (:10512–10521); `exposure_ev = exposure +
  log2(imaging_ratio)`; emissive total power passthrough; the proposal-mask /
  reuse capability folding (megakernel neural-bit strip + mixture
  renormalisation, reuse-mode zeroing off wavefront, :10610–10639).
- **Backend method pairs**: `_ensure_wavefront_{path,bdpt,sppm,mlt}_pass` +
  `_metal` siblings; `_render_scene_metal` routing to
  `_render_wavefront{,_sppm,_mlt}_metal` / `_render_megakernel_metal`; the
  Vulkan twin is `_record_wavefront_dispatch(cmd, scene_set)`.

## Goals / Non-Goals

**Goals**

- MLT host orchestration in one module with a hostless-testable pure core.
- Every derived value in the frame-constant path computed by pure,
  device-free functions; `_pack_uniforms` keeps its side-effect call sites
  and append order and consumes the results as plain values.
- One ensure/dispatch path per wavefront integrator on the renderer; backend
  divergence confined to per-backend pass factories and the existing
  pass-object surface.
- A written extraction pattern + ordering for the remaining clusters (USD
  live-edit, gizmo, detail maps) as follow-on changes.
- Every stage independently landable, bit-identical, RGB `.spv` untouched.

**Non-Goals**

- **Packing / byte layout** — serialization, offsets, MSL reflection
  ownership stay with `reflection-owned-byte-layouts`. Stage 2 hands it a
  pure packer; it does not redesign one.
- **Accumulation state hash / parameter registry** — untouched; owned by
  `param-registry-accumulation-reset`.
- **Changing the `is_metal` seam** — no new backend abstraction layer, no
  removal of the mandated short-circuits. `MetalContext` keeps `descriptor_sets
  is None`; the metal-backend spec is preserved verbatim.
- No shader edits, no binding changes, no behavior changes, no performance
  work, no implementation of the follow-on clusters.

## Decisions

### D1 — Extraction order: MLT → derivation → pass seam → pattern doc

Ordered by blast radius, smallest first, each stage a PR:

1. **Stage A (MLT chain controller)** — most self-contained: the pure
   precedent (`mlt_bootstrap.py`) and the loop recording
   (`wavefront_driver.record_mlt_*`) already exist; this stage completes the
   family. Lowest risk, establishes the review shape for the rest.
2. **Stage B (frame derivation)** — **soft** ordering with
   `reflection-owned-byte-layouts`: the scopes are disjoint (byte-layouts v1
   touches module-level layout tables and `_pack_uniforms_msl`; Stage B
   touches only the value expressions inside `_pack_uniforms`), so the only
   real coupling is a textual merge conflict on the same method. Land Stage B
   first when practical to avoid the conflict; neither change semantically
   depends on the other, and the byte-layouts proposal is being updated to
   say the same.
3. **Stage C (wavefront pass seam)** — largest diff; benefits from A having
   already collapsed the MLT pair (the hairiest of the four).
4. **Stage D (pattern + follow-on ordering)** — docs only; ships the recipe
   for USD live-edit / gizmo / detail maps as future changes.

*Alternative considered*: one big-bang extraction. Rejected — un-reviewable
diff, and a single parity failure would be unbisectable across clusters.

### D2 — Module shapes

- **`src/skinny/mlt_chain.py`** — pure functions mirroring what exists:
  `next_seed(frame_index) -> int` (the exact crc32 formula — its
  cross-process reproducibility is what the parity gate's determinism rests
  on), `iterations_per_frame(width, height, num_chains) -> int`,
  `uniform_tail_active(integrator_index, is_metal, execution_mode,
  pass_built) -> bool`, plus `run_bootstrap(mlt, *, seed, upload_uniforms,
  submit)` — the shared round-trip both backends call with backend-supplied
  callables (Vulkan: `_submit_one_shot_compute` + scene-set-recording
  lambdas; Metal: the pass's own submits). `resample_chain_seeds` stays in
  `mlt_bootstrap.py` — no move for moving's sake.
- **`src/skinny/frame_derive.py`** — pure functions, one per derived value
  (`detail_flags(...)`, `film_half_height_world(...)`,
  `exposure_stops(...)`, `fold_sampling_capabilities(mask, alpha, reuse_mode,
  execution_mode) -> (mask, alpha, reuse_mode, neural_stripped)`), taking
  scalars/arrays, returning scalars/tuples. **No dataclass, no
  `FrameDerived` bundle** — `_pack_uniforms` calls each function at its
  existing append site, so the append order (which the Metal MSL relocation
  table at renderer.py:174 depends on) is untouched by construction.
  *Alternative considered*: a `derive_frame_constants(state)` snapshot object
  consumed by both packers. Rejected for this change — it re-shapes the
  packer, which is the sibling's scope; per-site pure functions are the
  smallest diff that makes derivation hostless-testable.
- **Side effects stay put.** `_pack_uniforms` today mutates state
  (`_sync_lens_buffer()`, `_warn_neural_megakernel_once()`, stashing
  `_sppm_metal_photon_batch`). Stage B extracts only computation; every
  side-effectful call keeps its exact call site and order. The neural-strip
  warning becomes a returned flag (`neural_stripped`) the renderer acts on.
- **Stage C pass seam** — no new protocol. The pass objects already share a
  surface per backend (`record_dispatch(cmd, scene_set, ...)` /
  `record_frame` on Vulkan, `dispatch_frame(binds=, uniform_blob=,
  bindless_textures=, ...)` on Metal). The carve-out is: (1) each of
  `vk_wavefront` / `metal_wavefront` exposes a factory
  `build_pass(integrator, ctx, cfg)` absorbing the construction bodies of the
  `_ensure_*` twins (including the Vulkan descriptor-set 52–57 rebind, which
  moves next to `WavefrontMltPass` where those bindings live); (2) the
  renderer keeps one `_ensure_wavefront_pass(integrator)` (cache + rebuild
  key + factory call) and one frame-dispatch site per backend *entry point*
  — the windowed/headless Metal encoder loop vs the Vulkan command-buffer
  recording genuinely differ and stay separate, per the metal-backend spec.
  Net: the 8 `_ensure_*` methods collapse to 1 + two factories; the 3
  `_render_wavefront_*_metal` bodies (`_render_wavefront_metal`, shared by
  path and BDPT, plus the SPPM and MLT variants) collapse into one that
  passes per-integrator kwargs. The rebuild keys (`_mlt_pass_key`, the
  reuse/neural/record-mode keys) move verbatim — they gate
  rebuild-on-change and must not change value.
  **The None-fallback gates are behavior, not construction**, and must
  survive the move exactly: `_ensure_wavefront_mlt_pass` returns None on
  `is_metal` (the Metal sibling routes elsewhere), on a context without
  `compute_queue`, on `_scene_bindings is None or descriptor_sets is None`,
  and on a scene set-0 layout without `mlt_bindings` (renderer.py:2495–2502)
  — that None is what makes a megakernel-mode MLT selection fall back to the
  path tracer instead of crashing. SPPM has the analogous unbuildable→None
  fallback. Each gate is enumerated and preserved (renderer-side or
  factory-side) in Stage C, never silently dropped.

### D3 — Bit-identity verification, per stage

- **All stages**: no file under `src/skinny/shaders/` is touched — asserted
  by `git diff --stat` in review; RGB `.spv` byte-unchanged follows trivially.
  The parity matrix GPU sweep (`tests/pbrt/test_parity.py -k matrix`, Metal
  backend per CLAUDE.md) must pass with **unchanged** measured values — no
  baseline edits, no tolerance changes.
- **Stage A**: hostless unit tests pin the exact seed values
  (`next_seed(0)`, `next_seed(1)`, …) so cross-process reproducibility can't
  drift; one MLT suite combo (`int_caustic`) re-rendered at equal budget must
  be bit-identical to pre-stage (same seed ⇒ same chains ⇒ same image — MLT
  is deterministic per seed on both backends).
- **Stage B**: a golden byte-equality test — capture `_pack_uniforms()` /
  `_pack_uniforms_msl()` blobs across a state matrix (lens on/off, detail
  maps on/off, each integrator, wavefront/megakernel, neural-on-megakernel
  strip case) *before* the refactor on the same commit, assert byte equality
  after. This is stronger than image parity — but it is **not hostless**:
  capturing the blobs requires constructed `Renderer`s, i.e. GPU sessions,
  and the execution mode is fixed per session, so the matrix needs two
  guarded processes (megakernel + wavefront), run one at a time under the
  metal-dispatch-hygiene rules. The extracted `frame_derive` unit tests are
  the hostless half; the golden-blob gate is a gpu-marked test.
- **Stage C**: wavefront images bit-identical pre/post on both backends
  (mega≡wave and Vulkan≡Metal anchors in the parity harness already enforce
  this at zero extra cost); the pass-rebuild keys unit-tested for value
  equality with the old key tuples.

### D4 — The extraction pattern (Stage D deliverable)

One page in `docs/Architecture.md`: (1) identify the cluster's pure core
(state→values, no device); (2) extract it as module-level functions with the
renderer calling at unchanged sites; (3) move backend-paired orchestration
behind the existing pass/recorder surfaces, construction into the backend
modules; (4) gate with golden bytes or bit-identical images + the parity
matrix; (5) one stage per PR. Follow-on order: detail maps (smallest,
pure-ish), gizmo overlay, USD live-edit (largest, threads) — each its own
OpenSpec change.

## Risks / Trade-offs

- **[Risk] Uniform append-order drift breaks the Metal MSL relocation table**
  (renderer.py:174 relocates fields from the Vulkan append order via a
  cumulative-size drift guard). → Mitigation: Stage B never moves an append
  statement — only the value expressions; the existing drift guard plus the
  golden byte-equality test both fire on any slip.
- **[Risk] Hidden side effects inside `_pack_uniforms` reordered** (lens
  buffer sync, SPPM photon-batch stash, warn-once). → Mitigation: catalogue
  them first (task 3.1); side-effectful calls keep their exact sites;
  extracted functions are pure by construction (no `self`).
- **[Risk] MLT seed drift silently destroys cross-process reproducibility**
  (parity gate re-renders in a fresh interpreter; a hash-based seed already
  caused pass-by-luck 0.17/0.25/1.10 relMSE historically). → Mitigation:
  formula moved verbatim; unit test pins exact integer outputs.
- **[Risk] Sibling collision on `_pack_uniforms`** with
  `reflection-owned-byte-layouts`. → Mitigation: the scopes are disjoint
  (D1); the residual risk is a textual merge conflict, avoided by the soft
  land-Stage-B-first ordering stated in both proposals' Impact sections.
- **[Risk] Renderer-source-asserting tests break on the move** —
  `tests/test_mlt_host.py:214–226` does
  `inspect.getsource(Renderer._next_mlt_seed)` and greps renderer source for
  the MLT ensure/destroy/dispatch wiring; Stage A's deletion of those bodies
  raises `AttributeError` in the test itself. The same file is touched by
  all three renderer-cluster changes (this one,
  `param-registry-accumulation-reset`, `reflection-owned-byte-layouts`). →
  Mitigation: explicit tasks (1.3a, 3.1) re-point its assertions to the new
  authority (`mlt_chain`, the pass factories); each sibling change carries
  the equivalent task for its own moves. Additionally, the MLT-seed
  independence docstring (renderer.py:2452, "deliberately NOT derived from
  `_current_state_hash`") that `param-registry-accumulation-reset` pledges
  to preserve targets `_next_mlt_seed`, which Stage A relocates — whichever
  change lands second owns updating the note's pointer.
- **[Risk] Stage C regresses a rebuild-key subtlety** (keys gate pass
  reconstruction on reuse/neural/record/dims changes; a wrong key = stale
  pass = wrong image only after a runtime toggle, which the parity matrix
  may not exercise). → Mitigation: keys move verbatim + key-equality unit
  tests; one manual runtime-toggle smoke (none↔ReSTIR, neural on/off) per
  backend recorded in the PR.
- **[Trade-off] Per-site pure functions instead of a derived-state object**
  — leaves `_pack_uniforms` long-ish until the sibling lands, but keeps this
  change's diff mechanical and byte-provable. Accepted.
- **[Trade-off] `_render_headless_metal` / `_render_windowed_metal` stay** —
  they are presentation plumbing mandated distinct by the metal-backend spec,
  not carve-out targets. Accepted; revisit only if a real need appears.

## Migration Plan

Each stage is independently landable, gated, and revertable:

1. **Stage A** — add `mlt_chain.py` + hostless tests; rewire the four MLT
   helpers and both bootstrap twins to call it; delete the renderer-resident
   bodies. Gate: hostless suite + MLT bit-identity render + parity matrix.
2. **Stage B** — golden-blob capture test first (gpu-marked, guarded runner,
   red/green on the same commit); add `frame_derive.py` + hostless tests;
   replace inline expressions per append site. Gate: golden bytes + parity
   matrix. Landing before `reflection-owned-byte-layouts` avoids the merge
   conflict on `_pack_uniforms` (soft ordering, D1).
3. **Stage C** — move pass construction into `vk_wavefront.build_pass` /
   `metal_wavefront.build_pass`; collapse `_ensure_*` to one method and the
   Metal wavefront render bodies to one; keys verbatim + key tests. Gate:
   parity matrix both backends + runtime-toggle smoke.
4. **Stage D** — Architecture.md pattern page + follow-on ordering;
   MetropolisLightTransport.md + Wavefront.md updates; CLAUDE.md pointer.
   Gate: docs review only.

Rollback: any stage reverts cleanly (no data, no persisted format, no shader
touched); no sibling change semantically depends on any stage here — the
Stage-B-first ordering is conflict avoidance only.

## Open Questions

- Should `_render_material_preview_metal` (preview dock) fold into the Stage
  C factory surface, or stay with the tool-dock cluster (it shares
  `_build_metal_binds` but belongs to `metal-tool-dock-render` scope)?
  Default: stays; revisit in the follow-on list.
- Does `_pack_uniforms_msl` consume `frame_derive` directly in Stage B, or
  only via the shared `_pack_uniforms` body it already relocates from?
  Default: via the shared body (zero extra diff); the sibling decides the
  final packer shape.
- Exact module names (`mlt_chain` vs `mlt_host`, `frame_derive` vs
  `frame_constants`) — bikeshed at implementation; the spec names the
  responsibilities, not the filenames.
