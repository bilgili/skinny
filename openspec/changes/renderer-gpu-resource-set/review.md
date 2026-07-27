# Design review — renderer-gpu-resource-set

Adversarial review, 2026-07-27, against the tree at `8247148`. Findings are
recorded here rather than folded into the proposal, so the original reasoning
and the objections to it both survive. **Fold before implementing.**

**Verdict: survives, shrinks.** The cluster is real and the deepening is sound,
but the headline justification is false and several inventory facts are wrong.

## MAJOR

**M1 — "Zero external callers" is false.** True for `src/`, not for `tests/`.
Five test modules reach into the cluster:
`tests/test_metal_megakernel_binding_map.py:103` (`_build_metal_binds`),
`tests/test_metal_megakernel_rebuild.py:77` (`_ensure_mesh_buffer_capacity`),
`tests/test_metal_foundation.py:290,308,309` (three rebind methods),
`tests/test_metal_material_preview.py:15` (doc ref), and — worst —
`tests/test_mlt_host.py:332`, which slices `renderer.py`'s **source text** with
`src.index("def _build_metal_binds")` as the terminator. Moving that method out
raises `ValueError: substring not found`. Re-anchor that slice on a symbol that
stays *before* the move.

**M2 — Task 3.3 deletes guards that are not backend guards.**
`_ensure_mesh_buffer_capacity` (`:4906`) and `_rebind_volume_descriptor`
(`:8522`) test `descriptor_sets is None` for a **Vulkan** reason: descriptor
sets are allocated lazily by `_create_descriptors`, called only from
`_build_pipeline_for_current_graphs` (`:3656`), so a large mesh load can grow
buffers before any set exists (documented at `:4899-4905`). The
not-yet-built guard must survive inside the Vulkan adapter.
`tests/test_metal_foundation.py:275-309` asserts these no-ops against recorded
crashes; port them onto the set.

**M3 — Three descriptor writers stay outside the set**, so task 5.2's grep gate
and the fourth spec scenario are unsatisfiable: `_upload_graph_param_buffers`
(`:6940`, also a fifth growth site — it reallocates at `:6879`),
`_bind_record_buffer` (`:11228`), and `render_headless` (`:10837`, a per-frame
write the design's own Non-Goals exclude). Either move them or name them as
owned exceptions.

**M4 — Binding 14's Metal side is not in `_build_metal_binds`.** The bindless
texture pool is passed per dispatch as `bindless=("flatMaterialTextures", …)`
at five sites (`:7302`, `:9826`, `:9907`, `:9917`, `:9938`), while Vulkan
writes it as binding 14 from `_update_texture_pool_descriptors` (`:8432`),
which *is* in the move list. Model it as a `bindless` channel, or exclude it
from scenario 2.

**M5 — One Metal name per declaration is not enough.** Sampled images emit
**two** names (`envMap` + `envMapSampler`, `:9787-9791`) against one Vulkan
combined-image-sampler binding, and Metal dict membership is *dynamic*:
`graphParamsCombined` only when non-None (`:9800`), seven spectral names only
under `self._spectral` (`:9805-9817`). Metal side must be a list, plus a
presence predicate.

**M6 — Seven resources are allocated outside `_init_gpu` and already destroyed
by `cleanup`.** `_graph_params_combined` (`:6879`), `_preview_image` /
`_preview_readback` / `_preview_pipeline` (`:7105/7110/7120`),
`_record_pipeline` (`:11324`), `_drain_buffer` (three sites: `:11379`,
`:11394`, `:11486`), `_scene_bindings` — plus `volume_density_image`
(`:8493`/`:8506`), `_metal_common_sampler` (allocated in `__init__` at `:1481`,
Metal-bound at `:9779`, never passes through `_init_gpu`), and four more
reallocated by `resize()` (`:10988-11008`). Task 4.1 would drop seven destroys.
Task 1.3 must diff **both** directions; D1 needs a "lazily created" lifetime tag;
`resize` joins the growth-site list.

**M7 — Binding is not available at construction.** `_init_gpu` runs once at
`:2036`; `_create_descriptors` runs from the *lazy* pipeline build (`:3656`) and
re-runs on every rebuild, reading `_scene_graph_fragments` (`:4266`),
`_spectral` (`:4283`), `mlt_bindings` (`:4284`), `BINDLESS_TEXTURE_CAPACITY`
(`:4259`), `_scene_set0_layout` (`:4296`). The set needs an explicit
`bind(target, *, set0_layout, graph_fragments, spectral, mlt_bindings, …)`
phase with its non-resource inputs listed.

**M8 — The two adapters have different cadences.** `_build_metal_binds` is
called per dispatch (five sites) and re-reads every buffer reference fresh —
which is *why* the four `_rebind_*` methods can no-op on Metal. A symmetric
single `bind(target)` would either cache the Metal dict across a realloc (stale
buffer → the documented crash) or write Vulkan descriptors every frame. State:
Vulkan bind is edge-triggered, Metal bind is level-triggered and must stay a
pure function over live references, uncached.

**M9 — Four of the eleven `is_metal` branches are allocation divergences, not
binding ones**: mtlx skin slot stride (`:3707`), neural handoff flags
(`:3842-3844`), semaphores (`:4202`), fences (`:4224`). They cannot fold into a
bind-time branch. The proposal's third What-Changes bullet contradicts its own
design Context.

## MINOR

- **Counts overstated ~1.8×.** Actual resource constructions in `:3671-4235`:
  **37** over the five named classes, **42** counting `SampledImage3D`,
  `ReadbackBuffer`, `HudOverlay`, `TexturePool`, `ExternalTimelineSemaphore` —
  not ~70. `cleanup` has **43** destroys, not ~35. The D1 trade-off gets
  cheaper, not worse.
- **"Five `_rebind_*` in `:4236-4912`" — four exist there** (`:4701`, `:4756`,
  `:4788`, `:4823`); `:4863-4912` is `_ensure_mesh_buffer_capacity`; the fifth
  rebind is `_rebind_volume_descriptor` at `:8516`. Correct early-return lines:
  `4709, 4760, 4793, 4829, 8444, 8522`.
- Citation drift: `_ensure_mesh_buffer_capacity` def is `:4863` not `:4877`;
  `_sync_volume_grid`/`_rebind_volume_descriptor` span `:8470-8548`.
- **"Disjoint line ranges" vs `frame-plan-split` is false.** Both changes claim
  `render_headless:10837` and the five Metal `bindless=` dispatch sites in
  `:9820-9995`. Assign owners.
- **The `render_headless` binding-1 rewrite is dead work.** Comments at
  `:10830-10832` and `:4307-4309` both claim `render()` points binding 1 at the
  acquired swapchain image. It does not — all three `dstBinding=1` writes
  (`:4624`, `:10841`, `:11042`) point at `_offscreen_output`, and `render()`
  blits instead (`:10611-10614`). This answers `frame-plan-split`'s open
  question from the code: redundant, both comments stale. Task 1.2's golden must
  not enshrine it.
- **Open question answered by the code: sync objects are not the same set.**
  `in_flight_fences`/`image_available` size by `MAX_FRAMES_IN_FLIGHT` but
  `render_finished` sizes by swapchain image count (`:4206-4220`) — a
  presentation quantity. All three are raw `vk` handles with no binding and no
  Metal counterpart. They belong to `frame-plan-split`'s target, not the
  resource set. `command_buffers` (`:4198`) is pool-owned and never destroyed in
  `cleanup` — `close()` must not touch it.
- **Task 1.1's fixture matrix undercounts.** The inventory is flag-dependent
  within a backend (`_spectral_*` at `:3790-3822`, `:3976`, `:4041`, `:4065`;
  `mtlx_skin_record_size` from MaterialX reflection at `:3700-3707`; the MLT
  pool arm at `:4284`). State axes as backend × {RGB, spectral} × {megakernel,
  wavefront}, or scope the golden to one cell and say so.
- **The write-order risk pins the wrong thing.** `_create_descriptors` issues
  one `vkUpdateDescriptorSets` with a single writes array (`:4699`); intra-array
  order is not observable. What is load-bearing is the *inter-method* sequence
  `_create_descriptors → _upload_graph_param_buffers →
  _update_texture_pool_descriptors → _upload_material_types` (`:3656-3659`),
  because binding 14 and `GRAPH_BINDING_BASE` are written by the later calls.
- **Line arithmetic:** moved ≈1,530 (the What-Changes list also moves
  `_build_metal_binds` 75, `_update_texture_pool_descriptors` 38,
  `_rebind_volume_descriptor` 33, `_rewrite_size_dependent_descriptors` ~40);
  net ≈1,450 after the residual non-resource setup `_init_gpu` keeps.
