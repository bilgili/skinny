# Design review — frame-plan-split

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: shrink to ~15 lines inside `gpu-backend-adapter`, or drop.** The
duplication is a tenth of what the proposal implies, the wavefront arm is
already deduplicated, Metal already has the shape the change wants to build, and
the plan cannot name the thing the spec requires it to name.

## Accurate as claimed

The 98 / 237 / 170 line counts; the duplicated block is *code-identical*
(diffing `10634-10673` against `10860-10899`, the only deltas are comments);
D4 (`_current_state_hash` at `:10442-10456` derives from the params registry,
and `accumulation-reset-registry` owns it); `frame_derive.py` exists and owns
detail flags / film half-height / exposure stops / capability folding;
`test_pack_uniforms_golden.py:11-13` is gpu-marked for the stated reason.

## MAJOR

**M1 — The duplication is ~27 code lines, and the wavefront arm is already
shared.** `_record_wavefront_dispatch` (`renderer.py:2310`) carries a docstring
saying it is *"shared by the windowed + headless Vulkan seams"*. What is actually
duplicated: the accum barrier (10 lines), `hud_overlay.record_copy` (1), the `if`
gate (4), and the megakernel arm (11) — 27 lines out of 237 + 170. A 15-line
`_record_frame_body(cmd, f)` helper removes it without a plan value, a target
abstraction, or a spec delta.

**M2 — Metal has no duplication; task 4.2's "and the Metal twins" has no
referent.** `_render_headless_metal` (`:9964-9971`) and `_render_windowed_metal`
(`:9973-9993`) both call `_render_scene_metal` (`:9872`), which holds the only
Metal execution-mode gate (`:9877`). Metal already differs only in target. So
"two or four places" is wrong — it is two duplicates plus one structurally
*different* Metal gate (Metal has no `_wavefront_debug_pass` arm). Cite Metal as
the existing proof that D2's shape works, and scope the change to Vulkan.

**M3 — D2 is false, and one of the differences changes pixels.** Beyond output
destination + acquire/present + readback, the two paths differ in:
1. **HUD** — windowed uploads fresh bytes (`:10609`); headless never calls
   `_build_hud_bytes` and copies stale/zero staging (`:10876`; the only upload
   sites are `:3730`, `:10609`, `:11009`). **A screenshot in a windowed session
   composites the previous frame's HUD.**
2. not-ready contract (`return` at `:10564` vs zero bytes at `:10808`);
3. pick drain before vs after the fence wait (`:10823` vs `:10590`);
4. the binding-1 rewrite (`:10833-10846`);
5. submit semaphores (`:10766-10774` vs `:10945-10948`);
6. headless blocks on its own fence post-submit (`:10951`); windowed returns
   with the frame in flight;
7. 4 image barriers + scaling blit vs 3 barriers + 1:1 readback copy;
8. return type.

Unifying (1) is not a refactor — adding HUD to headless or removing it from
windowed changes rendered output, contradicting the "same images" non-goal.
Decide the HUD explicitly and announce it; it is a live bug independent of this
change.

**M4 — D1 is not achievable as stated; the plan cannot name the execution mode
or the pass sequence.**
- *Pure-derivable:* HUD/splat/neural-swap flags, group counts, band count
  (`:9843`), `_has_heavy_nonflat` (`:9861`), `effective_execution_mode_index`
  (`:2145`) — but that last one is the **requested** mode, not the effective one.
- *Not derivable:* `_ensure_wavefront_pass(...)` returning `None` silently falls
  back to the megakernel (Metal `:9880-9885`) or to the path tracer / env pass
  (`:2320`, `:2328`, `:2337`, `:2340`) — the mode a frame actually runs is
  decided by a device-side pipeline **build**. `mlt.seeded` (`:2321`) is pass
  state produced by a GPU round-trip. `_backend_render_ready` (`:2163-2180`)
  reads `pipeline is not None`.
- *Structurally hostile:* `_run_wavefront_mlt_bootstrap` (`:2296-2308`) is called
  **inside** the frame command-buffer recording (after `vkBeginCommandBuffer` at
  `:10632`); it submits separate one-shot command buffers with
  `vkQueueWaitIdle` (`:2289-2293`), reads back weights, resamples on the host,
  and re-uploads `_pack_uniforms()` **twice** (`mlt_chain.py:99-112`).
- *Execute-only:* `vkAcquireNextImageKHR` → `image_index` selects the present
  semaphore; `swapchain_info.extent` is device-queried; Metal's
  `acquire_next_image()` returning `None` aborts a frame *after* the dispatch
  already ran (`:9986-9988`).

Either hoist pass-building above planning so the plan can state the effective
mode — a real behaviour change, say so — or downgrade the field to
`requested_mode` and drop "pass sequence" from the spec. State the MLT bootstrap
as an explicit exception to "one execution body".

**M5 — D5 names the wrong invariant.** The comment at `:10585-10589` states two
things: the drain must run *once its frame is fence-visible*, and *before*
`_pack_uniforms` so the disarm lands in this frame's UBO. The first is
load-bearing, and the two functions satisfy it differently — `render()` drains
after the fence wait (`:10580-10590`), `render_headless()` drains *before* it
(`:10823-10828`), correct only because headless fully drains at the end of every
call (`:10951`). Under MLT the pack runs up to three times per reset frame, and
the load-bearing one is the post-bootstrap re-pack carrying the GPU-measured
`b`. Assert fence-visibility, not line order, and record that the two functions
differ today so normalising them is a conscious decision.

**M6 — Gate 5.2 does not cover the path this change rewrites most.**
`parity.render_linear` drives `HeadlessRenderer._accumulate` →
`render_headless()` (`headless.py:222-227`). **The parity matrix never calls
`render()`.** The only windowed coverage in the repo is
`tests/test_metal_windowed_smoke.py` — Metal, megakernel, gpu-marked. Vulkan
windowed (acquire, blit, 4 barriers, semaphores, present, HUD upload) has
**zero** automated coverage, and it is the larger half of the diff. Add a Vulkan
windowed smoke and a same-state windowed-vs-headless offscreen equality check,
both blocking.

**M7 — The sharpest available identity gate is missing.**
`tests/test_pack_uniforms_golden.py` pins the `fc` blob byte-for-byte across a
state matrix — exactly the gate for a change that moves the pack relative to the
pick drain and the MLT bootstrap. Absent from §5, as are
`tests/test_frame_derive.py` and any `resize()` coverage (`resize` rewrites
bindings 1/2/3 at `:11040-11050`, and the split re-homes the offscreen target).

**M8 — The headline value is already bought by `gpu-backend-adapter`.** Its
Stage-3 requirement is *"A recording adapter makes dispatch hostlessly
assertable"* with the scenario "pass sequencing is asserted without a device" —
the same claim as this change's spec. Its Stage 4 migrates the renderer's raw
`vk.vkCmd*` calls, i.e. rewrites the same 27 lines. Under D6's "land last", the
adapter has already made the sequence assertable and already touched the
duplicate before this change starts. Fold tasks 4.1/4.2 into that Stage 4 as one
shared `_record_frame_body`, and either drop this change or reduce it to what
the adapter does not deliver: the `update()` scene-sync grouping and the
fence-visibility assertion.

**M9 — The stated dependency on `renderer-gpu-resource-set` contradicts that
change's own contract.** It promises *"the renderer keeps attribute access …
so no call site outside changes"*. If that holds, execute needs nothing from it.
The genuine overlap is small and one-directional: `_build_metal_binds` (called at
`:9826`, `:9907`, `:9917`, `:9938`) moves under it, and the headless binding-1
write (`:10833-10846`) is a descriptor write of the same family. Replace
"Depends on" with "sequence after, to avoid two changes editing
`_build_metal_binds` call sites". The `gpu-backend-adapter` dependency is the
real one.

**M10 — Spec and design contradict each other on the plan's inputs.** The spec
requires derivation "in a process with no GPU device"; the design leans toward
"the plan reading renderer state" — which means holding a `Renderer`, which
cannot be constructed without a device (precisely why the pack golden is
gpu-marked). Task 3.3 is unachievable under that leaning. Decide before
implementing: a frozen input struct built by a thin reader, or a documented
duck-typed protocol tests can stub.

## MINOR

- **The binding-rewrite verdict is neither of D2's two options.** Binding 1 is
  written to `_offscreen_output` at exactly two places — descriptor creation
  (`:4622-4629`) and resize (`:11040-11046`); no site ever binds it to a
  swapchain image, and `render()` says so explicitly (`:10611-10615`). The
  headless rewrite (`:10833-10846`) is **vestigial**, and its docstring
  (`:10799-10802`) documents behaviour that does not exist. Delete both as a
  separate announced one-liner; drop the "or a latent bug" false dichotomy.
- The cited line pairing is misaligned by one statement; correct correspondence
  is `10634-10673` ↔ `10860-10899`.
- Impact's Metal list omits `_render_megakernel_metal` (`:9820`) and
  `_metal_megakernel_bands` (`:9843`) — the two that own the banding decision
  task 3.2 is about — plus `_run_wavefront_mlt_bootstrap_metal` (`:9930`) and
  `_mlt_metal_chain_batch` (`:9953`).
- **Tiling is four knobs, not one:** `bands` (`:9839`), `bound_heavy_eye`
  (`:9916`), `photon_batch` (`:9924`), `chain_batch` (`:9911`); three read
  `os.environ` at call time inside the frame path (`:9849`, `:9961`). The
  wavefront passes also tile internally, so "any dispatch banding or tiling"
  overpromises — scope to the four host-level knobs.
- Step counts are internally inconsistent (19 vs 18 vs "roughly 34" vs 35);
  `render()`'s 16 omits the `_backend_render_ready` early return (`:10563`).
- "Metal twins at `:9820-9995`" is loose — that span holds six methods and ends
  inside `_pack_uniforms`; the twins proper are `:9872-9993`.
- **No task covers the public entry points.** `update()` and
  `render()`/`render_headless()` are called by four front-ends plus
  `headless._accumulate` (`:222-227`). State whether `update()` survives as a
  facade — if it does, the split is internal and M10 becomes mandatory.
