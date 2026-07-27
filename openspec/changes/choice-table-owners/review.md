# Design review — choice-table-owners

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: survives, shrinks, and is not risk-free.** The drift is real and
larger than counted, but the value is smaller than claimed and the consolidation
would silently disarm three existing build gates.

## MAJOR

**M1 — Eleven divergent placeholders, not six.** All 17 in
`render_session.py:214-231` checked against the renderer:

| placeholder | proxy | renderer |
|---|---|---|
| `scatter_modes` | `["BSSRDF","Volume"]` | 4 entries (`renderer.py:1615-1620`) |
| `integrator_modes` | 3 | 4 (`:1631`) |
| `proposal_preset_modes` | `["bsdf"]` | 5 labels (`:1643-1656`) |
| `reuse_modes` | `["Off"]` | `["None","ReSTIR DI"]` (`:1716`) |
| `coat_`/`spec_sampler_modes` | `["Default"]` | `["Native","Heitz-2018 basis VNDF"]` (`sampling/lobe_samplers.py:44-48`) |
| `diff_sampler_modes` | `["Default"]` | `["Native","Uniform-hemisphere"]` |
| `restir_regime_modes` | `["Initial"]` | 3 (`:1748`) |
| `restir_combination_modes` | abbreviated | `["Unbiased (GRIS)","Biased (ΣM)"]` (`:1754`) |
| `tonemap_modes` | `["Filmic"]` | 4 (`:1839`) |
| `detail_maps_modes` | `["Off"]` | `["On","Off"]` (`:1952`) |

Only `direct_light_modes` and `furnace_modes` match. The spec scenario naming
"the six divergent" would certify a partial fix as done.

**M2 — Four placeholders cannot come from any static table.** `presets`,
`environments`, `models`, `tattoos` are populated by `load_user_presets()` /
`load_environments(hdr_dir)` / `load_tattoos(...)` (`renderer.py:1517`, `:1568`,
`:1941`) — filesystem scans. Their stubs are *deliberate*. "None remain —
including … the 17 placeholder choice-name lists" is unsatisfiable; scope the
requirement to the static axes.

**M3 — D1's (token, index, label) shape does not fit three of the seven axes it
names.**
- **execution mode**: 3 CLI tokens (`cli_common.py:701`), 2 renderer indices
  (`params.py:86-88`), and **no label list at all**. `auto` has neither index nor
  label.
- **reuse**: `--reuse` accepts only `("none",)` (`cli_common.py:588`) while the
  renderer axis has 2 entries (`:1715`) — the CLI subset is deliberate. "No
  consumer may restate an axis's membership" either forces `--reuse restir-di`
  to become accepted (a CLI behaviour change the proposal disclaims) or needs a
  per-consumer subset field the design lacks.
- **walk modes**: `WALK_CHOICES` (`cli_common.py:52`) has an alias
  `megakernel→fused` (`:53`), so tokens are not 1:1 with entries.

Give the record an optional `cli_exposed` flag and allow `index=None`, or drop
those axes and say why.

**M4 — `WALK_MODES` is given two owners.** `cli_common.WALK_CHOICES` (`:52`),
`vk_wavefront.py:1726`, `metal_wavefront.py:917`, plus a 4th copy in
`tests/test_metal_wavefront_bdpt_ab.py:61`. D1 puts it in the axis table; D4
lists it among the 14 pass constants. Pick one — as written the two workstreams
collide.

**M5 — "No risk to the parity matrix" is false: consolidating deletes existing
drift gates.** `tests/pbrt/test_matrix.py:229-243` asserts
`set(headless._INTEGRATORS) == set(parity.INTEGRATORS)`. Repointing
`headless._INTEGRATORS` at the same table makes it `set(X) == set(X)` — a build
gate silently becomes tautological. Same hazard for
`test_coverage_meta_execution_modes_pinned` (`:246-247`) and for
`tests/test_render_envelope.py:49`, which deliberately re-transcribes
`INTEGRATORS` as the doc-sync gate. Add a task: every meta-test whose two sides
collapse must be rewritten against an independently transcribed golden — the
project's existing pattern — not deleted.

**M6 — Two tests regex-scrape the renderer source literal task 2.2 removes.**
`tests/test_mlt_selection.py:41` and `tests/test_sppm_selection.py:26` do
`re.search(r"integrator_modes:\s*list\[str\]\s*=\s*\[([^\]]*)\]", src)` and
assert index positions. Repointing `renderer.py:1631` breaks both; rewriting them
to import the table makes them vacuous (they exist to pin "SPPM must be the 3rd
entry"). Re-pin the positions as a literal golden.

**M7 — D3's "correcting labels is UI-only" is wrong for integrators.**
`renderer.py:9435` does `req_integ = self.integrator_modes[self.integrator_index]
.lower()` and feeds the observability config row, asserted in
`tests/test_online_training_observability.py`. Integrator labels are load-bearing
as CLI tokens today; the table's `token` must replace that `.lower()` derivation
in the same change.

**M8 — Task 2.4 is not implementable as written.** `ParamSpec`
(`params.py:41-58`) has **no default field**. The current code already iterates
`STATIC_PARAMS` and falls back to `float(param.lo)`
(`render_session.py:314-317`); the overrides at `:318-323` exist precisely
because `lo` ≠ default (`env_intensity` lo 0.0 / default 1.0, `params.py:128`;
`mm_per_unit` lo 1.0 / default 5.0, `:129`). Reading "the registry" requires
adding `default` to every spec **and** repointing `Renderer.__init__` — otherwise
it creates a fourth authority. The count is also wrong: 8 values are pre-seeded
*before* the registry loop (`:305-312`, mostly non-param keys) and **4** are
overrides after it. Scope 2.4 as its own change or drop it.

**M9 — D4's per-backend examples are not among the 14 constants, and the real
ones are unnamed.** "Metal rebuild keys carry extra elements" is
`metal_wavefront.py:1355`, a factory-local tuple already pinned by
`tests/test_wavefront_pass_keys.py` — the proposed new test is redundant.
"Record-stack sizing formula" is `metal_wavefront.py:543` vs
`vk_wavefront.py:688` — a formula that *consumes* `MAX_BOUNCES`, which is safe to
share.

What actually must stay per-backend: `metal_wavefront.py:307,308,905,906` are
commented **"reflection fallback only — MSL stride is authoritative"**
(`RESERVOIR_STRIDE`, `GBUF_STRIDE`, `VERTEX_STRIDE`, `AUX_STRIDE`), whereas the
Vulkan copies (`vk_wavefront.py:1382,1712,1713`) are the authoritative
`≥ sizeof` allocation strides. Equal today, different *meanings* — sharing them
means a Vulkan-motivated bump silently moves a Metal fallback. Name these four
as the "separate but pinned" set. Metal also has no counterpart for
`HIT_STRIDE` / `NEURAL_STRIDE` / `REC_VERTEX_STRIDE` (`vk_wavefront.py:595-597`)
— it derives them from reflection.

**M10 — "One home" already exists: `wavefront_layout`, and it is
shader-derived.** `wavefront_layout.py:97 REC_MAX_BOUNCES = 6` duplicates
`vk_wavefront.py:591` / `metal_wavefront.py:451`; and
`wavefront_layout.py:107 REC_VERTEX_STRIDE = rec_vertex_size()` (76 B, derived
from the `.slang` declaration under `reflection-owned-byte-layouts`) is
duplicated verbatim as the **hand-typed** `vk_wavefront.py:597
REC_VERTEX_STRIDE = 76`. A stale hand-copy of a derived value — a stronger catch
than most of the 14, and unmentioned. Conversely `VERTEX_STRIDE = 128` is **not**
`wavefront_layout.py:240 BDPT_VERTEX_STRIDE = 120` (padded allocation stride);
do not merge them. Route the layout-derived constants to the existing owner
rather than inventing a new home.

**M11 — The kernel-name table cannot deliver the claimed import-time failure.**
The spec promises "a stale name fails at import rather than producing a runtime
dispatch failure". A table of *strings* gives one edit point, not compile-time
verification: the string is passed to `slangc -entry <name>` at pass
construction (`vk_wavefront.py:485`) and indexed as `self._pipelines[entry]` at
dispatch (`:754`) — both runtime. Task 3.3's negative control would *disprove*
the claim it is meant to confirm. Reword to "one edit point", or make the
guarantee real with an import-time assertion that every table name resolves in
the compiled module.

**M12 — 37 kernels, not 34.** **33** appear in all three files (`wfPath*` 5,
`wfScatter`, `wfBuildArgs`, `wfBdpt*` 14, `wfSppm*` 8, `wfMlt*` 4). Beyond
those: `wfNeuralProposal` (vk+metal only) and `restirFill` / `restirSpatial` /
`restirResolve` (`vk_wavefront.py:1400,1490`; `metal_wavefront.py:274`),
duplicated across both backends but with **no driver consumer** — so "imported by
the driver and both backend pass modules" does not fit them. State 33-of-37.

**M13 — The value case is overstated: the drifted labels are a startup flash.**
`QtRendererProxy.apply_snapshot` (`render_session.py:335-336`) overwrites
`_choices` from every frame snapshot built by `choice_names_from_renderer`
(`:234-241`), and the Qt combo rebuild (`ui/qt/backend.py:307-315`) re-adds items
under `QSignalBlocker` and restores the persisted index — so a short placeholder
never clobbers a persisted value. "Labels that are already wrong in shipped UI"
should read "wrong for the frames before the first snapshot". That materially
changes the cost/benefit of reworking 17 entries.

## MINOR

- **Integrator copies are ≥7, not 4.** Unlisted:
  `cli_common.DEFAULT_EXECUTION_FOR_INTEGRATOR` (`:43` — a per-integrator
  attribute that belongs *in* the table), `renderer.integrator_modes` (`:1631`),
  `render_session.py:218`, `parity.INTEGRATORS`.
- **Existing table modules are the precedent D1 ignores.**
  `sampling/lobe_samplers.py` is a frozen dataclass with `name` / `cli_token` /
  `shader_id` / valid-lobes plus a `lobe_sampler_modes()` projection —
  exactly D1's shape. Three of the eleven divergent placeholders
  (`coat_`/`spec_`/`diff_sampler_modes`) are fixed by importing it, no new table.
  `params.RESOLUTION_PRESETS` is a second precedent.
- **Task 2.6's source gate must exempt `tests/`** — it would delete
  `tests/test_render_envelope.py:49` and the pinned goldens the project
  deliberately maintains as independent transcriptions (see M5).
- **`test_coverage_meta_app_integrators_covered` is skippable**
  (`tests/pbrt/test_matrix.py:234-237`, `pytest.skip` when headless import fails
  because vulkan is absent). Do not lean on it as the safety net for repointing
  `_INTEGRATORS`.
- **Cheapest fix in the whole proposal:** `headless.py:359` argparse choices
  duplicate `_TONEMAPS` keys in the same file → `choices=list(_TONEMAPS)`. Worth
  calling out as the pilot for D5.
