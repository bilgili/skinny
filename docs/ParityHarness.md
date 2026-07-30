# Skinny — Parity Matrix Harness

This document covers the standing regression that renders every valid
combination of integrator, execution mode, backend, and sampling axis, then
gates each one against pbrt v4 truth and against the wavefront path-tracer
anchor.

For the renderer overview see [Architecture.md](Architecture.md).

---

## Parity Matrix Harness (`pbrt/parity_core.py`, `pbrt/parity.py`, `pbrt/metrics.py`, `tests/pbrt/`)

A standing regression that renders every supported renderer **combination**
against a reference and against itself, so adding a feature re-tests all
renderers automatically.

**Two modules, one surface.** `pbrt/parity_core.py` holds the **pure, hostless**
half — the `SceneSpec` manifest schema + `load_manifest`, the `RenderCombo`
matrix and the `combo_is_valid` / `spectral_envelope` delegation to
`render_envelope`, the anchor/axis-class bookkeeping, the self-consistency
tolerance tables + `self_consistency_tol`, the result builders, and the
`render_log_path` helpers. It imports only stdlib, numpy, the capability flags,
`metrics` and `render_envelope`, so the matrix logic is exercisable with no GPU,
no renderer and no USD (`pxr`). `pbrt/parity.py` is the **GPU render adapter**
(`render_linear`, `render_combo`, `evaluate`, `scene_has_environment` and the
scene-source/env helpers), with the renderer *and* `import_pbrt` imported lazily
inside the functions that need them. It re-exports the core by an explicit import
block — including the consumed private names `_DEFAULT_SELF_CONSISTENCY`,
`_DEFAULT_SPECTRAL_SELF_CONSISTENCY` and `_render_log` — so every historical
`from skinny.pbrt.parity import …` resolves unchanged; `tests/pbrt/test_parity_core.py`
pins that surface.

The spectral tolerance table is **derived**, not duplicated:
`_DEFAULT_SPECTRAL_SELF_CONSISTENCY` = `_DEFAULT_SELF_CONSISTENCY` overlaid with
`_SPECTRAL_TOL_OVERLAY` (the only two rows that genuinely widen — `mode.relmse`
0.02→0.03 and `integrator.relmse` 0.06→0.09). A new tolerance class is therefore
written once, and a pinned-literal test fails on any drift in either table.

**Validity table (one source of truth).** A `RenderCombo(integrator,
execution_mode, proposals, reuse)` is a point in the matrix; `combo_is_valid`
prunes it — but it states no rule itself. Every envelope rule lives in
**`skinny/render_envelope.py`**, the shared render-envelope predicate: SPPM and
MLT are wavefront-only; the neural directional proposal is wavefront + path +
flat-material only (BDPT ignores it); ReSTIR DI direct-light reuse is path +
wavefront; spectral is flat-material only with no neural proposal and no reuse; a
scene flagged `megakernel_ok: false` (e.g. the 28.8M-tri dragon, which OOMs the
megakernel) is wavefront-only. The [Compatibility
matrix](RenderingModes.md#compatibility-matrix) documents that predicate.

`render_envelope.evaluate(query)` returns **every** violated rule as an ordered
list of `(code, reason)` pairs, never just the first, because its three consumers
need different ones:

| Consumer | Selects | Why |
|----------|---------|-----|
| `parity.combo_is_valid` / `spectral_envelope` | first violation in canonical order | reproduces the matrix's historical skip reasons and precedence |
| the four CLI guards in `cli_common.py` | the codes each guard owns (`CLI_GUARD_CODES`), in the guard's own precedence | `reject_mlt_unsupported` reports not-yet-wired *before* the megakernel refusal — the opposite of the matrix — and `path`+neural under the megakernel must be **accepted** (the renderer strips the bit at runtime) while `bdpt`+neural is refused, though both trip the same first rule |
| the renderer's spectral scene gate (`renderer.py`, material-pack time) | `RENDERER_SCENE_CODES` only | the CLI cannot see the material set; scanning only the flat-material code means the gate never newly refuses at runtime what is elsewhere a recorded skip |

Consumers own only code selection and prose — never a rule. Codes owned by no CLI
guard are recorded in `CLI_UNOWNED_CODES`, so one-sided acceptance is data, not an
oversight. The capability flags `SPECTRAL_IMPLEMENTED` / `MLT_IMPLEMENTED` stay
the kill-switches, read live at evaluation time. `tests/test_render_envelope.py`
holds a committed snapshot of the full cartesian query space (including the
capability-flag-off variants) as a permanent golden.

Every skip carries an explicit reason; `enumerate_combos(scene)` yields the valid
set, anchor-first. A coverage meta-test fails if an integrator the app exposes
(`renderer.integrator_modes`) has no table entry — a new integrator without a
predicate rule breaks the build, now for the CLI and the matrix at once.

**Dual gate.** Each valid combo renders once (linear-HDR accumulation) and feeds
two gates: **pbrt-truth** (`pbrt_truth_result` — exposure-aligned relMSE/FLIP vs
the checked-in pbrt v4 reference EXR, relaxed to a per-combo `baseline` when a
known mismatch is recorded) and **self-consistency** (`self_consistency_result`
— each combo vs the `(Path, wavefront)` anchor image at a per-axis tolerance:
tight for a pure `megakernel ≡ wavefront` mode change, looser for BDPT/SPPM,
unbiasedness for the neural/ReSTIR axes). Self-consistency never uses a baseline
escape, so a shared material bug (which makes both modes wrong identically) stays
green there while pbrt-truth records the delta. The **spectral axis** keeps the
same axis *class* against the megakernel spectral anchor but consults a separate
tolerance table (`_DEFAULT_SPECTRAL_SELF_CONSISTENCY` + a per-scene
`spectral_self_consistency` override): spectral wavefront is not bit-identical to
the megakernel (it threads the hero wavelengths through the staged records, a
different sample sequence), so mega≡wave is a decorrelated-but-unbiased MC delta
rather than the RGB bit-identity — measured on Metal and recorded harness-first.
The RGB tolerance table is never widened by a spectral override.

**Standard metric battery.** `metrics.compute_metrics(img, ref=None) ->
ImageMetrics` is the single place a number is computed: error vs reference (MSE,
RMSE, MAE, relMSE, PSNR, FLIP) plus single-image stats (variance, Immerkær
noise-σ, firefly outlier fraction). No call-site invents its own error formula.

**Corpus & references.** `tests/pbrt/corpus/manifest.json` lists scenes (pbrt
sources imported at gate time, or `.usda` assets loaded directly for the heavy
bathroom/dragon); `tests/pbrt/regen_refs.py` regenerates the reference EXRs
offline from the pinned pbrt v4. Tiers: `not gpu` (matrix construction + scene
import, runs anywhere), `gpu` (the full sweep), `slow` (higher-spp confirmation).

**Confirming-scene suite (`tests/assets/suite/`, `tests/pbrt/test_suite.py`).**
Minimal per-axis discriminating scenes — one lobe family / transport path /
sampling mode each — that fail *precisely* when their axis breaks, where the
heavy bathroom/dragon scenes would bury the defect in noise. They register in the
same `manifest.json` as `usd:`-source entries (path resolved relative to the repo
root) and are swept by the same validity table + dual gate, plus two suite-only
gate classes:
- **authoring equivalence** (`authoring_equivalence_result`) — every scene is
  authored twice, a plain `UsdPreviewSurface` `.usda` and a MaterialX
  `_mtlx.usda`+`.mtlx`; the two must render within tolerance (they render
  bit-identically in practice). OpenPBR-only PBR-material scenes record an
  equivalence *skip* (no UsdPreviewSurface counterpart).
- **white-furnace closure** (`pbrt/furnace.py`) — a lossless material under
  `furnace_index` must vanish into the constant furnace environment; the gate
  measures spatial **uniformity** (not an absolute `== 1.0`, since the furnace
  env carries its own integrator-dependent radiance constant) with recorded
  baselines, plus a per-material-flag probe.

Scenes are generated by `tests/assets/suite/_gen/` — the pbrt-expressible ones by
writing a tessellated-`trianglemesh` `.pbrt` and importing it through
`import_pbrt` (so the `.pbrt` / plain `.usda` / MaterialX `.usda` are provably the
same scene, and pbrt renders the identical triangles — no analytic-vs-tessellated
mismatch); the OpenPBR PBR-material scenes by extracting `standard_surface`
parameters from the `assets/materialxusd` cards onto a shared shaderball. Suite
reference EXRs regenerate via `regen_refs.py --scene suite`. Coverage meta-tests
in `test_suite.py` fail the build if a suite scene lacks a disposition for any
applicable gate class (pbrt-truth / equivalence / furnace).

---
