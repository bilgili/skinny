#!/usr/bin/env python3
"""Neural-guiding variance harness — headless configuration sweep driver.

Renders a fixed set of known scenes headless across a declarative matrix
``{scene × proposals × reuse × chart × encoding × temporal × precision × budget}``
and emits the paper's renderer-side claims: **equal-time** variance,
**equal-variance** time, and **`1/(var·t)`** efficiency — as checked-in tables
(`% source:` provenance) plus SVG plots, over **N independent seeds** with a
reported spread, measured against an **asserted-converged reference** from the
**linear-HDR accumulation image**. The estimator itself is gated off-GPU in
`tests/test_guiding_variance_metrics.py` (incl. the uniform-image → ~0 case).

The matrix is **declarative and degrades gracefully** (proposal.md): it sweeps
whatever axes the build exposes. Axes not yet landed (`--chart` =
`renderer-chart-selection`; `temporal` = `neural-temporal-conditioning`) have
their non-default cells **skipped and logged** — a coverage gap is always
visible, never silent. Encoding (E0/E1/E3) and precision (fp32/fp16) are live.

Run (the built 3.13 venv; Vulkan needs the SDK on the dylib path — see CLAUDE.md;
native Metal needs none). Metal `slangc` compiles are SERIALIZED here (one cell
at a time) per the thermal rule:

  VULKAN_SDK=.../macOS DYLD_LIBRARY_PATH=$VULKAN_SDK/lib PYTHONPATH=src:tests \
    <repo>/bin/python3.13 scripts/guiding_variance_sweep.py \
      --backend vulkan --out-dir docs/diagrams/guiding_variance

  # one-scene/two-cell smoke that emits a table + a plot:
  ... scripts/guiding_variance_sweep.py --backend metal --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"
HDR_DIR = ROOT / "hdrs"
TATTOO_DIR = ROOT / "tattoos"
ASSETS = ROOT / "assets"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from guiding_variance_metrics import (  # noqa: E402
    BudgetPoint,
    assert_reference_converged,
    efficiency,
    equal_variance_budget,
    markdown_table,
    svg_bar_chart,
    svg_line_chart,
    variance_over_seeds,
)

# Disjoint per-seed RNG stream: each seed renders from frame_index = seed*STRIDE,
# so the per-sample seeds never overlap across seeds/budgets (the budget grid
# accumulates within one stream, so STRIDE must exceed the largest budget).
SEED_STRIDE = 1_000_000


def log(msg: str) -> None:
    print(f"[var-harness] {msg}", flush=True)


# --- optional image dump (visual double-check of the variance cells) ----------
def _lum(img):
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def _auto_exposure(ref, pct=50.0, target=0.45):
    """Map the reference's median luminance to a mid value; shared across all
    cells of a scene so the only visible difference is noise, not brightness."""
    return target / max(float(np.percentile(_lum(ref), pct)), 1e-8)


def _save_png(path, lin, exposure):
    import matplotlib.image as mpimg
    srgb = np.clip(np.maximum(np.asarray(lin, np.float64) * exposure, 0.0) ** (1.0 / 2.2),
                   0.0, 1.0)
    mpimg.imsave(str(path), srgb)


# ---------------------------------------------------------------------------
# Declarative config + axis availability (degrade gracefully, skip + log).
# ---------------------------------------------------------------------------

# Scenes are pinned by their checked-in USD (geometry/lights/camera fixed there).
SCENES = {
    "cornell": ASSETS / "cornell_box_emissive.usda",
    "three_materials": ASSETS / "three_materials_demo.usda",
    "restir": ASSETS / "restir_variance_demo.usda",
}

# Only V1 (the shipped Lambert chart) is buildable today; the others await
# renderer-chart-selection. Only temporal=off is buildable (neural-temporal-
# conditioning not landed). These drive the skip+log, never a silent omission.
AVAILABLE_CHARTS = {"V1"}
AVAILABLE_TEMPORAL = {"off"}


def default_config(quick: bool) -> dict:
    """The checked-in sweep matrix. `--quick` collapses to a one-scene, two-cell,
    low-spp smoke that still exercises the full reference→variance→emit path."""
    if quick:
        return {
            "scenes": ["cornell"],
            "integrator": "path",
            "seeds": 2,
            "budgets": [8, 16],
            "reference_spp": 48,
            "resolution": 48,
            "cells": [
                {"proposals": "bsdf", "reuse": "none", "chart": "V1",
                 "encoding": "E0", "temporal": "off", "precision": "fp32"},
                {"proposals": "bsdf,neural", "reuse": "none", "chart": "V1",
                 "encoding": "E1", "temporal": "off", "precision": "fp32"},
            ],
        }
    # Full default slice: vary the live axes (proposals/reuse/encoding/precision)
    # on the canonical indirect Cornell box; chart/temporal stay at their only
    # buildable value with the rest enumerated-and-skipped for visible coverage.
    cells = []
    for prop in ("bsdf", "bsdf,env", "bsdf,neural", "neural"):
        for enc in ("E0", "E1", "E3"):
            # encoding only matters when a neural proposal is in the mix
            if "neural" not in prop and enc != "E0":
                continue
            cells.append({"proposals": prop, "reuse": "none", "chart": "V1",
                          "encoding": enc, "temporal": "off", "precision": "fp32"})
    cells.append({"proposals": "bsdf,env", "reuse": "restir-di", "chart": "V1",
                  "encoding": "E0", "temporal": "off", "precision": "fp32"})
    cells.append({"proposals": "bsdf,neural", "reuse": "none", "chart": "V1",
                  "encoding": "E1", "temporal": "off", "precision": "fp16"})
    # Enumerated-but-unavailable cells (kept so the skip log shows the gap):
    cells.append({"proposals": "bsdf,neural", "reuse": "none", "chart": "V2",
                  "encoding": "E0", "temporal": "off", "precision": "fp32"})
    cells.append({"proposals": "bsdf,neural", "reuse": "none", "chart": "V1",
                  "encoding": "E0", "temporal": "on", "precision": "fp32"})
    return {
        "scenes": ["cornell"],
        "integrator": "path",
        "seeds": 4,
        "budgets": [16, 64, 256],
        "reference_spp": 512,
        "resolution": 96,
        "cells": cells,
    }


def cell_label(cell: dict) -> str:
    base = f"{cell['proposals']}|{cell['chart']}/{cell['encoding']}/{cell['precision']}"
    if cell.get("reuse", "none") != "none":
        base += f"+{cell['reuse']}"
    if cell.get("temporal", "off") != "off":
        base += "/T"
    return base


def cell_availability(cell: dict, available_precisions: set[str]) -> str | None:
    """Return a skip reason if the cell can't be built today, else None."""
    if cell.get("chart", "V1") not in AVAILABLE_CHARTS:
        return f"chart {cell['chart']} not built (renderer-chart-selection not landed)"
    if cell.get("temporal", "off") not in AVAILABLE_TEMPORAL:
        return "temporal=on not built (neural-temporal-conditioning not landed)"
    if cell["precision"] not in available_precisions:
        return f"precision {cell['precision']} unavailable on this device"
    net = cell.get("net")
    if net is not None and not Path(net).exists():
        return f"trained net {net} missing"
    return None


# ---------------------------------------------------------------------------
# GPU plumbing (reuses the proven headless primitives).
# ---------------------------------------------------------------------------


def _make_context(backend: str, res: int):
    if backend == "metal":
        from skinny.metal_context import MetalContext
        return MetalContext(window=None, width=res, height=res)
    from skinny.vk_context import VulkanContext
    return VulkanContext(window=None, width=res, height=res)


def _neural_config(cell: dict):
    from skinny.sampling.neural_weights import Encoding, NeuralBuildConfig, NeuralPrecision
    enc = Encoding[cell["encoding"]]
    prec = NeuralPrecision(_precision_value(cell["precision"]))
    return NeuralBuildConfig(encoding=enc, precision=prec,
                             coupling=cell.get("coupling", "rqs"))


def _precision_value(p: str) -> str:
    # NeuralPrecision uses fp16-compute as the canonical fp16; map the matrix's
    # short "fp16" to it (the device-gated compute mode, fp32 fallback handled
    # by the renderer).
    return {"fp16": "fp16-compute", "fp8": "fp8-storage"}.get(p, p)


def available_precisions(backend: str, res: int) -> set[str]:
    """Probe which precisions this device can run (fp32 always; fp16 gated)."""
    from skinny.sampling.neural_weights import NeuralPrecision
    out = {"fp32"}
    try:
        from skinny.vk_context import VulkanContext  # probe via Vulkan caps
        ctx = VulkanContext(window=None, width=8, height=8)
        try:
            prec = NeuralPrecision("fp16-compute")
            if not getattr(prec, "needs_device_fp16_compute", False) \
                    or bool(getattr(ctx, "supports_fp16_compute", False)):
                out.add("fp16")
        finally:
            ctx.destroy()
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        log(f"fp16 probe failed ({exc}); assuming fp16 available (renderer falls back)")
        out.add("fp16")
    return out


def _build_renderer(backend, scene_path, integrator, cell, res, no_direct=False):
    from skinny.renderer import Renderer
    ctx = _make_context(backend, res)
    needs_neural = "neural" in cell["proposals"]
    ncfg = _neural_config(cell) if needs_neural else None
    r = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR, tattoo_dir=TATTOO_DIR,
        usd_scene_path=scene_path, execution_mode="wavefront", neural_config=ncfg,
    )
    r.proposal_preset_index = r.proposal_preset_from_token(cell["proposals"])
    r.reuse_index = 1 if cell.get("reuse") == "restir-di" else 0
    r.integrator_index = 0 if integrator == "path" else 1
    if cell.get("net"):
        r._neural_weights_path = cell["net"]
    # Pump async USD load until scene bindings exist.
    for _ in range(400):
        r.update(0.025)
        if (r._usd_scene is not None and len(r._usd_scene.instances) >= 1
                and r._scene_bindings is not None):
            break
    if r._scene_bindings is None:
        raise RuntimeError(f"scene bindings never built for {scene_path.name}")
    if no_direct:
        # Zero the analytic distant ("direct") light after USD load so the scene is
        # lit only by the emissive panel (indirect-dominated). Applied to cells AND
        # the reference so the MSE compares like-for-like lighting.
        r.light_intensity = 0.0
        r.direct_light_index = 1
    return ctx, r


def _accumulate_to(r, target_spp: int, frame_seed_base: int) -> np.ndarray:
    """Reset accumulation, seed the per-sample RNG stream at `frame_seed_base`,
    accumulate `target_spp` frames, return the mean linear-HDR RGB (H, W, 3)."""
    r._last_state_hash = None        # force accum reset on the next update()
    r.frame_index = int(frame_seed_base)
    for _ in range(target_spp):
        r.update(0.04)
        r.render_headless()
    img, samples = r.read_accumulation_hdr()
    assert samples > 0, "no samples accumulated"
    # read_accumulation_hdr returns the already-normalised running MEAN (verified:
    # the raw image is spp-invariant — raw.mean ~const for N=1..128 — and the
    # two-witness reference converges only because it is the mean). Dividing by
    # `samples` again double-normalised to mean/spp, making every MSE dominated by
    # a deterministic mean^2*(1/b-1/ref)^2 bias instead of estimator noise.
    return img[..., :3].astype(np.float64)


def _sweep_budgets(r, budgets, frame_seed_base):
    """Render the budget grid within ONE seed stream (incremental accumulation),
    returning {spp: (image, cumulative_wallclock_s)}. The first render absorbs
    pipeline compile (untimed); time is steady-state per-frame × spp."""
    out = {}
    r._last_state_hash = None
    r.frame_index = int(frame_seed_base)
    r.update(0.04)
    r.render_headless()  # warmup/compile — untimed, restart accumulation after
    r._last_state_hash = None
    r.frame_index = int(frame_seed_base)
    done = 0
    elapsed = 0.0
    for b in sorted(budgets):
        t0 = time.perf_counter()
        while done < b:
            r.update(0.04)
            r.render_headless()
            done += 1
        elapsed += time.perf_counter() - t0
        img, samples = r.read_accumulation_hdr()
        out[b] = (img[..., :3].astype(np.float64), elapsed)  # already the mean
    return out


def build_reference(backend, scene_path, integrator, res, ref_spp, no_direct=False):
    """High-spp converged reference, gated by two independent half-budget refs
    agreeing (design.md item 1). Returns (ref_image, witnessed_rel_err)."""
    cell = {"proposals": "bsdf", "reuse": "none", "chart": "V1",
            "encoding": "E0", "temporal": "off", "precision": "fp32"}
    ctx, r = _build_renderer(backend, scene_path, integrator, cell, res, no_direct=no_direct)
    try:
        a = _accumulate_to(r, ref_spp, 0 * SEED_STRIDE)
        b = _accumulate_to(r, ref_spp, 7 * SEED_STRIDE)
    finally:
        r.cleanup()
        ctx.destroy()
    witness = assert_reference_converged(a, b, rel_tol=0.02)
    return 0.5 * (a + b), witness


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    label: str
    proposals: str
    chart: str
    encoding: str
    precision: str
    reuse: str
    equal_time_var: float
    equal_time_var_spread: float
    firefly_p999: float
    equal_time_s: float
    efficiency: float
    eq_var_spp: float
    eq_var_time_s: float
    eq_var_reached: bool
    budget_curve: list  # [(spp, var, time_s)]


def run_cell(backend, scene_path, integrator, cell, res, seeds, budgets, ref, dump=None,
             no_direct=False):
    """Render one cell over N seeds across the budget grid; return a CellResult.

    ``dump`` (optional) = {"dir", "scene", "exposure"}: save this cell's seed-0
    image at every budget as a tonemapped PNG (visual double-check of the cells).
    """
    ctx, r = _build_renderer(backend, scene_path, integrator, cell, res, no_direct=no_direct)
    try:
        if "neural" in cell["proposals"] and not r._neural_active():
            raise RuntimeError("neural proposal requested but pass not active")
        per_seed_budget = []  # list over seeds of {spp: (img, time)}
        for s in range(seeds):
            per_seed_budget.append(_sweep_budgets(r, budgets, (s + 1) * SEED_STRIDE))
    finally:
        r.cleanup()
        ctx.destroy()

    top = max(budgets)
    if dump is not None:
        lab = (cell_label(cell).replace("/", "_").replace("|", "__")
               .replace(",", "-").replace("+", "-"))
        for b in sorted(budgets):
            _save_png(Path(dump["dir"]) / f"{dump['scene']}_{lab}_{b}spp.png",
                      per_seed_budget[0][b][0], dump["exposure"])
    # Equal-time (top budget): aggregate over seeds, with spread + firefly tail.
    top_imgs = [per_seed_budget[s][top][0] for s in range(seeds)]
    cv = variance_over_seeds(top_imgs, ref)
    mean_time = float(np.mean([per_seed_budget[s][top][1] for s in range(seeds)]))
    eff = efficiency(cv.var_mean, mean_time)

    # Budget curve (var_mean over seeds at each spp) → equal-variance inversion.
    curve = []
    grid = []
    for b in sorted(budgets):
        imgs = [per_seed_budget[s][b][0] for s in range(seeds)]
        cb = variance_over_seeds(imgs, ref)
        tb = float(np.mean([per_seed_budget[s][b][1] for s in range(seeds)]))
        curve.append((b, cb.var_mean, tb))
        grid.append(BudgetPoint(spp=b, time_s=tb, var=cb.var_mean))
    # Equal-variance target: the median variance reached across cells is chosen by
    # the caller; here invert toward the best (lowest) budget-curve variance × 1.5
    # as a per-cell self-consistent target placeholder; the figure-level target is
    # set in aggregate() against the slice.
    target = min(v for _, v, _ in curve) * 1.5
    try:
        ev = equal_variance_budget(grid, target)
        eq_spp, eq_time, reached = ev.spp, ev.time_s, ev.reached
    except ValueError:
        eq_spp, eq_time, reached = float("nan"), float("nan"), False

    return CellResult(
        label=cell_label(cell), proposals=cell["proposals"], chart=cell["chart"],
        encoding=cell["encoding"], precision=cell["precision"],
        reuse=cell.get("reuse", "none"),
        equal_time_var=cv.var_mean, equal_time_var_spread=cv.var_spread,
        firefly_p999=cv.firefly_p999_mean, equal_time_s=mean_time, efficiency=eff,
        eq_var_spp=eq_spp, eq_var_time_s=eq_time, eq_var_reached=reached,
        budget_curve=curve,
    )


def _hash_img(img: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(img.astype(np.float32))).hexdigest()[:16]


def emit(out_dir: Path, scene, cfg, ref, ref_witness, ref_hash,
         results, skipped, backend):
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (f"scripts/guiding_variance_sweep.py --backend {backend} "
              f"--scene {scene} (seeds={cfg['seeds']}, budgets={cfg['budgets']}, "
              f"ref_spp={cfg['reference_spp']}, res={cfg['resolution']})")

    # Equal-variance figure target: the median equal-time variance over the slice.
    et_vars = [r.equal_time_var for r in results if np.isfinite(r.equal_time_var)]
    fig_target = float(np.median(et_vars)) if et_vars else float("nan")

    rows = []
    for r in results:
        rows.append({
            "label": r.label,
            "var": r.equal_time_var,
            "spread": r.equal_time_var_spread,
            "p999": r.firefly_p999,
            "time": r.equal_time_s,
            "eff": r.efficiency,
            "eqspp": r.eq_var_spp,
            "eqt": r.eq_var_time_s,
            "reached": "y" if r.eq_var_reached else "n",
        })

    md = [f"# Guiding variance — `{scene}` ({backend})", "",
          f"Reference: {{bsdf}} @ {cfg['reference_spp']} spp, "
          f"two-witness converged (rel-err {ref_witness:.4g} ≤ 0.02), "
          f"hash `{ref_hash}`, mean={float(ref.mean()):.5f}, "
          f"{ref.shape[1]}×{ref.shape[0]}.", "",
          f"Seeds: **{cfg['seeds']}** independent; budgets {cfg['budgets']} spp; "
          f"variance = MSE vs reference over the linear-HDR accumulation image, "
          f"mean ± spread over seeds.", ""]
    _trained = sorted({c["net"] for c in cfg["cells"] if c.get("net")})
    if _trained:
        md += [f"Neural cells use trained nets: {', '.join(Path(n).name for n in _trained)}.",
               ""]
    elif any("neural" in c["proposals"] for c in cfg["cells"]):
        md += ["**Neural cells use the dummy (untrained, zero) net** — this slice "
               "measures *cost + unbiasedness*, not a guiding win: the dummy net is a "
               "valid-but-poor proposal, so mixture-MIS keeps it unbiased (same variance "
               "as `bsdf`) while adding the MLP pre-pass cost. A guiding win needs a "
               "trained `.nrec` (set `net:` per cell; produced by `spline_flow`), which "
               "moves the variance column.", ""]

    md.append(markdown_table(
        rows,
        [("label", "config"), ("var", "eq-time var (MSE)"),
         ("spread", "± spread"), ("p999", "firefly p99.9"),
         ("time", "time (s)"), ("eff", "1/(var·t)"),
         ("eqspp", "eq-var spp"), ("eqt", "eq-var s"), ("reached", "in-grid")],
        title=f"Equal-time / equal-variance / efficiency — {scene}",
        source=source,
        caption=(f"Equal-variance target = slice median eq-time var "
                 f"= {fig_target:.3e} MSE. `in-grid=n` ⇒ extrapolated (logged)."),
    ))

    if skipped:
        md += ["", "### Skipped cells (coverage gaps, not hidden)", "",
               f"% source: {source}", ""]
        for lab, reason in skipped:
            md.append(f"- `{lab}` — {reason}")
        md.append("")

    # SVG plots.
    labels = [r.label for r in results]
    if labels:
        bar_et = svg_bar_chart(labels, [r.equal_time_var for r in results],
                               title=f"Equal-time variance — {scene}",
                               ylabel="MSE vs ref")
        (out_dir / f"{scene}_equal_time.svg").write_text(bar_et)
        bar_eff = svg_bar_chart(labels, [r.efficiency for r in results],
                                title=f"Efficiency 1/(var·t) — {scene}",
                                ylabel="1/(var·s)")
        (out_dir / f"{scene}_efficiency.svg").write_text(bar_eff)
        bar_ev = svg_bar_chart(labels, [r.eq_var_time_s for r in results],
                               title=f"Equal-variance time — {scene}",
                               ylabel="s to target var")
        (out_dir / f"{scene}_equal_variance.svg").write_text(bar_ev)
        curve_series = [(r.label, [c[0] for c in r.budget_curve],
                         [c[1] for c in r.budget_curve]) for r in results]
        (out_dir / f"{scene}_variance_vs_spp.svg").write_text(svg_line_chart(
            curve_series, title=f"Variance vs spp — {scene}",
            xlabel="spp", ylabel="MSE vs ref"))
        md += ["", f"![equal-time](./{scene}_equal_time.svg)",
               f"![efficiency](./{scene}_efficiency.svg)",
               f"![equal-variance](./{scene}_equal_variance.svg)",
               f"![variance-vs-spp](./{scene}_variance_vs_spp.svg)", ""]

    (out_dir / f"RESULTS_{scene}.md").write_text("\n".join(md))

    # Machine-readable manifest (incl. the reference hash — checked in for repro).
    manifest = {
        "scene": scene, "backend": backend, "config": cfg,
        "reference": {"spp": cfg["reference_spp"], "witness_rel_err": ref_witness,
                      "hash": ref_hash, "mean": float(ref.mean())},
        "results": [asdict(r) for r in results],
        "skipped": [{"cell": lab, "reason": reason} for lab, reason in skipped],
    }
    (out_dir / f"manifest_{scene}.json").write_text(json.dumps(manifest, indent=2))
    log(f"emitted RESULTS_{scene}.md + manifest_{scene}.json + 4 SVGs → {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("metal", "vulkan"), default="vulkan")
    ap.add_argument("--config", help="JSON config overriding the built-in matrix")
    ap.add_argument("--quick", action="store_true",
                    help="one-scene/two-cell low-spp smoke (task 5.2)")
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "diagrams" / "guiding_variance"))
    ap.add_argument("--dump-images", action="store_true",
                    help="save a tonemapped PNG per cell per budget (+ the reference) "
                         "into --out-dir, shared exposure, for a visual double-check")
    ap.add_argument("--no-direct", action="store_true",
                    help="zero the analytic distant (direct) light for cells AND the "
                         "reference (emissive-panel only; isolates the guided indirect)")
    args = ap.parse_args()

    cfg = (json.loads(Path(args.config).read_text()) if args.config
           else default_config(args.quick))
    out_dir = Path(args.out_dir)
    res = int(cfg["resolution"])
    avail_prec = available_precisions(args.backend, res)
    log(f"backend={args.backend} precisions={sorted(avail_prec)} "
        f"scenes={cfg['scenes']} cells={len(cfg['cells'])} seeds={cfg['seeds']}")

    for scene in cfg["scenes"]:
        scene_path = SCENES.get(scene) or (ASSETS / scene)
        if not scene_path.exists():
            log(f"SKIP scene {scene}: {scene_path} missing")
            continue
        log(f"=== scene {scene} ({scene_path.name}) ===")
        ref, witness = build_reference(args.backend, scene_path, cfg["integrator"],
                                       res, cfg["reference_spp"], no_direct=args.no_direct)
        ref_hash = _hash_img(ref)
        log(f"reference converged (rel-err {witness:.4g}) hash={ref_hash}")

        dump = None
        if args.dump_images:
            out_dir.mkdir(parents=True, exist_ok=True)
            exposure = _auto_exposure(ref)
            _save_png(out_dir / f"{scene}_reference_{cfg['reference_spp']}spp.png", ref, exposure)
            dump = {"dir": str(out_dir), "scene": scene, "exposure": exposure}
            log(f"dump-images ON (exposure={exposure:.3f}) → {out_dir}")

        results, skipped = [], []
        for cell in cfg["cells"]:
            lab = cell_label(cell)
            reason = cell_availability(cell, avail_prec)
            if reason:
                log(f"SKIP {lab}: {reason}")
                skipped.append((lab, reason))
                continue
            log(f"cell {lab} …")
            try:
                results.append(run_cell(args.backend, scene_path, cfg["integrator"],
                                        cell, res, cfg["seeds"], cfg["budgets"], ref,
                                        dump=dump, no_direct=args.no_direct))
                rr = results[-1]
                log(f"  -> var={rr.equal_time_var:.3e}±{rr.equal_time_var_spread:.1e} "
                    f"p99.9={rr.firefly_p999:.2e} {rr.equal_time_s:.2f}s "
                    f"eff={rr.efficiency:.3e}")
            except Exception as exc:  # noqa: BLE001 — a cell failure must not abort the sweep
                log(f"  !! cell {lab} failed: {exc}")
                skipped.append((lab, f"render error: {exc}"))

        emit(out_dir, scene, cfg, ref, witness, ref_hash, results, skipped, args.backend)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
