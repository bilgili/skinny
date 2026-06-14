"""Variance / efficiency metrics for the neural-guiding variance harness.

PURE numpy — **no renderer, no GPU import** — so the MAX-rigor measurement core
(the part whose outputs become paper claims) is unit-testable off-device. The
GPU sweep driver (`scripts/guiding_variance_sweep.py`) feeds rendered linear-HDR
accumulation images into these functions; `tests/test_guiding_variance_metrics.py`
pins their behaviour on synthetic inputs (incl. the 3.5 estimator-validation
case: a uniform image → ~0 variance).

The headline efficiency metric is `1/(var·t)` — **identical** to
`spline_flow`'s `ParametrizationResults §5` and to the in-renderer 6.3 gate in
`test_neural_headless.py` — so renderer and synthetic-study numbers compare
directly. `var` is the per-pixel squared error vs the converged reference,
aggregated as the mean over pixels with a firefly percentile reported alongside
(the heavy tail the paper cares about).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Reference convergence gate (design.md item 1: a variance number against an
# unconverged reference is meaningless — gate on it before any comparison).
# ---------------------------------------------------------------------------


def rel_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Global energy difference ``|mean(a) - mean(b)| / mean(b)``.

    Noise-robust: zero-mean Monte-Carlo noise cancels in the spatial mean, so
    this exposes systematic BIAS only. The same statistic `test_neural_headless`
    uses for its unbiasedness gates."""
    mb = float(np.mean(b))
    return abs(float(np.mean(a)) - mb) / max(abs(mb), 1e-8)


class ReferenceNotConverged(AssertionError):
    """Raised when a reference render fails its convergence gate."""


def assert_reference_converged(
    ref_a: np.ndarray,
    ref_b: np.ndarray | None = None,
    *,
    running_err: float | None = None,
    rel_tol: float = 0.01,
) -> float:
    """Gate a reference before it is used as ground truth.

    Two accepted witnesses (design.md item 1):

    * **Two independent high-spp references agree** — pass `ref_b`; the gate is
      ``rel_mean_diff(ref_a, ref_b) <= rel_tol``.
    * **Running error below threshold** — pass `running_err` (the reference's own
      converged-vs-half-budget relative error); the gate is
      ``running_err <= rel_tol``.

    Returns the witnessed relative error. Raises `ReferenceNotConverged`
    otherwise so the harness refuses to report numbers against a bad reference.
    """
    if ref_b is None and running_err is None:
        raise ValueError("need either a second reference or a running_err witness")
    witness = running_err if ref_b is None else rel_mean_diff(ref_a, ref_b)
    if not math.isfinite(witness) or witness > rel_tol:
        raise ReferenceNotConverged(
            f"reference not converged: witnessed rel-err {witness:.4g} > tol {rel_tol:.4g}"
        )
    return float(witness)


# ---------------------------------------------------------------------------
# Per-cell error / variance vs the reference (task 3.1).
# ---------------------------------------------------------------------------


def image_mse(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean squared error over all pixels & channels — the per-cell ``var``."""
    d = np.asarray(img, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return float(np.mean(d * d))


def firefly_percentile(img: np.ndarray, ref: np.ndarray, p: float = 99.9) -> float:
    """The ``p``-th percentile of the per-element absolute error — the firefly
    tail the paper reports alongside the mean (the heavy outliers a bare MSE
    hides)."""
    d = np.abs(np.asarray(img, dtype=np.float64) - np.asarray(ref, dtype=np.float64))
    return float(np.percentile(d, p))


def bulk_mse(img: np.ndarray, ref: np.ndarray, clip: float) -> float:
    """Firefly-robust bulk MSE: both images clamped to ``clip`` before the MSE,
    so the heavy outliers don't swamp the bulk signal (matches the 6.3 gate)."""
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, clip)
    b = np.clip(np.asarray(ref, dtype=np.float64), 0.0, clip)
    d = a - b
    return float(np.mean(d * d))


@dataclass
class CellVariance:
    """A cell's error/variance aggregated over seeds — mean **and** spread, never
    a single-run value (design.md item 2)."""

    n_seeds: int
    var_mean: float          # mean over seeds of per-seed image MSE vs ref
    var_spread: float        # std over seeds of that MSE (the reported spread)
    firefly_p999_mean: float  # mean over seeds of the firefly percentile
    seed_pixel_var_mean: float  # mean over pixels of the per-pixel variance across seeds
    mean_radiance: float     # mean of the seed-averaged image (sanity)

    def as_row(self) -> dict:
        return {
            "n_seeds": self.n_seeds,
            "var_mean": self.var_mean,
            "var_spread": self.var_spread,
            "firefly_p999": self.firefly_p999_mean,
            "seed_pixel_var": self.seed_pixel_var_mean,
            "mean": self.mean_radiance,
        }


def variance_over_seeds(
    seed_images: list[np.ndarray] | np.ndarray,
    ref: np.ndarray,
    *,
    firefly_p: float = 99.9,
) -> CellVariance:
    """Aggregate a cell's per-seed images into mean ± spread (task 3.1).

    `seed_images` is N independent renders of the SAME cell at the SAME budget
    (distinct RNG seeds). Returns:

    * `var_mean` / `var_spread` — mean & std over seeds of the per-seed MSE vs
      `ref` (the headline variance with its spread);
    * `firefly_p999_mean` — mean over seeds of the firefly percentile;
    * `seed_pixel_var_mean` — mean over pixels of the variance computed *across*
      the seed images (the estimator's own per-pixel variance, independent of the
      reference — this is what goes to ~0 on a deterministic/uniform input, the
      3.5 validation case).
    """
    imgs = np.asarray(seed_images, dtype=np.float64)
    if imgs.ndim < 3:
        raise ValueError("seed_images must be a stack of images (N, H, W[, C])")
    n = imgs.shape[0]
    if n < 1:
        raise ValueError("need at least one seed image")
    per_seed_mse = np.array([image_mse(imgs[i], ref) for i in range(n)])
    per_seed_ff = np.array([firefly_percentile(imgs[i], ref, firefly_p) for i in range(n)])
    # Per-pixel variance ACROSS seeds (reference-independent estimator variance).
    pixel_var = np.var(imgs, axis=0)  # population variance over the seed axis
    return CellVariance(
        n_seeds=int(n),
        var_mean=float(np.mean(per_seed_mse)),
        # Sample std over seeds (ddof=1); 0 for a single seed.
        var_spread=float(np.std(per_seed_mse, ddof=1)) if n > 1 else 0.0,
        firefly_p999_mean=float(np.mean(per_seed_ff)),
        seed_pixel_var_mean=float(np.mean(pixel_var)),
        mean_radiance=float(np.mean(imgs)),
    )


# ---------------------------------------------------------------------------
# Equal-time / equal-variance / efficiency (tasks 3.2 – 3.4).
# ---------------------------------------------------------------------------


def efficiency(var: float, t: float, eps: float = 1e-30) -> float:
    """`1/(var·t)` — the headline efficiency, identical to
    `ParametrizationResults §5`. Higher = better. Guarded against var/t→0."""
    return 1.0 / (max(float(var), eps) * max(float(t), eps))


@dataclass
class BudgetPoint:
    """One point on a cell's budget sweep: `spp` samples cost `time_s` wallclock
    and reach error `var`."""

    spp: int
    time_s: float
    var: float


@dataclass
class EqualVariance:
    """Result of inverting a budget sweep for a target variance (task 3.3)."""

    target_var: float
    spp: float
    time_s: float
    reached: bool  # False ⇒ the grid never hit the target; spp/time are extrapolated/clamped
    note: str = ""


def equal_variance_budget(
    grid: list[BudgetPoint],
    target_var: float,
) -> EqualVariance:
    """Spp + wallclock for a cell to reach `target_var` (task 3.3).

    Monte-Carlo error falls as ``var ∝ 1/spp``, so we interpolate in **log-log**
    (`log var` vs `log spp`) between the two bracketing budget points — the
    correct shape for a variance-vs-spp curve — and read wallclock off a linear
    `time ∝ spp` model from the same bracket. If the target lies outside the
    measured grid, the result is clamped/extrapolated and `reached=False` so the
    table never silently claims a budget it didn't measure.
    """
    pts = sorted((p for p in grid if p.spp > 0 and p.var > 0 and np.isfinite(p.var)),
                 key=lambda p: p.spp)
    if len(pts) < 2:
        raise ValueError("need at least two valid budget points to invert")
    vmin = min(p.var for p in pts)
    vmax = max(p.var for p in pts)
    # Bracket in log-log. var decreases as spp increases.
    lo = hi = None
    for a, b in zip(pts, pts[1:]):
        v_hi, v_lo = max(a.var, b.var), min(a.var, b.var)
        if v_lo <= target_var <= v_hi:
            lo, hi = a, b
            break
    reached = vmin <= target_var <= vmax
    if lo is None or hi is None:
        # Outside the grid → extrapolate from the nearest segment.
        lo, hi = (pts[0], pts[1]) if target_var > vmax else (pts[-2], pts[-1])
    lx0, lx1 = math.log(lo.spp), math.log(hi.spp)
    ly0, ly1 = math.log(lo.var), math.log(hi.var)
    if ly1 == ly0:
        spp = float(lo.spp)
    else:
        frac = (math.log(target_var) - ly0) / (ly1 - ly0)
        spp = math.exp(lx0 + frac * (lx1 - lx0))
    # time ∝ spp from the bracket's per-sample cost (mean of the two endpoints).
    rate = 0.5 * (lo.time_s / max(lo.spp, 1) + hi.time_s / max(hi.spp, 1))
    time_s = spp * rate
    return EqualVariance(
        target_var=float(target_var), spp=float(spp), time_s=float(time_s),
        reached=bool(reached),
        note="" if reached else "target outside measured grid — extrapolated",
    )


# ---------------------------------------------------------------------------
# Output: transcribable tables + SVG plots (tasks 4.1 / 4.2).
# ---------------------------------------------------------------------------


def markdown_table(
    rows: list[dict],
    columns: list[tuple[str, str]],
    *,
    title: str,
    source: str,
    caption: str = "",
) -> str:
    """A checked-in result table with a ``% source:`` provenance note for
    transcription into the paper (task 4.1). `columns` is a list of
    ``(key, header)``; values are formatted with `_fmt`."""
    out = [f"### {title}", ""]
    out.append(f"% source: {source}")
    if caption:
        out += ["", caption]
    out += ["", "| " + " | ".join(h for _, h in columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |")
    return "\n".join(out) + "\n"


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def _svg_header(w: int, h: int) -> str:
    # Theme-aware via CSS vars, matching the repo diagram convention; falls back
    # to readable defaults when rendered standalone.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="-apple-system, sans-serif" font-size="11">\n'
        f'  <style>text{{fill:var(--fg,#222)}} '
        f'.axis{{stroke:var(--muted,#888);stroke-width:1}} '
        f'.bar{{fill:var(--accent,#3b7dd8)}} '
        f'.curve{{fill:none;stroke-width:1.8}}</style>\n'
    )


def svg_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    width: int = 520,
    height: int = 300,
) -> str:
    """A bar chart (equal-time / equal-variance bars, task 4.2) as standalone
    SVG per the repo's "SVG not ASCII-art" diagram convention."""
    if len(labels) != len(values):
        raise ValueError("labels/values length mismatch")
    pad_l, pad_r, pad_t, pad_b = 56, 16, 34, 64
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    vmax = max([v for v in values if np.isfinite(v)] + [1e-30])
    n = max(len(values), 1)
    bw = plot_w / n * 0.7
    gap = plot_w / n
    s = [_svg_header(width, height)]
    s.append(f'  <text x="{width/2:.0f}" y="18" text-anchor="middle" '
             f'font-weight="600">{_esc(title)}</text>\n')
    s.append(f'  <text x="14" y="{pad_t+plot_h/2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 14 {pad_t+plot_h/2:.0f})">{_esc(ylabel)}</text>\n')
    # axis
    s.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
             f'y2="{pad_t+plot_h}"/>\n')
    s.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" '
             f'x2="{pad_l+plot_w}" y2="{pad_t+plot_h}"/>\n')
    for i, (lab, v) in enumerate(zip(labels, values)):
        vv = v if np.isfinite(v) else 0.0
        bh = (vv / vmax) * plot_h if vmax > 0 else 0
        x = pad_l + i * gap + (gap - bw) / 2
        y = pad_t + plot_h - bh
        s.append(f'  <rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                 f'height="{bh:.1f}"><title>{_esc(lab)}: {vv:.3e}</title></rect>\n')
        s.append(f'  <text x="{x+bw/2:.1f}" y="{pad_t+plot_h+14:.0f}" '
                 f'text-anchor="end" transform="rotate(-35 {x+bw/2:.1f} '
                 f'{pad_t+plot_h+14:.0f})">{_esc(lab)}</text>\n')
        s.append(f'  <text x="{x+bw/2:.1f}" y="{y-3:.1f}" text-anchor="middle" '
                 f'font-size="9">{vv:.2e}</text>\n')
    s.append('</svg>\n')
    return "".join(s)


def svg_line_chart(
    series: list[tuple[str, list[float], list[float]]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = True,
    logy: bool = True,
    width: int = 560,
    height: int = 320,
) -> str:
    """Variance-vs-spp/time curves (task 4.2). `series` is a list of
    ``(label, xs, ys)``. Log axes by default (the natural scale for a
    variance-vs-spp curve)."""
    palette = ["#3b7dd8", "#d8693b", "#3bb273", "#9b59b6", "#d8b53b", "#555"]
    pad_l, pad_r, pad_t, pad_b = 60, 110, 34, 46
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def _tx(xs):
        return [math.log10(max(x, 1e-30)) if logx else x for x in xs]

    def _ty(ys):
        return [math.log10(max(y, 1e-30)) if logy else y for y in ys]

    allx = [v for _, xs, _ in series for v in _tx(xs)]
    ally = [v for _, _, ys in series for v in _ty(ys)]
    if not allx or not ally:
        raise ValueError("no data points")
    xmin, xmax = min(allx), max(allx)
    ymin, ymax = min(ally), max(ally)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0

    def px(x):
        return pad_l + (x - xmin) / xr * plot_w

    def py(y):
        return pad_t + plot_h - (y - ymin) / yr * plot_h

    s = [_svg_header(width, height)]
    s.append(f'  <text x="{width/2:.0f}" y="18" text-anchor="middle" '
             f'font-weight="600">{_esc(title)}</text>\n')
    s.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
             f'y2="{pad_t+plot_h}"/>\n')
    s.append(f'  <line class="axis" x1="{pad_l}" y1="{pad_t+plot_h}" '
             f'x2="{pad_l+plot_w}" y2="{pad_t+plot_h}"/>\n')
    s.append(f'  <text x="{pad_l+plot_w/2:.0f}" y="{height-8}" '
             f'text-anchor="middle">{_esc(xlabel)}{" (log10)" if logx else ""}</text>\n')
    s.append(f'  <text x="16" y="{pad_t+plot_h/2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {pad_t+plot_h/2:.0f})">'
             f'{_esc(ylabel)}{" (log10)" if logy else ""}</text>\n')
    for i, (label, xs, ys) in enumerate(series):
        col = palette[i % len(palette)]
        tx, ty = _tx(xs), _ty(ys)
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(tx, ty))
        s.append(f'  <polyline class="curve" stroke="{col}" points="{pts}"/>\n')
        for x, y in zip(tx, ty):
            s.append(f'  <circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.4" '
                     f'fill="{col}"/>\n')
        ly = pad_t + 14 + i * 16
        s.append(f'  <line x1="{pad_l+plot_w+12}" y1="{ly}" '
                 f'x2="{pad_l+plot_w+30}" y2="{ly}" stroke="{col}" '
                 f'stroke-width="2"/>\n')
        s.append(f'  <text x="{pad_l+plot_w+34}" y="{ly+3}" '
                 f'font-size="10">{_esc(label)}</text>\n')
    s.append('</svg>\n')
    return "".join(s)


def _esc(t: str) -> str:
    return (str(t).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
