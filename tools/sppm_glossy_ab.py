"""SPPM glossy-continuation A/B on three_materials_demo (change sppm-glossy-final-gather).

Renders the marble/wood/brass demo three ways on one reused HeadlessRenderer:
  - path reference (integrator=path, wavefront)
  - SPPM PM-1   (integrator=sppm, glossy threshold = 0  -> delta-only, brass washes out)
  - SPPM glossy (integrator=sppm, glossy threshold = default ~0.5 -> brass reflects)

Reads LINEAR accumulation (read_accumulation_hdr, mean already), measures the
brass / wood / marble screen-space patches, tracks brass convergence-to-path over
passes, and writes a labelled side-by-side PNG at a shared tonemap/exposure.

Usage:
  PYTHONPATH=$PWD/src ./bin/python3.13 tools/sppm_glossy_ab.py \
      --backend metal --sppm-passes 128 --path-passes 256 --out /tmp/sppm_glossy
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEMO = REPO / "assets" / "three_materials_demo.usda"

# Screen-space sphere centroids at 256x256 (from tests/test_headless.py).
CENTERS = {"marble": (60, 140), "wood": (128, 140), "brass": (196, 140)}
HALF = 16  # 33x33 patch


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def patch(img: np.ndarray, name: str, w: int, h: int) -> np.ndarray:
    cx, cy = CENTERS[name]
    sx, sy = w / 256.0, h / 256.0
    cx, cy = int(round(cx * sx)), int(round(cy * sy))
    hh = max(1, int(round(HALF * (w / 256.0))))
    x0, x1 = max(0, cx - hh), min(w, cx + hh + 1)
    y0, y1 = max(0, cy - hh), min(h, cy + hh + 1)
    return img[y0:y1, x0:x1, :3]


def patch_stats(img: np.ndarray, name: str, w: int, h: int) -> dict:
    p = patch(img, name, w, h)
    lum = luminance(p)
    return {"mean_lum": float(lum.mean()), "std_lum": float(lum.std()),
            "mean_rgb": [float(c) for c in p.reshape(-1, 3).mean(0)]}


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Exposure-aligned relative L2 of two patches (a vs reference b)."""
    a, b = a.reshape(-1, 3).astype(np.float64), b.reshape(-1, 3).astype(np.float64)
    s = float((a * b).sum() / max((a * a).sum(), 1e-12))  # least-squares scale a->b
    d = s * a - b
    return float(np.sqrt((d * d).sum() / max((b * b).sum(), 1e-12)))


def aces(x: np.ndarray) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def tonemap(img: np.ndarray, exposure: float) -> np.ndarray:
    srgb = aces(img[..., :3] * (2.0 ** exposure))
    srgb = np.where(srgb <= 0.0031308, srgb * 12.92,
                    1.055 * np.power(np.clip(srgb, 1e-6, None), 1 / 2.4) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--sppm-passes", type=int, default=128)
    ap.add_argument("--path-passes", type=int, default=256)
    ap.add_argument("--out", default="/tmp/sppm_glossy")
    args = ap.parse_args()

    from skinny.headless import HeadlessRenderer, RenderOptions, _load_scene

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, H = args.w, args.h

    hr = HeadlessRenderer(W, H, backend=args.backend, execution_mode="wavefront")
    r = hr.renderer
    try:
        # Load scene + warm up async USD mesh streaming until the 3 spheres exist.
        r.set_usd_scene(_load_scene(str(DEMO), None))
        for _ in range(400):
            if r._usd_scene is not None and len(r._usd_scene.instances) >= 3:
                break
            r.update(0.02)
        n_inst = 0 if r._usd_scene is None else len(r._usd_scene.instances)
        print(f"[warmup] instances={n_inst}", flush=True)
        assert n_inst >= 3, f"demo did not stream 3 spheres (got {n_inst})"

        def configure(integrator_index: int, glossy):
            r.integrator_index = integrator_index
            if integrator_index == 2:
                if glossy is None:
                    if hasattr(r, "_sppm_glossy_roughness_override"):
                        delattr(r, "_sppm_glossy_roughness_override")
                else:
                    r._sppm_glossy_roughness_override = float(glossy)
            r._last_state_hash = None  # force accumulation reset for a clean A/B
            r.accum_frame = 0

        def accumulate(passes: int) -> np.ndarray:
            for _ in range(passes):
                r.update(1.0 / 60.0)
                r.render_headless()
            arr, _ = r.read_accumulation_hdr()
            return np.asarray(arr, dtype=np.float64)[..., :3]

        # ── Path reference ──
        configure(0, None)
        print(f"[path] accumulating {args.path_passes} spp ...", flush=True)
        img_path = accumulate(args.path_passes)

        # ── SPPM PM-1 (threshold 0) ──
        configure(2, 0.0)
        print(f"[sppm thr=0] accumulating {args.sppm_passes} passes ...", flush=True)
        img_pm1 = accumulate(args.sppm_passes)

        # ── SPPM glossy (default ~0.5) with brass convergence checkpoints ──
        configure(2, None)
        checkpoints = sorted({8, 24, 64, args.sppm_passes})
        conv = []
        done = 0
        brass_ref = patch(img_path, "brass", W, H)
        print(f"[sppm default] accumulating {args.sppm_passes} passes ...", flush=True)
        for cp in checkpoints:
            for _ in range(cp - done):
                r.update(1.0 / 60.0)
                r.render_headless()
            done = cp
            arr, _ = r.read_accumulation_hdr()
            img_cp = np.asarray(arr, dtype=np.float64)[..., :3]
            e = rel_l2(patch(img_cp, "brass", W, H), brass_ref)
            conv.append({"passes": cp, "brass_relL2_to_path": e})
            print(f"  pass {cp:4d}: brass relL2->path = {e:.4f}", flush=True)
        img_def = img_cp

        # ── Metrics ──
        report = {"backend": args.backend, "w": W, "h": H,
                  "path_passes": args.path_passes, "sppm_passes": args.sppm_passes,
                  "glossy_default": float(getattr(__import__("skinny.renderer",
                      fromlist=["_SPPM_GLOSSY_ROUGHNESS_DEFAULT"]),
                      "_SPPM_GLOSSY_ROUGHNESS_DEFAULT")),
                  "convergence": conv, "patches": {}}
        for name in CENTERS:
            ref = patch(img_path, name, W, H)
            report["patches"][name] = {
                "path": patch_stats(img_path, name, W, H),
                "sppm_thr0": patch_stats(img_pm1, name, W, H),
                "sppm_default": patch_stats(img_def, name, W, H),
                "relL2_thr0_to_path": rel_l2(patch(img_pm1, name, W, H), ref),
                "relL2_default_to_path": rel_l2(patch(img_def, name, W, H), ref),
            }
        np.save(out / "img_path.npy", img_path)
        np.save(out / "img_sppm_thr0.npy", img_pm1)
        np.save(out / "img_sppm_default.npy", img_def)
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
        print("[metrics]\n" + json.dumps(report, indent=2), flush=True)

        # ── Labelled side-by-side at a SHARED tonemap/exposure ──
        try:
            from PIL import Image, ImageDraw
        except Exception:
            print("PIL missing — skipping composite", flush=True)
            return
        med = float(np.median(luminance(img_path)[luminance(img_path) > 1e-4])) \
            if np.any(luminance(img_path) > 1e-4) else 0.18
        exposure = float(np.log2(0.18 / max(med, 1e-4)))  # map scene median -> mid-grey
        panels = [("path reference", img_path),
                  ("SPPM thr=0 (PM-1)", img_pm1),
                  ("SPPM default (glossy)", img_def)]
        pad, lab = 8, 22
        tiles = []
        for title, im in panels:
            full = tonemap(im, exposure)
            crop = tonemap(patch(im, "brass", W, H), exposure)
            crop_img = Image.fromarray(crop).resize((W, W), Image.NEAREST)
            tiles.append((title, Image.fromarray(full), crop_img))
        cw = W
        canvas = Image.new("RGB", (3 * cw + 4 * pad, H + W + 3 * lab + 2 * pad),
                           (24, 24, 24))
        d = ImageDraw.Draw(canvas)
        for i, (title, full, crop) in enumerate(tiles):
            x = pad + i * (cw + pad)
            d.text((x, 2), title, fill=(240, 240, 240))
            canvas.paste(full, (x, lab))
            d.text((x, lab + H + 4), "brass patch (nearest-zoom)", fill=(200, 200, 120))
            canvas.paste(crop, (x, lab + H + lab))
        comp = out / "sppm_glossy_ab.png"
        canvas.save(comp)
        print(f"[composite] {comp}", flush=True)
    finally:
        hr.cleanup()


if __name__ == "__main__":
    main()
