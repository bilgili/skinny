#!/usr/bin/env python3
"""Orbit-camera turntable. Mutates a camera xform on a Usd.Stage per frame."""
from __future__ import annotations
import argparse
import math
from pathlib import Path

from pxr import Usd, UsdGeom, Gf
from PIL import Image
from skinny.headless import HeadlessRenderer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="assets/cornell_box_sphere.usda")
    ap.add_argument("--outdir", default="turntable")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(args.scene)

    # Find the scene's first authored camera (or None).
    scene_cam = None
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Camera):
            scene_cam = prim
            break

    pad = len(str(args.frames - 1))

    with HeadlessRenderer(args.width, args.height) as r:
        if scene_cam is not None:
            # Rotate the existing camera around Y by reading its base transform
            # and multiplying by a Y-rotation for each frame.
            xf = UsdGeom.Xformable(scene_cam)
            ops = xf.GetOrderedXformOps()
            if ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
                base_mat = ops[0].Get()
            else:
                base_mat = Gf.Matrix4d(1.0)
            # Pivot = scene origin
            pivot = Gf.Vec3d(0.0, 0.0, 0.0)
            # Build the rotate-around-Y helper
            rot_op = ops[0] if (ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform) \
                else xf.AddTransformOp()
            for i in range(args.frames):
                angle_deg = 360.0 * i / args.frames
                angle_rad = math.radians(angle_deg)
                c, s = math.cos(angle_rad), math.sin(angle_rad)
                ry = Gf.Matrix4d(
                    c,  0.0, -s,  0.0,
                    0.0, 1.0,  0.0, 0.0,
                    s,  0.0,  c,  0.0,
                    0.0, 0.0,  0.0, 1.0,
                )
                # Translate to pivot, rotate, translate back, then apply base.
                t_fwd = Gf.Matrix4d(1.0)
                t_fwd.SetTranslateOnly(-pivot)
                t_bck = Gf.Matrix4d(1.0)
                t_bck.SetTranslateOnly(pivot)
                new_mat = base_mat * t_fwd * ry * t_bck
                rot_op.Set(new_mat)
                arr = r.render_to_array(stage, samples=args.samples)
                out = outdir / f"frame_{i:0{pad}d}.png"
                Image.fromarray(arr, "RGBA").save(out)
                print(f"  {out} ({i + 1}/{args.frames})")
        else:
            # No authored camera — orbit via the renderer's built-in orbit camera.
            for i in range(args.frames):
                r.renderer.orbit_camera.yaw = 2.0 * math.pi * i / args.frames
                arr = r.render_to_array(stage, samples=args.samples)
                out = outdir / f"frame_{i:0{pad}d}.png"
                Image.fromarray(arr, "RGBA").save(out)
                print(f"  {out} ({i + 1}/{args.frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
