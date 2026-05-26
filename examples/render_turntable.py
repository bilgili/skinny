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

    pad = max(4, len(str(args.frames - 1)))

    with HeadlessRenderer(args.width, args.height) as r:
        if scene_cam is not None:
            # Rotate the existing camera around Y by reading its base transform
            # once, then each frame clearing the op stack and writing a single
            # clean transform op.  This is robust regardless of how the camera
            # was originally authored (TypeTransform, separate translate/rotateY
            # ops, DCC exports, etc.).
            xf = UsdGeom.Xformable(scene_cam)
            base_mat = xf.GetLocalTransformation(Usd.TimeCode.Default())
            for i in range(args.frames):
                angle_deg = 360.0 * i / args.frames
                ry = Gf.Matrix4d().SetRotate(
                    Gf.Rotation(Gf.Vec3d(0, 1, 0), angle_deg)
                )
                xf.ClearXformOpOrder()
                op = xf.AddTransformOp()
                op.Set(base_mat * ry)
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
