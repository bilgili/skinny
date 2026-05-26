#!/usr/bin/env python3
"""Render a single offscreen image. Thin wrapper over skinny.headless."""
from __future__ import annotations
import argparse
from skinny.headless import render_scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="render.png")
    ap.add_argument("--scene", default="assets/cornell_box_sphere.usda")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--integrator", choices=["path", "bdpt"], default="path")
    ap.add_argument("--no-direct", action="store_true")
    args = ap.parse_args()
    render_scene(
        args.scene, args.output, width=args.width, height=args.height,
        samples=args.samples, integrator=args.integrator,
        direct_light=not args.no_direct,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
