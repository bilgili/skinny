"""Command-line entry for the pbrt importer: ``skinny-import-pbrt``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skinny-import-pbrt",
        description="Convert a pbrt v4 scene to a skinny-loadable USD stage.",
    )
    parser.add_argument("scene", help="input .pbrt scene file")
    parser.add_argument("-o", "--output", required=True, help="output .usda/.usd path")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the report")
    parser.add_argument(
        "-m",
        "--materialx",
        action="store_true",
        help="export a MaterialX (.mtlx) sidecar of rich standard_surface "
        "materials alongside the .usda (the stage references it instead of "
        "authoring UsdPreviewSurface shaders)",
    )
    args = parser.parse_args(argv)

    from .api import import_pbrt

    _stage, report = import_pbrt(args.scene, out=args.output, materialx=args.materialx)
    if not args.quiet:
        print(report)
    # non-zero exit if anything was unsupported, so scripts can gate on it
    return 1 if report.has_unsupported() else 0


if __name__ == "__main__":
    sys.exit(main())
