from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import compile_module
from .verification import VerificationOptions, verify_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slangpile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("module")
    build.add_argument("-o", "--out-dir", default="generated")
    build.add_argument("--verify", action="store_true")
    build.add_argument("--slangc")

    check = subparsers.add_parser("check")
    check.add_argument("module")
    check.add_argument("--slangc")

    verify = subparsers.add_parser("verify")
    verify.add_argument("path")
    verify.add_argument("--slangc")
    verify.add_argument("-I", "--include-dir", action="append", default=[])
    verify.add_argument("--entry", dest="entry_point")
    verify.add_argument("--profile")
    verify.add_argument("--target")

    args = parser.parse_args(argv)

    if args.command == "build":
        compiled = compile_module(args.module)
        path = compiled.write(args.out_dir)
        print(path)
        if args.verify:
            result = compiled.verify(args.out_dir, slangc=args.slangc)
            _print_verification(result)
            return 0 if result.ok else 1
        return 0

    if args.command == "check":
        compile_module(args.module)
        return 0

    if args.command == "verify":
        result = verify_file(
            args.path,
            VerificationOptions(
                slangc=args.slangc,
                include_dirs=[Path(p) for p in args.include_dir],
                entry_point=args.entry_point,
                profile=args.profile,
                target=args.target,
            ),
        )
        _print_verification(result)
        return 0 if result.ok else 1

    return 2


def _print_verification(result) -> None:
    for diagnostic in result.diagnostics:
        print(diagnostic.format(), file=sys.stderr)
    if result.stderr and not result.diagnostics:
        print(result.stderr, file=sys.stderr)
    if result.stdout:
        print(result.stdout)


if __name__ == "__main__":
    raise SystemExit(main())

