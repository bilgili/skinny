"""Probe: link (NOT pipeline-compile) the Metal wavefront path program with
SKINNY_METAL_NEURAL=1 and dump every global parameter's category + buffer-slot
index per entry — the slot-cap planning map for phase 6. Linking alone does not
spike MTLCompilerService, but run under guarded_metal.sh anyway.

    PYTHONPATH=$PWD/src TIMEOUT_S=300 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/probe_metal_neural_layout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"

ENTRIES = ["wfPathGenerate", "wfPathIntersect", "wfBuildArgs", "wfScatter",
           "wfPathShadeFlat", "wfPathResolve", "wfPathShade"]


def main() -> int:
    from skinny.megakernel_sources import emit_megakernel_sources
    from skinny.metal_context import MetalContext
    from skinny.metal_wavefront import _metal_slang_session

    emit_megakernel_sources(SHADER_DIR, [])
    ctx = MetalContext(window=None, width=64, height=64)
    session = _metal_slang_session(ctx, SHADER_DIR,
                                   {"SKINNY_METAL_NEURAL": "1"})
    src = SHADER_DIR / "wavefront" / "wavefront_path.slang"
    module = session.load_module_from_source(
        "wavefront_path", src.read_text(encoding="utf-8"), str(src))

    first = True
    for entry in ENTRIES:
        prog = session.link_program([module], [module.entry_point(entry)])
        rows = []
        for p in prog.layout.parameters:
            if first:
                print("[attrs]", [a for a in dir(p) if not a.startswith("_")])
                first = False
            tl = p.type_layout
            kind = str(getattr(tl, "kind", "?"))
            idx = getattr(p, "binding_index", None)
            space = getattr(p, "binding_space", None)
            off = getattr(tl, "size", None)
            rows.append((p.name, kind, idx, space, off))
        print(f"== {entry}: {len(rows)} globals")
        for name, kind, idx, space, off in rows:
            print(f"   {name:32s} {kind:28s} idx={idx} space={space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
