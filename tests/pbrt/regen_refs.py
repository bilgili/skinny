#!/usr/bin/env python3
"""Offline reference-EXR generator for the parity corpus (NOT a test).

Renders pbrt v4 reference images at the corpus resolution and drops them into
``tests/pbrt/corpus/refs/``. The parity gate consumes only the checked-in EXRs;
this script is the deliberate, occasional step that produces them (re-run when a
scene or the pinned pbrt version changes).

Heavy scenes (`bathroom`, `dragon`) point at the external pbrt-v4-scenes trees;
the synthetic corpus scenes regenerate from their in-repo ``.pbrt`` sources. A
temp copy of the scene with the Film block patched (resolution + output name) is
written *beside the source* so relative geometry/texture includes still resolve.

Usage:
    python tests/pbrt/regen_refs.py --scene bathroom --res 256 --spp 256
    python tests/pbrt/regen_refs.py --scene dragon   --res 256 --spp 256
    python tests/pbrt/regen_refs.py --scene all       # synthetic corpus refs

pbrt binary defaults to ``~/projects/pbrt-v4/build/pbrt`` (override --pbrt).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
REFS = os.path.join(CORPUS, "refs")
# Confirming-scene suite lives under tests/assets/suite/<scene>/<scene>.pbrt
# (change confirming-test-scenes); its refs share the corpus refs/ dir under a
# ``suite_`` prefix so the single manifest/gate consumes them uniformly.
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUITE_ROOT = os.path.join(REPO_ROOT, "tests", "assets", "suite")
HOME = os.path.expanduser("~")
DEFAULT_PBRT = os.path.join(HOME, "projects", "pbrt-v4", "build", "pbrt")
SCENES_ROOT = os.path.join(HOME, "projects", "pbrt-v4-scenes")

# corpus ref name → external pbrt scene source (heavy scenes only; the synthetic
# corpus scenes regenerate from corpus/<name>.pbrt).
HEAVY_SOURCES = {
    "bathroom": os.path.join(SCENES_ROOT, "contemporary-bathroom", "contemporary-bathroom.pbrt"),
    # sssdragon variant matching assets/dragon_sss.usda; pick the variant the
    # USD was imported from if a follow-up needs an exact match.
    "dragon": os.path.join(SCENES_ROOT, "sssdragon", "dragon_const.pbrt"),
    # pbrt procedural cloud medium (pbrt-cloud-procedural-medium): analytic fBm
    # density, MEDIUM_CLOUD; usda regenerates locally like the nanovdb clouds.
    "clouds": os.path.join(SCENES_ROOT, "clouds", "clouds.pbrt"),
}

# xresolution/yresolution/pixelsamples each appear ONLY in the Film/Sampler
# block; the output name is overridden via pbrt's --outfile flag (NOT a regex,
# because "string filename" also names every plymesh geometry file).
_RES_X = re.compile(r'("integer xresolution"\s*)\[?\s*\d+\s*\]?')
_RES_Y = re.compile(r'("integer yresolution"\s*)\[?\s*\d+\s*\]?')
_SPP = re.compile(r'("integer pixelsamples"\s*)\[?\s*\d+\s*\]?')
# The whole Integrator statement: the `Integrator "type" ...` line plus any
# following indented parameter continuation lines (pbrt params often wrap). Used
# only when an integrator override is requested (e.g. rendering a heavy scene's
# reference with pbrt's `path` integrator to match skinny's path-tracer anchor
# instead of the scene's authored `sppm`).
_INTEGRATOR = re.compile(r'Integrator\b[^\n]*(?:\n[ \t]+[^\n]*)*')


def patch_scene(text: str, width: int, height: int, spp: int | None,
                integrator: str | None = None) -> str:
    text = _RES_X.sub(rf'\1[ {width} ]', text)
    text = _RES_Y.sub(rf'\1[ {height} ]', text)
    if spp is not None:
        text = _SPP.sub(rf'\1[ {spp} ]', text)
    if integrator is not None:
        # Replace the authored integrator with the override (maxdepth 5 mirrors the
        # corpus scenes' bounce budget). The Film `maxcomponentvalue` clamp is left
        # intact so reference and skinny render under the same firefly suppression.
        text = _INTEGRATOR.sub(f'Integrator "{integrator}" "integer maxdepth" 5',
                               text, count=1)
    return text


def render_ref(name: str, src_pbrt: str, width: int, height: int,
               spp: int | None, pbrt: str, integrator: str | None = None) -> str:
    if not os.path.isfile(pbrt):
        raise SystemExit(f"pbrt binary not found: {pbrt}")
    if not os.path.isfile(src_pbrt):
        raise SystemExit(f"scene not found: {src_pbrt}")
    os.makedirs(REFS, exist_ok=True)
    scene_dir = os.path.dirname(src_pbrt)
    dst = os.path.join(REFS, f"{name}.exr")
    patched = patch_scene(open(src_pbrt).read(), width, height, spp, integrator)
    # temp scene beside the source so relative includes resolve
    fd, tmp = tempfile.mkstemp(suffix=".pbrt", prefix=f"_regen_{name}_", dir=scene_dir)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(patched)
        # --outfile overrides the Film filename without touching geometry refs.
        subprocess.run([pbrt, "--outfile", dst, os.path.basename(tmp)],
                       cwd=scene_dir, check=True)
        print(f"wrote {dst}  ({width}x{height}, spp={spp or 'scene'}"
              f"{', integrator=' + integrator if integrator else ''})")
        return dst
    finally:
        os.remove(tmp)


def _suite_pbrt_scenes() -> list[tuple[str, str]]:
    """Discover suite pbrt counterparts as (ref_name, src_pbrt) pairs.

    ``tests/assets/suite/<scene>/<scene>.pbrt`` → ref ``suite_<scene>.exr``. A
    scene folder with no ``.pbrt`` (MaterialX-only / furnace scene) is skipped
    silently — the manifest records its ``pbrt_skip`` reason.
    """
    out: list[tuple[str, str]] = []
    if not os.path.isdir(SUITE_ROOT):
        return out
    for scene in sorted(os.listdir(SUITE_ROOT)):
        # Furnace scenes are gated against the analytic value 1.0 (rendered under
        # skinny's furnace mode), not a pbrt reference — pbrt has no furnace mode,
        # so a pbrt render of them would be meaningless. Skip.
        if scene.startswith("furnace_"):
            continue
        src = os.path.join(SUITE_ROOT, scene, f"{scene}.pbrt")
        if os.path.isfile(src):
            out.append((f"suite_{scene}", src))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True,
                    help="bathroom | dragon | <corpus-name> | all | suite | "
                         "suite:<scene>")
    ap.add_argument("--res", type=int, default=256, help="square resolution (default 256)")
    ap.add_argument("--spp", type=int, default=None, help="override pixelsamples")
    ap.add_argument("--integrator", default=None,
                    help="override the scene's Integrator (e.g. 'path' to render a "
                         "heavy scene's reference with the path tracer matching "
                         "skinny's parity anchor instead of the authored sppm)")
    ap.add_argument("--pbrt", default=DEFAULT_PBRT)
    ns = ap.parse_args()

    if ns.scene == "suite":
        pairs = _suite_pbrt_scenes()
        if not pairs:
            raise SystemExit(f"no suite .pbrt scenes under {SUITE_ROOT}")
        for name, src in pairs:
            render_ref(name, src, ns.res, ns.res, ns.spp, ns.pbrt, ns.integrator)
    elif ns.scene.startswith("suite:"):
        scene = ns.scene.split(":", 1)[1]
        src = os.path.join(SUITE_ROOT, scene, f"{scene}.pbrt")
        render_ref(f"suite_{scene}", src, ns.res, ns.res, ns.spp, ns.pbrt, ns.integrator)
    elif ns.scene in HEAVY_SOURCES:
        render_ref(ns.scene, HEAVY_SOURCES[ns.scene], ns.res, ns.res, ns.spp,
                   ns.pbrt, ns.integrator)
    elif ns.scene == "all":
        for f in sorted(os.listdir(CORPUS)):
            if f.endswith(".pbrt"):
                name = f[:-5]
                render_ref(name, os.path.join(CORPUS, f), ns.res, ns.res, ns.spp, ns.pbrt)
    else:
        src = os.path.join(CORPUS, f"{ns.scene}.pbrt")
        render_ref(ns.scene, src, ns.res, ns.res, ns.spp, ns.pbrt)


if __name__ == "__main__":
    main()
