"""A/B render gate for `scene-intake-interface` (task 6.2).

Renders every scene-adoption path this change touched and writes the linear
accumulation buffers to a directory. Run it on `main` and on the branch and
compare the two directories: the images must be identical, because the change
is a refactor of *who* produces a scene, not of what a scene looks like.

    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 \
        tests/fixtures/_ab_scene_intake.py <out-dir>

Paths covered:

- `pbrt_import`   — the synchronous adoption path (`set_usd_scene`)
- `stream_*`      — the async streaming path, animated Z-up, two time codes
- `skel_*`        — the CPU skeletal path, two time codes
- `empty_*`       — `create_empty_scene` and the post-edit resync path
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUITE = PROJECT_ROOT / "tests" / "assets" / "suite"
W = H = 96

# Z-up, animated transform + light + camera: the streaming path plus the
# per-frame re-read, with the up-axis rotation live.
#
# Not the `anim_reread.json` stage — that one exists to pin extracted *values*,
# so its camera stares at the floor and every render of it comes out black.
# Identical black images are a vacuous A/B pass, so this stage authors a camera
# that looks at the geometry (rotateX 90° turns the camera's local -Z toward
# world +Y under Z-up) and a dome so misses are not black either.
ANIM_USDA = """#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 24
    timeCodesPerSecond = 24
    upAxis = "Z"
    metersPerUnit = 1
)

def Sphere "Ball"
{
    double3 xformOp:translate.timeSamples = {
        0: (-3, 0, 0),
        24: (3, 0, 2),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double radius = 1.2
}

def Sphere "Static"
{
    double3 xformOp:translate = (0, 4, -1)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double radius = 1
}

def Camera "Cam"
{
    float focalLength = 35
    float verticalAperture = 24
    double3 xformOp:translate = (0, -16, 3)
    double3 xformOp:rotateXYZ = (90, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}

def DistantLight "Sun"
{
    float inputs:intensity = 4
    color3f inputs:color = (1, 0.9, 0.8)
    double3 xformOp:rotateXYZ.timeSamples = {
        0: (-45, 20, 0),
        24: (-70, 60, 10),
    }
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}

def SphereLight "Bulb"
{
    float inputs:intensity = 400
    float inputs:radius = 0.4
    color3f inputs:color = (0.6, 0.8, 1)
    double3 xformOp:translate.timeSamples = {
        0: (2, 1, 3),
        24: (-2, 2, 5),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def DomeLight "Sky"
{
    float inputs:intensity = 0.4
    color3f inputs:color = (0.5, 0.6, 0.9)
}
"""

# A minimal UsdSkel stage. The repo's skeletal tests all gate on an
# ElephantWithMonochord asset that is not checked in, so without this the CPU
# skinning path — which this change rewrote — would have no GPU coverage.
SKEL_USDA = """#usda 1.0
(
    startTimeCode = 0
    endTimeCode = 20
    timeCodesPerSecond = 24
    upAxis = "Y"
    metersPerUnit = 1
)

def SkelRoot "Root"
{
    def Skeleton "Skel" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {
        rel skel:animationSource = </Root/Skel/Anim>
        uniform matrix4d[] bindTransforms = [
            ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ),
            ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,2,0,1) )
        ]
        uniform token[] joints = ["A", "A/B"]
        uniform matrix4d[] restTransforms = [
            ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ),
            ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,2,0,1) )
        ]

        def SkelAnimation "Anim"
        {
            uniform token[] joints = ["A", "A/B"]
            quatf[] rotations.timeSamples = {
                0:  [(1,0,0,0), (1,0,0,0)],
                20: [(1,0,0,0), (0.7071,0,0,0.7071)],
            }
            half3[] scales = [(1,1,1), (1,1,1)]
            float3[] translations = [(0,0,0), (0,2,0)]
        }
    }

    def Mesh "Body" (
        prepend apiSchemas = ["SkelBindingAPI"]
    )
    {
        rel skel:skeleton = </Root/Skel>
        uniform token[] skel:joints = ["A", "A/B"]
        int[] primvars:skel:jointIndices = [
            0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
            1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1
        ] (
            elementSize = 4
            interpolation = "vertex"
        )
        float[] primvars:skel:jointWeights = [
            1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0,
            1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0
        ] (
            elementSize = 4
            interpolation = "vertex"
        )
        point3f[] points = [
            (-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0),
            (-1, 2, 0), (1, 2, 0), (1, 4, 0), (-1, 4, 0)
        ]
        int[] faceVertexCounts = [3, 3, 3, 3]
        int[] faceVertexIndices = [0,1,2, 0,2,3, 4,5,6, 4,6,7]
    }
}

def Camera "Cam"
{
    float focalLength = 35
    float verticalAperture = 24
    double3 xformOp:translate = (0, 2, 12)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def DistantLight "Sun"
{
    float inputs:intensity = 4
    double3 xformOp:rotateXYZ = (-30, 25, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}
"""


def _accum(r) -> np.ndarray:
    """Linear HDR accumulation, not the tonemapped display pixels."""
    r.render_headless()
    return np.asarray(r.read_accumulation(), dtype=np.float32)


def _settle(r, frames: int = 4) -> None:
    for _ in range(frames):
        r.update(0.0)


def _pump_until_streamed(r, want: int, budget: int = 400) -> None:
    while budget > 0 and (
        r._usd_scene is None or len(r._usd_scene.instances) < want
    ):
        r.update(0.025)
        budget -= 1


def main(out_dir: Path) -> None:
    from skinny.metal_context import MetalContext
    from skinny.renderer import Renderer
    from skinny.usd_loader import load_scene_from_usd

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="skinny-ab-intake-"))
    anim_path = tmp / "anim.usda"
    anim_path.write_text(ANIM_USDA)
    skel_path = tmp / "skel.usda"
    skel_path.write_text(SKEL_USDA)

    shots: dict[str, np.ndarray] = {}
    common = dict(
        shader_dir=PROJECT_ROOT / "src" / "skinny" / "shaders",
        hdr_dir=PROJECT_ROOT / "hdrs",
        tattoo_dir=PROJECT_ROOT / "tattoos",
    )

    # ── 1. Synchronous adoption: a pbrt-imported scene via set_usd_scene ──
    ctx = MetalContext(window=None, width=W, height=H)
    r = None
    try:
        r = Renderer(vk_ctx=ctx, **common)
        scene = load_scene_from_usd(SUITE / "mat_emissive" / "mat_emissive.usda")
        r.set_usd_scene(scene)
        _settle(r, 8)
        shots["pbrt_import"] = _accum(r)
    finally:
        if r is not None:
            r.cleanup()
        ctx.destroy()

    # ── 2. Streaming adoption + per-frame re-read on a Z-up animated stage ──
    ctx = MetalContext(window=None, width=W, height=H)
    r = None
    try:
        r = Renderer(vk_ctx=ctx, usd_scene_path=anim_path, **common)
        _pump_until_streamed(r, want=2)
        r.clock.playing = False
        for tc in (0.0, 12.0, 24.0):
            r.clock.current_time_code = tc
            _settle(r, 6)
            shots[f"stream_{tc:g}"] = _accum(r)
    finally:
        if r is not None:
            r.cleanup()
        ctx.destroy()

    # ── 3. Skeletal (CPU LBS) at two time codes ──
    ctx = MetalContext(window=None, width=W, height=H)
    r = None
    try:
        r = Renderer(vk_ctx=ctx, usd_scene_path=skel_path, **common)
        _pump_until_streamed(r, want=1)
        print(f"[ab] skel bindings: "
              f"{None if r._skeletal is None else len(r._skeletal.meshes)}")
        r.clock.playing = False
        for tc in (0.0, 20.0):
            r.clock.current_time_code = tc
            _settle(r, 6)
            shots[f"skel_{tc:g}"] = _accum(r)
    finally:
        if r is not None:
            r.cleanup()
        ctx.destroy()

    # ── 4. Force-replace + post-edit resync ──
    ctx = MetalContext(window=None, width=W, height=H)
    r = None
    try:
        r = Renderer(vk_ctx=ctx, **common)
        r.create_empty_scene()
        _settle(r, 6)
        shots["empty_created"] = _accum(r)
        r.add_primitive("Sphere", name="Ball")
        _settle(r, 6)
        shots["empty_after_add"] = _accum(r)
        # Disable it, then edit again: the second resync must preserve the
        # runtime flag (finding #7).
        r._usd_scene.instances[0].enabled = False
        r.add_primitive("Cube", name="Box")
        _settle(r, 6)
        shots["empty_carryover"] = _accum(r)
        print("[ab] enabled flags after carry-over: "
              f"{[i.enabled for i in r._usd_scene.instances]}")
    finally:
        if r is not None:
            r.cleanup()
        ctx.destroy()

    for name, img in shots.items():
        np.save(out_dir / f"{name}.npy", img)
        print(f"[ab] {name}: shape {img.shape} mean {float(img.mean()):.6f}")
    print(f"[ab] wrote {len(shots)} images to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
