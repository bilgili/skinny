"""Capture the per-frame animated re-read as a fixture (task 1.3).

Drives a real headless renderer over a time-code sweep and records what the
per-frame extraction produces: instance world transforms, distant and sphere
lights, and the camera override. The recorded values are the identity target
for `scene_intake.read_at_time`.

Run from the repo root with the guarded Metal runner:

    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 \
        tests/fixtures/_capture_anim_reread.py

Re-capture it; never hand-edit `anim_reread.json` to match the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "anim_reread.json"

# Z-up on purpose: the up-axis rotation `rt` is where the per-frame composition
# math lives (light directions and camera basis are rotated, instance
# transforms are post-multiplied by rt4). A Y-up stage would leave rt None and
# test nothing.
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
        0: (-4, 0, 1),
        24: (4, 0, 3),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double radius = 1.5
}

def Sphere "Static"
{
    double3 xformOp:translate = (0, 6, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    double radius = 0.75
}

def Camera "Cam"
{
    float focalLength = 35
    float verticalAperture = 24
    double3 xformOp:translate.timeSamples = {
        0: (0, -14, 2),
        24: (3, -12, 5),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def DistantLight "Sun"
{
    float inputs:intensity.timeSamples = {
        0: 3,
        24: 7,
    }
    color3f inputs:color = (1, 0.9, 0.8)
    double3 xformOp:rotateXYZ.timeSamples = {
        0: (-45, 20, 0),
        24: (-70, 60, 10),
    }
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}

def SphereLight "Bulb"
{
    float inputs:intensity.timeSamples = {
        0: 400,
        24: 900,
    }
    float inputs:radius = 0.4
    color3f inputs:color = (0.6, 0.8, 1)
    double3 xformOp:translate.timeSamples = {
        0: (2, 2, 3),
        24: (-2, 3, 6),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"""

TIME_CODES = [0.0, 6.0, 12.0, 18.0, 24.0]
W = H = 64


def _vec(a) -> list:
    return np.asarray(a, dtype=np.float64).round(6).tolist()


def _snapshot(r) -> dict:
    scene = r._usd_scene
    ov = r._usd_camera_override
    return {
        "instances": {
            inst.name: _vec(inst.transform) for inst in scene.instances
        },
        "lights_dir": [
            {
                "prim_path": lt.prim_path,
                "direction": _vec(lt.direction),
                "radiance": _vec(lt.radiance),
            }
            for lt in scene.lights_dir
        ],
        "lights_sphere": [
            {
                "prim_path": lt.prim_path,
                "position": _vec(lt.position),
                "radiance": _vec(lt.radiance),
                "radius": round(float(lt.radius), 6),
            }
            for lt in scene.lights_sphere
        ],
        "camera": None if ov is None else {
            "position": _vec(ov.position),
            "forward": _vec(ov.forward),
            "up": _vec(ov.up),
            "focal_length_mm": round(float(ov.focal_length_mm), 6),
            "vertical_aperture_mm": round(float(ov.vertical_aperture_mm), 6),
        },
    }


def main() -> None:
    import tempfile

    from skinny.metal_context import MetalContext
    from skinny.renderer import Renderer

    tmp = Path(tempfile.mkdtemp(prefix="skinny-anim-fixture-"))
    scene_path = tmp / "anim_z_up.usda"
    scene_path.write_text(ANIM_USDA)

    ctx = MetalContext(window=None, width=W, height=H)
    r = None
    try:
        r = Renderer(
            vk_ctx=ctx,
            shader_dir=PROJECT_ROOT / "src" / "skinny" / "shaders",
            hdr_dir=PROJECT_ROOT / "hdrs",
            tattoo_dir=PROJECT_ROOT / "tattoos",
            usd_scene_path=scene_path,
        )
        deadline = 400
        while deadline > 0 and (
            r._usd_scene is None or len(r._usd_scene.instances) < 2
        ):
            r.update(0.025)
            deadline -= 1
        if r._usd_scene is None or len(r._usd_scene.instances) < 2:
            raise SystemExit("USD stream never completed")
        if not r.clock.has_animation:
            raise SystemExit("stage reported no animation")

        r.clock.playing = False
        frames = {}
        for tc in TIME_CODES:
            r.clock.current_time_code = float(tc)
            r.update(0.0)
            frames[f"{tc:g}"] = _snapshot(r)

        OUT.write_text(json.dumps({
            "stage": ANIM_USDA,
            "time_codes": TIME_CODES,
            "frames": frames,
        }, indent=2) + "\n")
        print(f"wrote {OUT} ({len(frames)} time codes)")
    finally:
        if r is not None:
            r.cleanup()
        ctx.destroy()


if __name__ == "__main__":
    main()
