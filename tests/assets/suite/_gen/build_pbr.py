#!/usr/bin/env python3
"""Generate the MaterialX-only "PBR material" mini-shaderball scenes
(Stage 3 of change confirming-test-scenes).

These 7 scenes each drop one OpenPBR "physically based material" reference card
onto the shared mini-shaderball geometry. Unlike the pbrt-expressible suite
scenes (``build.py``), OpenPBR standard_surface has no pbrt or UsdPreviewSurface
counterpart, so each scene ships **only** a MaterialX authoring:
``<scene>_mtlx.usda`` + ``<scene>_mtlx.mtlx``.

Pipeline per scene:

1. Import a neutral shaderball pbrt (grey sphere on grey floor, overhead area
   light) through skinny's ``import_pbrt(..., materialx=True)``. That writes the
   ``_mtlx.usda`` + ``.mtlx`` sidecar with the canonical
   ``shape_0`` (emitter) / ``shape_1`` (ground) / ``shape_2`` (sphere) triples.
2. Rewrite the sphere's ``shape_2_mat_surface`` standard_surface node in the
   ``.mtlx`` in place with the OpenPBR→standard_surface-mapped inputs for that
   material. The emitter and ground nodes are left untouched.

The OpenPBR values are extracted from the bound material of each source card in
the main checkout (gitignored, absent in worktrees):
``assets/materialxusd/tests/physically_based/<Name>_OPBR_MAT_PBM.usda``.

Run from the repo root (worktree has no venv — use the main-checkout interpreter):

    PYTHONPATH=src /Users/ahmetbilgili/projects/skinny/.venv/bin/python \
        tests/assets/suite/_gen/build_pbr.py

Idempotent: re-running overwrites the generated files in place.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import geom  # noqa: E402

SUITE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── neutral mini-shaderball ────────────────────────────────────────────────
# Same layout as build.py's single-material scenes: a downward overhead area
# light, a mid-grey floor, and a placeholder grey sphere. The sphere material is
# a throwaway — its standard_surface node is rewritten per PBR card below.
_HEADER = """LookAt 0 1.5 6  0 0.7 0  0 1 0
Camera "perspective" "float fov" 30
Sampler "independent" "integer pixelsamples" {spp}
Integrator "path" "integer maxdepth" {maxdepth}
Film "rgb" "integer xresolution" 128 "integer yresolution" 128 "string filename" "{name}.exr"
WorldBegin
"""

_AREA_LIGHT = """AttributeBegin
  AreaLightSource "diffuse" "rgb L" [12 12 12]
  {light_geom}
AttributeEnd
"""

_GROUND = """AttributeBegin
  Material "diffuse" "rgb reflectance" [0.18 0.18 0.18]
  {ground_geom}
AttributeEnd
"""


def _sphere_block(material: str) -> str:
    pts, idx = geom.uv_sphere(0.0, 0.7, 0.0, 0.7)
    return (f"AttributeBegin\n  {material}\n"
            f"  {geom.trianglemesh(pts, idx)}\nAttributeEnd\n")


def _neutral_pbrt(name: str, *, spp: int, maxdepth: int) -> str:
    return "".join([
        _HEADER.format(spp=spp, maxdepth=maxdepth, name=name),
        _AREA_LIGHT.format(light_geom=geom.trianglemesh(*geom.ceiling_light())),
        _GROUND.format(ground_geom=geom.trianglemesh(*geom.ground())),
        _sphere_block('Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]'),
    ])


# ─── PBR material table ─────────────────────────────────────────────────────
# OpenPBR→standard_surface mapped inputs (already read from the bound material of
# each source card). `source` = provenance filename for the manifest notes.
#
# Each `inputs` dict maps standard_surface input name -> (mtx_type, value_str).

def _c(r: float, g: float, b: float) -> tuple[str, str]:
    return ("color3", f"{r}, {g}, {b}")


def _f(v: float) -> tuple[str, str]:
    return ("float", f"{v}")


PBR: dict[str, dict] = {
    "mat_pbr_gold": {
        "source": "Gold_OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 8,
        "inputs": {
            "base_color": _c(1.0, 0.72, 0.315),
            "metalness": _f(1.0),
            "specular_IOR": _f(1.5),
            "specular_roughness": _f(0.02),
        },
    },
    "mat_pbr_copper": {
        "source": "Copper_OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 8,
        "inputs": {
            "base_color": _c(0.988, 0.688, 0.448),
            "metalness": _f(1.0),
            "specular_IOR": _f(1.5),
            "specular_roughness": _f(0.02),
        },
    },
    "mat_pbr_glass": {
        "source": "Glass_OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 12,
        "inputs": {
            "base_color": _c(1.0, 1.0, 1.0),
            "transmission": _f(1.0),
            "transmission_color": _c(1.0, 1.0, 1.0),
            "specular_IOR": _f(1.52),
            "specular_roughness": _f(0.0),
        },
    },
    "mat_pbr_plastic_pc": {
        "source": "Plastic__PC__OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 12,
        "inputs": {
            "base_color": _c(1.0, 1.0, 1.0),
            "transmission": _f(1.0),
            "transmission_color": _c(1.0, 1.0, 1.0),
            "specular_IOR": _f(1.585),
            "specular_roughness": _f(0.1),
        },
    },
    "mat_pbr_skin1": {
        "source": "Skin_I_OPBR_MAT_PBM.usda",
        "material_class": "subsurface",
        "spp": 256, "maxdepth": 8,
        "inputs": {
            "base_color": _c(0.847, 0.638, 0.552),
            "specular_IOR": _f(1.4),
            "specular_roughness": _f(0.5),
            "subsurface": _f(1.0),
            "subsurface_color": _c(0.847, 0.638, 0.552),
        },
    },
    "mat_pbr_graycard": {
        "source": "Gray_Card_OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 8,
        "inputs": {
            "base_color": _c(0.18, 0.18, 0.18),
            "specular_IOR": _f(1.5),
            "specular_roughness": _f(0.5),
        },
    },
    "mat_pbr_musou_black": {
        "source": "Musou_Black_OPBR_MAT_PBM.usda",
        "material_class": "flat",
        "spp": 256, "maxdepth": 8,
        "inputs": {
            "base_color": _c(0.006, 0.006, 0.006),
            "specular_IOR": _f(1.5),
            "specular_roughness": _f(0.9),
        },
    },
}


def _rewrite_sphere_material(mtlx_path: str, inputs: dict) -> None:
    """Replace shape_2_mat_surface's standard_surface inputs in the .mtlx.

    Uses the MaterialX API so the result is guaranteed valid. shape_0 (emitter)
    and shape_1 (ground) nodes are untouched.
    """
    import MaterialX as mx

    doc = mx.createDocument()
    mx.readFromXmlFile(doc, mtlx_path)
    node = doc.getNode("shape_2_mat_surface")
    if node is None:
        raise SystemExit(f"{mtlx_path}: shape_2_mat_surface node not found")
    # Drop the placeholder inputs, then re-add the mapped OpenPBR ones.
    for inp in list(node.getInputs()):
        node.removeInput(inp.getName())
    for name, (mtx_type, value) in inputs.items():
        inp = node.addInput(name, mtx_type)
        inp.setValueString(value)
    valid, msg = doc.validate()
    if not valid:
        raise SystemExit(f"{mtlx_path}: MaterialX validation failed: {msg}")
    mx.writeToXmlFile(doc, mtlx_path)


def build_one(name: str, cfg: dict) -> None:
    folder = os.path.join(SUITE, name)
    os.makedirs(folder, exist_ok=True)
    # Neutral pbrt is a scratch input for the importer; write it under _gen so it
    # is NOT mistaken for a runnable suite .pbrt counterpart.
    scratch = os.path.join(os.path.dirname(__file__), f"_neutral_{name}.pbrt")
    with open(scratch, "w") as fh:
        fh.write(_neutral_pbrt(name, spp=cfg["spp"], maxdepth=cfg["maxdepth"]))

    from skinny.pbrt.api import import_pbrt

    mtlx_usda = os.path.join(folder, f"{name}_mtlx.usda")
    import_pbrt(scratch, out=mtlx_usda, materialx=True)
    mtlx = os.path.join(folder, f"{name}_mtlx.mtlx")
    _rewrite_sphere_material(mtlx, cfg["inputs"])
    os.remove(scratch)

    # Prune any orphan env HDR the importer may have emitted (the neutral scene
    # has no infinite light, so there should be none — belt and braces).
    for f in os.listdir(folder):
        if f.startswith("light_infinite_") and f.endswith("_const.hdr"):
            os.remove(os.path.join(folder, f))
    print(f"  -> {mtlx_usda} (+ .mtlx)  [{cfg['source']}]")


def main() -> None:
    only = sys.argv[1:] or list(PBR)
    for name in only:
        if name not in PBR:
            raise SystemExit(f"unknown scene {name!r}; known: {sorted(PBR)}")
        print(f"building {name}")
        build_one(name, PBR[name])


if __name__ == "__main__":
    main()
