#!/usr/bin/env python3
"""Generate the confirming-scene suite (change confirming-test-scenes).

For every pbrt-expressible scene this writes the hand-authored ``.pbrt`` source
and then produces the committed ``.usda`` (plain UsdPreviewSurface) and
``_mtlx.usda`` + ``.mtlx`` (MaterialX) variants via ``import_pbrt`` — skinny's
own pbrt→USD bridge — so the three authorings are provably the same scene. The
pbrt reference EXRs are generated separately by ``tests/pbrt/regen_refs.py
--scene suite`` (needs the pinned pbrt binary).

Run from the repo root:

    PYTHONPATH=src python tests/assets/suite/_gen/build.py

Idempotent: re-running overwrites the generated files in place.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import geom  # noqa: E402

SUITE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─── canonical mini-shaderball layout ──────────────────────────────────────
# Camera looks slightly down at a unit sphere sitting on a mid-grey floor; a
# downward area light overhead. Shared by every single-material scene so a lobe
# swap is the only variable.
_HEADER = """LookAt 0 1.5 6  0 0.7 0  0 1 0
Camera "perspective" "float fov" 30
Sampler "independent" "integer pixelsamples" {spp}
Integrator "{integrator}" "integer maxdepth" {maxdepth}
Film "rgb" "integer xresolution" 128 "integer yresolution" 128 "string filename" "{name}.exr"
WorldBegin
"""

_AREA_LIGHT = """AttributeBegin
  AreaLightSource "diffuse" "rgb L" [{lr} {lg} {lb}]
  {light_geom}
AttributeEnd
"""

_GROUND = """AttributeBegin
  Material "diffuse" "rgb reflectance" [0.18 0.18 0.18]
  {ground_geom}
AttributeEnd
"""


def _sphere_block(material: str, cx=0.0, cy=0.7, cz=0.0, r=0.7,
                  emissive: str | None = None) -> str:
    pts, idx = geom.uv_sphere(cx, cy, cz, r)
    inner = ""
    if emissive is not None:
        inner += f"  AreaLightSource \"diffuse\" \"rgb L\" [{emissive}]\n"
    if material:
        inner += f"  {material}\n"
    inner += f"  {geom.trianglemesh(pts, idx)}\n"
    return f"AttributeBegin\n{inner}AttributeEnd\n"


def _scene(name: str, material: str, *, spp=256, integrator="path", maxdepth=8,
           env: str | None = None, area_light: str | None = "12 12 12",
           emissive_sphere: str | None = None, extra: str = "") -> str:
    parts = [_HEADER.format(spp=spp, integrator=integrator, maxdepth=maxdepth, name=name)]
    if env is not None:
        parts.append(f'LightSource "infinite" "rgb L" [{env}]\n')
    if emissive_sphere is None and area_light is not None:
        lr, lg, lb = area_light.split()
        parts.append(_AREA_LIGHT.format(
            lr=lr, lg=lg, lb=lb,
            light_geom=geom.trianglemesh(*geom.ceiling_light())))
    parts.append(_GROUND.format(ground_geom=geom.trianglemesh(*geom.ground())))
    if extra:
        parts.append(extra)
    parts.append(_sphere_block(material, emissive=emissive_sphere))
    return "".join(parts)


# ─── scene table ────────────────────────────────────────────────────────────
# Each entry: (folder, pbrt-text). material_class / spp / integrator captured in
# the pbrt itself; the manifest (written by hand) records tolerances + gates.

def _two_spheres(name: str, mat_a: str, mat_b: str, *, spp=256, env=None,
                 area_light="12 12 12") -> str:
    parts = [_HEADER.format(spp=spp, integrator="path", maxdepth=8, name=name)]
    if env is not None:
        parts.append(f'LightSource "infinite" "rgb L" [{env}]\n')
    lr, lg, lb = area_light.split()
    parts.append(_AREA_LIGHT.format(lr=lr, lg=lg, lb=lb,
                                    light_geom=geom.trianglemesh(*geom.ceiling_light())))
    parts.append(_GROUND.format(ground_geom=geom.trianglemesh(*geom.ground())))
    parts.append(_sphere_block(mat_a, cx=-0.9, r=0.7))
    parts.append(_sphere_block(mat_b, cx=0.9, r=0.7))
    return "".join(parts)


def _diffuse(r: float, g: float, b: float) -> str:
    return f'Material "diffuse" "rgb reflectance" [{r} {g} {b}]'


def _wall(pts, idx, material: str, emissive: str | None = None) -> str:
    inner = ""
    if emissive is not None:
        inner += f"  AreaLightSource \"diffuse\" \"rgb L\" [{emissive}]\n"
    inner += f"  {material}\n  {geom.trianglemesh(pts, idx)}\n"
    return f"AttributeBegin\n{inner}AttributeEnd\n"


def _cornell_box(name: str, *, left="0.63 0.06 0.05", right="0.14 0.45 0.09",
                 neutral="0.73 0.73 0.73", spp=256, maxdepth=8,
                 emitter="16 16 16", baffle=False, center_obj: str | None = None) -> str:
    """A [-2,2]×[0,4]×[-2,2] Cornell box; camera looks in along -Z."""
    A, B = -2.0, 2.0
    parts = [_HEADER.format(spp=spp, integrator="path", maxdepth=maxdepth, name=name)]
    # emitter: small downward quad recessed at the ceiling
    parts.append(_wall(*geom.ceiling_light(half=0.7, y=3.95), material=_diffuse(0.0, 0.0, 0.0),
                       emissive=emitter))
    if baffle:  # opaque lip below the emitter → direct light blocked, indirect dominates
        parts.append(_wall(*geom.ceiling_light(half=1.1, y=3.5), material=_diffuse(0.5, 0.5, 0.5)))
    # floor, ceiling, back (neutral)
    parts.append(_wall(*geom.quad((A, 0, B), (B, 0, B), (B, 0, A), (A, 0, A)), _diffuse(*neutral.split())))
    parts.append(_wall(*geom.quad((A, 4, A), (B, 4, A), (B, 4, B), (A, 4, B)), _diffuse(*neutral.split())))
    parts.append(_wall(*geom.quad((A, 0, A), (B, 0, A), (B, 4, A), (A, 4, A)), _diffuse(*neutral.split())))
    # left (red), right (green)
    parts.append(_wall(*geom.quad((A, 0, B), (A, 0, A), (A, 4, A), (A, 4, B)), _diffuse(*left.split())))
    parts.append(_wall(*geom.quad((B, 0, A), (B, 0, B), (B, 4, B), (B, 4, A)), _diffuse(*right.split())))
    if center_obj is not None:
        parts.append(_sphere_block(center_obj, cx=0.0, cy=1.0, cz=0.0, r=1.0))
    return "".join(parts)


def _light_grid(name: str, floor_mat: str, *, n=4, spp=256) -> str:
    """A glossy floor under an n×n grid of small emissive quads (ReSTIR DI)."""
    parts = [_HEADER.format(spp=spp, integrator="path", maxdepth=8, name=name)]
    parts.append(_wall(*geom.ground(size=4.0), floor_mat))
    span, y, h = 3.0, 3.0, 0.18
    for i in range(n):
        for j in range(n):
            cx = -span + 2 * span * (i / (n - 1))
            cz = -span + 2 * span * (j / (n - 1))
            q = geom.quad((cx - h, y, cz - h), (cx + h, y, cz - h),
                          (cx + h, y, cz + h), (cx - h, y, cz + h))
            parts.append(_wall(*q, _diffuse(0, 0, 0), emissive="40 40 40"))
    parts.append(_sphere_block(_diffuse(0.5, 0.5, 0.5), cx=0.0, cy=0.7, cz=0.0, r=0.7))
    return "".join(parts)


_CHECKER_QUAD = geom.trianglemesh(
    [(-1.2, -1.2, 0), (1.2, -1.2, 0), (1.2, 1.2, 0), (-1.2, 1.2, 0)],
    [0, 1, 2, 0, 2, 3], uv=[(0, 0), (1, 0), (1, 1), (0, 1)])

_TEXTURED = f"""LookAt 0 0 3.2  0 0 0  0 1 0
Camera "perspective" "float fov" 40
Sampler "independent" "integer pixelsamples" 256
Integrator "path" "integer maxdepth" 5
Film "rgb" "integer xresolution" 128 "integer yresolution" 128 "string filename" "mat_textured.exr"
WorldBegin
LightSource "infinite" "rgb L" [1 1 1]
Texture "uvtex" "spectrum" "imagemap" "string filename" "texture_uv.png"
AttributeBegin
  Material "diffuse" "texture reflectance" "uvtex"
  {_CHECKER_QUAD}
AttributeEnd
"""

SCENES: dict[str, str] = {
    # ── materials (one lobe family each) ──
    "mat_diffuse": _scene("mat_diffuse",
                          'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]'),
    "mat_conductor": _two_spheres(
        "mat_conductor",
        'Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0',
        'Material "conductor" "spectrum eta" "metal-Au-eta" "spectrum k" "metal-Au-k" "float roughness" 0.3',
        env="0.4 0.4 0.4"),
    "mat_dielectric": _scene(
        "mat_dielectric",
        'Material "dielectric" "float eta" 1.5', maxdepth=12, env="0.5 0.5 0.5"),
    "mat_plastic": _scene(
        "mat_plastic",
        'Material "coateddiffuse" "rgb reflectance" [0.2 0.3 0.7] "float roughness" 0.1'),
    "mat_emissive": _scene(
        "mat_emissive", 'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]',
        emissive_sphere="6 5 4"),
    "mat_subsurface": _scene(
        "mat_subsurface",
        'Material "subsurface" "rgb sigma_a" [0.02 0.04 0.06] "rgb sigma_s" [2.5 2.5 3.0] "float eta" 1.33',
        integrator="volpath", maxdepth=32),
    # ── textured material ──
    "mat_textured": _TEXTURED,
    # ── integrators (transport discriminators) ──
    "int_caustic": _scene(
        "int_caustic",
        'Material "dielectric" "float eta" 1.5', maxdepth=16, area_light="30 30 30"),
    "int_indirect_box": _cornell_box(
        "int_indirect_box", left="0.73 0.73 0.73", right="0.73 0.73 0.73",
        emitter="40 40 40", baffle=True, maxdepth=12),
    "int_bleed": _cornell_box(
        "int_bleed", emitter="16 16 16", maxdepth=10,
        center_obj=_diffuse(0.73, 0.73, 0.73)),
    # ── sampling modes ──
    "samp_many_lights": _light_grid(
        "samp_many_lights",
        'Material "coateddiffuse" "rgb reflectance" [0.2 0.2 0.2] "float roughness" 0.05'),
    "samp_env_glossy": _scene(
        "samp_env_glossy",
        'Material "conductor" "spectrum eta" "metal-Al-eta" "spectrum k" "metal-Al-k" "float roughness" 0.25',
        env="0.8 0.8 0.9", area_light=None),
    # ── furnace closure (lossless materials; reference is the analytic 1.0) ──
    "furnace_lambert": _scene(
        "furnace_lambert", 'Material "diffuse" "rgb reflectance" [1 1 1]', spp=128),
    "furnace_conductor": _scene(
        "furnace_conductor",
        'Material "conductor" "float roughness" 0 "rgb reflectance" [1 1 1]', spp=128),
    "furnace_dielectric": _scene(
        "furnace_dielectric", 'Material "dielectric" "float eta" 1.5', spp=128, maxdepth=16),
    "furnace_rough_conductor": _scene(
        "furnace_rough_conductor",
        'Material "conductor" "float roughness" 0.4 "rgb reflectance" [1 1 1]', spp=128),
    "furnace_per_material": _two_spheres(
        "furnace_per_material",
        'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]',
        'Material "diffuse" "rgb reflectance" [1 1 1]', spp=128),
}


def build_one(name: str, pbrt_text: str, *, do_import: bool = True) -> None:
    folder = os.path.join(SUITE, name)
    os.makedirs(folder, exist_ok=True)
    pbrt_path = os.path.join(folder, f"{name}.pbrt")
    with open(pbrt_path, "w") as fh:
        fh.write(pbrt_text)
    print(f"wrote {pbrt_path}")
    if not do_import:
        return
    from skinny.pbrt.api import import_pbrt
    plain = os.path.join(folder, f"{name}.usda")
    import_pbrt(pbrt_path, out=plain, materialx=False)
    print(f"  -> {plain}")
    # Furnace scenes are energy-closure probes (no authoring-equivalence gate), so
    # they ship only the plain USD — no MaterialX variant to keep the tree lean.
    if name.startswith("furnace_"):
        return
    mtlx = os.path.join(folder, f"{name}_mtlx.usda")
    import_pbrt(pbrt_path, out=mtlx, materialx=True)
    print(f"  -> {mtlx} (+ .mtlx)")


def main() -> None:
    only = sys.argv[1:] or list(SCENES)
    for name in only:
        if name not in SCENES:
            raise SystemExit(f"unknown scene {name!r}; known: {sorted(SCENES)}")
        build_one(name, SCENES[name])


if __name__ == "__main__":
    main()
