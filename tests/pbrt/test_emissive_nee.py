"""Emissive-mesh NEE gate (change `emissive-mesh-nee`).

GPU tests reuse the pbrt parity harness (`skinny.pbrt.parity.render_linear`,
native Metal via `select_backend`) to render synthetic emissive-mesh scenes:

  * correctness  — a >256-triangle emissive mesh keeps all its energy (no silent
                   256-cap truncation) and converges to an equivalent low-poly
                   emitter.
  * variance     — power-weighted selection beats uniform-by-index at equal spp
                   on an uneven multi-emitter scene.
  * unbiasedness — power and uniform selection converge to the same mean image.
  * no-regression— the small `diffuse_arealight` quad stays within corpus parity.

The power-vs-uniform A/B uses `render_linear(..., emissive_uniform=True)`, which
flips the host `Renderer._emissive_uniform_selection` toggle: the *same* shader
path then reproduces exact uniform-by-index selection (the inline CDF is built
uniform), so the two renders differ only in the packed selection distribution.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from skinny.pbrt import parity


# ── pbrt scene synthesis ──────────────────────────────────────────────

def _tess_quad(cx: float, cy: float, cz: float, half: float, n: int):
    """An n×n-tessellated horizontal quad (plane y=cy) → (points, indices).

    Winding matches the `diffuse_arealight` corpus quad (geometric normal -y),
    so the area light emits downward onto the scene below it.
    """
    pts: list[float] = []
    for j in range(n + 1):
        for i in range(n + 1):
            x = cx - half + 2.0 * half * i / n
            z = cz - half + 2.0 * half * j / n
            pts += [x, cy, z]
    idx: list[int] = []
    stride = n + 1
    for j in range(n):
        for i in range(n):
            a = j * stride + i
            b = j * stride + i + 1
            c = (j + 1) * stride + i + 1
            d = (j + 1) * stride + i
            idx += [a, b, c, a, c, d]
    return pts, idx


def _emitter_block(L, cx, cy, cz, half, n) -> str:
    pts, idx = _tess_quad(cx, cy, cz, half, n)
    pstr = " ".join(f"{v:g}" for v in pts)
    istr = " ".join(str(v) for v in idx)
    return (
        "AttributeBegin\n"
        f'  AreaLightSource "diffuse" "rgb L" [{L[0]:g} {L[1]:g} {L[2]:g}]\n'
        f'  Shape "trianglemesh" "point3 P" [ {pstr} ] '
        f'"integer indices" [ {istr} ]\n'
        "AttributeEnd\n"
    )


def _scene(emitters, *, res: int = 64, spp: int = 128, receiver: bool = True) -> str:
    """Assemble a pbrt scene: camera + emitter quads + a diffuse sphere/floor."""
    head = (
        "LookAt 0 1 6  0 0 0  0 1 0\n"
        'Camera "perspective" "float fov" 35\n'
        f'Sampler "independent" "integer pixelsamples" {spp}\n'
        'Integrator "path" "integer maxdepth" 8\n'
        f'Film "rgb" "integer xresolution" {res} "integer yresolution" {res}\n'
        "WorldBegin\n"
    )
    body = "".join(_emitter_block(**e) for e in emitters)
    if receiver:
        body += (
            "AttributeBegin\n"
            '  Material "diffuse" "rgb reflectance" [0.7 0.5 0.4]\n'
            '  Shape "sphere" "float radius" 1\n'
            "AttributeEnd\n"
            "AttributeBegin\n"
            '  Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]\n'
            "  Translate 0 -1 0\n"
            '  Shape "trianglemesh" "point3 P" '
            "[ -6 0 -6  6 0 -6  6 0 6  -6 0 6 ] "
            '"integer indices" [ 0 1 2 0 2 3 ]\n'
            "AttributeEnd\n"
        )
    return head + body


def _render(scene: str, *, res: int, spp: int, emissive_uniform: bool = False):
    # render_linear takes a pbrt *file path*; stage the scene text in a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".pbrt", delete=False) as fh:
        fh.write(scene)
        path = fh.name
    try:
        return parity.render_linear(
            path, res, res, spp=spp, env_off=True,
            emissive_uniform=emissive_uniform,
        )
    finally:
        os.unlink(path)


def _mean(img) -> float:
    return float(np.mean(np.asarray(img, dtype=np.float64)))


# ── tests ─────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_high_poly_emissive_keeps_energy():
    """A >256-triangle emissive quad must illuminate the scene the same as a
    2-triangle quad of equal area and radiance.

    Pre-fix the 256-cap drops half of the 16×16 (512-tri) quad's triangles, so
    the high-poly scene is biased ~2x dark — this asserts the fixed behaviour
    (energy converges), so it is RED until the cap is removed.
    """
    res, spp = 64, 96
    half, cy = 1.5, 4.0
    low = _render(_scene([dict(L=(12, 12, 12), cx=0, cy=cy, cz=0, half=half, n=1)],
                         res=res, spp=spp), res=res, spp=spp)
    high = _render(_scene([dict(L=(12, 12, 12), cx=0, cy=cy, cz=0, half=half, n=16)],
                          res=res, spp=spp), res=res, spp=spp)
    m_low, m_high = _mean(low), _mean(high)
    assert m_low > 1e-4, f"low-poly emitter produced no light (mean={m_low:.5f})"
    ratio = m_high / m_low
    assert 0.9 <= ratio <= 1.1, (
        f"high-poly (512-tri) emitter energy diverged from low-poly: "
        f"mean_high={m_high:.5f} mean_low={m_low:.5f} ratio={ratio:.3f} "
        f"(expected ~1.0; <1 ⇒ triangles silently truncated)"
    )


@pytest.mark.gpu
def test_power_weighted_lower_variance():
    """On an uneven multi-emitter scene (one bright emitter + many dim, occluded
    distractor triangles), power-weighted selection has materially lower equal-spp
    variance than uniform-by-index.

    Uniform picks ~all of its samples on the 512 dim occluded triangles (≈0.4%
    land on the bright emitter); power picks the bright emitter ≈95% of the time.
    Both are unbiased, so each is measured against a converged power reference —
    the uniform render's relMSE is dominated by its much higher variance.
    """
    from skinny.pbrt import metrics

    res = 48
    bright = dict(L=(60, 60, 60), cx=0, cy=4.0, cz=0, half=0.5, n=1)   # 2 tris
    # 512 dim triangles below the floor (y=-4, floor at y=-1) → occluded, ~0
    # contribution, but they flood the uniform selection distribution.
    distract = dict(L=(0.2, 0.2, 0.2), cx=0, cy=-4.0, cz=0, half=2.0, n=16)
    scene = _scene([bright, distract], res=res, spp=1)

    ref = _render(scene, res=res, spp=384)             # converged (power)
    pw = _render(scene, res=res, spp=48)               # power, equal-spp
    un = _render(scene, res=res, spp=48, emissive_uniform=True)  # uniform, equal-spp

    rm_power = metrics.relmse(pw, ref)
    rm_uniform = metrics.relmse(un, ref)
    assert rm_uniform > 3.0 * rm_power, (
        f"power-weighted selection not materially lower variance: "
        f"relMSE(power)={rm_power:.4f} relMSE(uniform)={rm_uniform:.4f} "
        f"(expected uniform > 3x power)"
    )


@pytest.mark.gpu
def test_power_weighted_unbiased():
    """Power and uniform selection converge to the same mean image — only the
    variance differs, not the expected value.

    A mild 2-emitter scene (different brightness, few triangles) so uniform
    converges at a feasible spp; the converged means must match within noise.
    """
    res, spp = 48, 256
    a = dict(L=(30, 30, 30), cx=-1.0, cy=4.0, cz=0, half=0.6, n=2)
    b = dict(L=(6, 6, 6), cx=1.2, cy=4.0, cz=0, half=0.6, n=2)
    scene = _scene([a, b], res=res, spp=spp)

    m_power = _mean(_render(scene, res=res, spp=spp))
    m_uniform = _mean(_render(scene, res=res, spp=spp, emissive_uniform=True))
    rel = abs(m_power - m_uniform) / max(m_uniform, 1e-6)
    assert rel < 0.05, (
        f"power and uniform selection disagree on the converged mean "
        f"(biased): mean_power={m_power:.5f} mean_uniform={m_uniform:.5f} "
        f"rel={rel:.4f} (expected < 0.05 — same expectation, only variance differs)"
    )


@pytest.mark.gpu
def test_diffuse_arealight_no_regression():
    """The small 2-triangle corpus quad stays within parity vs the pbrt ref."""
    corpus = os.path.join(os.path.dirname(__file__), "corpus")
    specs = {s.name: s for s in parity.load_manifest(corpus)}
    spec = specs["diffuse_arealight"]
    if not parity.reference_exists(spec, corpus):
        pytest.skip("no reference EXR for diffuse_arealight")
    result = parity.evaluate(spec, corpus)
    assert result.passed, (
        f"diffuse_arealight regressed: relMSE={result.relmse:.4f} "
        f"(<= {spec.relmse_tol}), FLIP={result.flip:.4f} (<= {spec.flip_tol})"
    )
