"""Parity harness: render an imported pbrt scene in skinny and compare to a
checked-in pbrt v4 reference EXR (design D8/D9).

The comparison uses skinny's **linear-HDR accumulation** (not the tonemapped
sRGB display) against the reference, with relMSE + FLIP gated per scene. Heavy
imports (the renderer, GPU) are lazy so this module imports without a GPU.

Reference EXRs are generated offline with a pbrt v4 binary (see the corpus
manifest); the gate itself needs no pbrt binary.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from . import metrics
from .api import import_pbrt


@dataclass
class SceneSpec:
    name: str
    file: str
    ref: str
    width: int
    height: int
    spp: int
    relmse_tol: float
    flip_tol: float


@dataclass
class ParityResult:
    name: str
    relmse: float
    flip: float
    passed: bool


def load_manifest(corpus_dir: str) -> list[SceneSpec]:
    with open(os.path.join(corpus_dir, "manifest.json")) as fh:
        data = json.load(fh)
    return [SceneSpec(**s) for s in data["scenes"]]


def render_linear(scene_pbrt: str, width: int, height: int, spp: int,
                  backend: str | None = None) -> np.ndarray:
    """Import a pbrt scene and render it in skinny; return linear-HDR (H,W,3).

    Requires a working GPU backend; raises if unavailable.
    """
    from skinny.headless import HeadlessRenderer  # lazy: pulls in the renderer/GPU

    with tempfile.TemporaryDirectory() as tmp:
        usd = os.path.join(tmp, "scene.usda")
        import_pbrt(scene_pbrt, out=usd)
        with HeadlessRenderer(width, height, gpu=backend) as r:
            r.render_to_array(usd, samples=spp)
            arr, _samples = r.renderer.read_accumulation_hdr()
    return np.asarray(arr, dtype=np.float64)[..., :3]


def evaluate(spec: SceneSpec, corpus_dir: str, backend: str | None = None) -> ParityResult:
    """Render *spec* in skinny and compare to its reference EXR."""
    ref_path = os.path.join(corpus_dir, spec.ref)
    ref = metrics.read_exr(ref_path)
    img = render_linear(
        os.path.join(corpus_dir, spec.file), spec.width, spec.height, spp=spec.spp,
        backend=backend,
    )
    aligned = metrics.align_exposure(img, ref)
    rm = metrics.relmse(aligned, ref)
    fl = metrics.flip(aligned, ref)
    passed = rm <= spec.relmse_tol and fl <= spec.flip_tol
    return ParityResult(spec.name, rm, fl, passed)


def reference_exists(spec: SceneSpec, corpus_dir: str) -> bool:
    return os.path.isfile(os.path.join(corpus_dir, spec.ref))
