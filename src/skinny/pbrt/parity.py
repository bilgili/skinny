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
    fields = set(SceneSpec.__dataclass_fields__)
    return [SceneSpec(**{k: v for k, v in s.items() if k in fields}) for s in data["scenes"]]


def scene_has_environment(scene_pbrt: str) -> bool:
    """True if the pbrt scene defines an ``infinite`` light (an environment)."""
    from .parser import parse_file
    from .state import build_scene

    scene = build_scene(parse_file(scene_pbrt))
    return any(light.type == "infinite" for light in scene.lights)


def render_linear(scene_pbrt: str, width: int, height: int, spp: int,
                  gpu: str | None = None, env_off: bool = False,
                  integrator: str = "path",
                  execution_mode: str = "megakernel",
                  emissive_uniform: bool = False) -> np.ndarray:
    """Import a pbrt scene and render it in skinny; return linear-HDR (H,W,3).

    *gpu* is the vendor preference (intel/nvidia/amd/discrete/auto); the rhi
    backend (vulkan/metal) is resolved via :func:`skinny.backend_select.select_backend`
    — ``auto`` → native Metal on a Metal-capable Apple-Silicon host (full parity
    with Vulkan), else Vulkan; honours ``SKINNY_BACKEND``. So the parity /
    convergence gates exercise the host's real default backend rather than always
    MoltenVK-under-Vulkan.
    *env_off* zeroes skinny's default ambient environment so scenes with no pbrt
    ``infinite`` light render against a black background as pbrt does.
    *integrator* selects ``"path"`` or ``"bdpt"``.
    *emissive_uniform* (test hook) forces uniform-by-index emissive-triangle
    selection instead of the default power-weighted distribution, so the same
    binary can render the power-vs-uniform A/B for the emissive-mesh-nee gate.
    Requires a working GPU backend; raises if unavailable.
    """
    from skinny.backend_select import select_backend
    from skinny.headless import HeadlessRenderer, RenderOptions  # lazy: renderer/GPU

    backend = select_backend()
    with tempfile.TemporaryDirectory() as tmp:
        usd = os.path.join(tmp, "scene.usda")
        import_pbrt(scene_pbrt, out=usd)
        with HeadlessRenderer(width, height, gpu=gpu, backend=backend,
                              execution_mode=execution_mode) as r:
            # Set before the scene build so _upload_emissive_triangles sees it.
            r.renderer._emissive_uniform_selection = bool(emissive_uniform)
            r._prepare(usd, RenderOptions(samples=spp, integrator=integrator))
            if env_off:
                r.renderer.env_intensity = 0.0
            # skinny injects a synthetic default DistantLight (/Skinny/DefaultLight)
            # onto every loaded scene; a pbrt scene is fully lit by its own lights,
            # so disable the default key light to avoid a phantom extra shadow.
            r.renderer.direct_light_index = 1
            r.renderer._last_state_hash = None
            r._accumulate(spp)
            arr, _samples = r.renderer.read_accumulation_hdr()
    return np.asarray(arr, dtype=np.float64)[..., :3]


def evaluate(spec: SceneSpec, corpus_dir: str, gpu: str | None = None) -> ParityResult:
    """Render *spec* in skinny and compare to its reference EXR."""
    ref_path = os.path.join(corpus_dir, spec.ref)
    ref = metrics.read_exr(ref_path)
    scene_path = os.path.join(corpus_dir, spec.file)
    img = render_linear(
        scene_path, spec.width, spec.height, spp=spec.spp,
        gpu=gpu, env_off=not scene_has_environment(scene_path),
    )
    aligned = metrics.align_exposure(img, ref)
    rm = metrics.relmse(aligned, ref)
    fl = metrics.flip(aligned, ref)
    passed = rm <= spec.relmse_tol and fl <= spec.flip_tol
    return ParityResult(spec.name, rm, fl, passed)


def reference_exists(spec: SceneSpec, corpus_dir: str) -> bool:
    return os.path.isfile(os.path.join(corpus_dir, spec.ref))
