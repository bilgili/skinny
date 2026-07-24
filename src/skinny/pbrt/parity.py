"""Parity harness: render an imported pbrt scene in skinny and compare to a
checked-in pbrt v4 reference EXR (design D8/D9).

The comparison uses skinny's **linear-HDR accumulation** (not the tonemapped
sRGB display) against the reference, with relMSE + FLIP gated per scene. Heavy
imports (the renderer, GPU, USD) are lazy so this module imports without a GPU.

This module is the **GPU render adapter** (`render_linear`, `render_combo`,
`evaluate`, plus the scene-source/env helpers) over the pure matrix core in
:mod:`skinny.pbrt.parity_core` — schema, validity oracle, tolerance tables and
result builders, all re-exported below so the historical
``skinny.pbrt.parity`` surface is unchanged.

Reference EXRs are generated offline with a pbrt v4 binary (see the corpus
manifest); the gate itself needs no pbrt binary.
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from . import metrics

# ─── compatibility facade ─────────────────────────────────────────────────
#
# Explicit (never ``import *``) so the consumed private names travel too: tests
# read ``parity._DEFAULT_SELF_CONSISTENCY`` / ``_DEFAULT_SPECTRAL_SELF_CONSISTENCY``
# and call ``parity._render_log`` directly (design D2).
from .parity_core import (  # noqa: F401  (re-exported for the historical surface)
    _DEFAULT_SELF_CONSISTENCY,
    _DEFAULT_SPECTRAL_SELF_CONSISTENCY,
    _render_log,
    ANCHOR,
    EXECUTION_MODES,
    INTEGRATORS,
    PROPOSAL_AXES,
    REUSE_AXES,
    SPECTRAL_ANCHOR,
    ParityResult,
    RenderCombo,
    SceneSpec,
    absolute_radiance_result,
    all_combos,
    authoring_equivalence_result,
    combo_axis_class,
    combo_is_valid,
    enumerate_combos,
    load_manifest,
    materialx_specs,
    pbrt_truth_result,
    reference_exists,
    render_log_path,
    self_consistency_anchor,
    self_consistency_result,
    self_consistency_tol,
    spectral_envelope,
    spectral_selfconsistency_assertable,
)


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
                  emissive_uniform: bool = False,
                  materialx: bool = False,
                  proposals: str | None = None,
                  reuse: str | None = None,
                  usd_path: str | None = None,
                  furnace: bool = False,
                  furnace_material: int | None = None,
                  spectral: bool = False) -> np.ndarray:
    """Render a scene in skinny; return linear-HDR (H,W,3).

    The scene source is either a pbrt file (*scene_pbrt*, imported to USD at call
    time) or, when *usd_path* is set, an existing ``.usda`` asset loaded directly
    (used for the heavy bathroom/dragon scenes).

    *gpu* is the vendor preference (intel/nvidia/amd/discrete/auto); the rhi
    backend (vulkan/metal) is resolved via :func:`skinny.backend_select.select_backend`
    — ``auto`` → native Metal on a Metal-capable Apple-Silicon host (full parity
    with Vulkan), else Vulkan; honours ``SKINNY_BACKEND``. So the parity /
    convergence gates exercise the host's real default backend rather than always
    MoltenVK-under-Vulkan.
    *env_off* zeroes skinny's default ambient environment so scenes with no pbrt
    ``infinite`` light render against a black background as pbrt does.
    *integrator* selects ``"path"``, ``"bdpt"`` or ``"sppm"``.
    *proposals* / *reuse* arm the scene-sampling axes (constructor-only on the
    headless renderer): ``proposals="bsdf,neural"`` activates the neural
    directional proposal (asserted live); ``reuse="restir-di"`` activates ReSTIR
    DI direct-light reuse.
    *emissive_uniform* (test hook) forces uniform-by-index emissive-triangle
    selection instead of the default power-weighted distribution, so the same
    binary can render the power-vs-uniform A/B for the emissive-mesh-nee gate.
    *materialx* imports through the ``-mtlx`` path (rich ``standard_surface``
    sidecar) instead of UsdPreviewSurface, so the same reference EXRs gate both
    export paths; the bound meshes resolve their rich overrides via the usd_loader
    ``.mtlx`` intake.
    *furnace* enables white-furnace energy-closure mode (constant-white
    environment, analytic lights disabled) by setting ``renderer.furnace_index``
    before accumulation; *furnace_material*, when given, arms the *per-material*
    furnace bit (bit 10) on that material index only instead of the global mode
    (change confirming-test-scenes / furnace-closure).
    Requires a working GPU backend; raises if unavailable.
    """
    from skinny.backend_select import select_backend
    from skinny.headless import HeadlessRenderer, RenderOptions  # lazy: renderer/GPU

    from .api import import_pbrt  # lazy: pulls in pxr/USD

    # SPPM and MLT are wavefront-only (no megakernel path) — force the execution
    # mode so callers can pass the integrator without also threading it.
    if integrator in ("sppm", "mlt"):
        execution_mode = "wavefront"

    backend = select_backend()
    want_neural = bool(proposals) and "neural" in proposals

    def _run(scene_usd: str) -> np.ndarray:
        with HeadlessRenderer(width, height, gpu=gpu, backend=backend,
                              execution_mode=execution_mode,
                              proposals=proposals, reuse=reuse,
                              spectral=spectral) as r:
            # Set before the scene build so _upload_emissive_triangles sees it.
            r.renderer._emissive_uniform_selection = bool(emissive_uniform)
            r._prepare(scene_usd, RenderOptions(samples=spp, integrator=integrator))
            if want_neural and not r.renderer._neural_active():
                raise RuntimeError(
                    "neural proposal requested but not active (needs wavefront + a "
                    "neural proposal token + a flat-material first hit)"
                )
            if env_off:
                r.renderer.env_intensity = 0.0
            # skinny synthesizes a default DistantLight for scenes that author no
            # directional light (the per-frame mirror falls back to the slider
            # light only when `_usd_scene.lights_dir` is empty); a pbrt scene is
            # fully lit by its own lights, so disable that default to avoid a
            # phantom extra shadow. `direct_light_index` is a GLOBAL off switch —
            # it also zeroes AUTHORED distant lights (`_upload_distant_lights`) —
            # so it must stay 0 for scenes that author one (disney-cloud's sun
            # rendered black under the unconditional disable;
            # nanovdb-volume-rendering).
            authored_dir = bool(getattr(r.renderer._usd_scene, "lights_dir", None))
            r.renderer.direct_light_index = 0 if authored_dir else 1
            # White-furnace closure (change confirming-test-scenes): global
            # furnace swaps in the constant-white env + disables lights; the
            # per-material path arms only one material's furnace bit and leaves
            # the scene's own lighting so the flagged object closes while the
            # rest renders normally.
            if furnace and furnace_material is None:
                r.renderer.furnace_index = 1
            elif furnace_material is not None:
                r.renderer.toggle_material_furnace(furnace_material, True)
            r.renderer._last_state_hash = None
            r._accumulate(spp)
            arr, _samples = r.renderer.read_accumulation_hdr()
            # Apply the pbrt film imaging ratio (exposure_time·iso/100) read from the
            # authored camera as a linear output scale, so the headless A/B sees
            # pbrt-equivalent absolute radiance (change pbrt-radiometric-parity). The
            # ratio is no longer baked into emitters at import; it rides the camera
            # film params (FilmParameters), set by _apply_camera_override. ratio 1.0
            # for a default-film scene ⇒ unchanged.
            ratio = float(r.renderer.film.imaging_ratio())
            out = np.asarray(arr, dtype=np.float64)[..., :3]
            return out * ratio if ratio != 1.0 else out

    if usd_path is not None:
        return _run(usd_path)
    with tempfile.TemporaryDirectory() as tmp:
        usd = os.path.join(tmp, "scene.usda")
        import_pbrt(scene_pbrt, out=usd, materialx=materialx)
        return _run(usd)


def _repo_root(corpus_dir: str) -> str:
    # corpus_dir == <repo>/tests/pbrt/corpus
    return os.path.abspath(os.path.join(corpus_dir, "..", "..", ".."))


def _scene_source(spec: SceneSpec, corpus_dir: str) -> dict:
    """Resolve the scene source into render_linear kwargs (pbrt file or usd asset)."""
    if spec.usd:
        usd = spec.usd if os.path.isabs(spec.usd) else os.path.join(_repo_root(corpus_dir), spec.usd)
        return {"scene_pbrt": usd, "usd_path": usd}
    return {"scene_pbrt": os.path.join(corpus_dir, spec.file), "usd_path": None}


def _usd_has_dome(usd_path: str) -> bool:
    """Cheap text scan for a dome/environment light in a .usda."""
    try:
        with open(usd_path, encoding="utf-8", errors="ignore") as fh:
            head = fh.read(200_000)
    except OSError:
        return False
    return "DomeLight" in head


def _env_off_for(spec: SceneSpec, corpus_dir: str, src: dict) -> bool:
    """True if skinny's default ambient env should be zeroed for this scene."""
    if src["usd_path"] is not None:
        return not _usd_has_dome(src["usd_path"])
    return not scene_has_environment(src["scene_pbrt"])


def render_combo(spec: SceneSpec, combo: RenderCombo, corpus_dir: str,
                 gpu: str | None = None) -> np.ndarray:
    """Render *spec* with *combo* and return the linear-HDR image (H,W,3)."""
    src = _scene_source(spec, corpus_dir)
    _render_log(f"START {spec.name:24s} {combo.label}")
    t0 = time.time()
    img = render_linear(
        src["scene_pbrt"], spec.width, spec.height, spp=spec.spp,
        gpu=gpu, env_off=_env_off_for(spec, corpus_dir, src),
        integrator=combo.integrator, execution_mode=combo.execution_mode,
        proposals=combo.proposals_token(), reuse=combo.reuse_token(),
        materialx=spec.materialx, usd_path=src["usd_path"],
        spectral=combo.spectral,
    )
    _render_log(f"DONE  {spec.name:24s} {combo.label}  ({time.time() - t0:.1f}s)")
    return img


def evaluate(spec: SceneSpec, corpus_dir: str, gpu: str | None = None,
             combo: RenderCombo | None = None) -> ParityResult:
    """Render *spec* (default: the path/megakernel combo) and gate against its
    reference EXR. Honours ``spec.materialx`` and ``spec.usd``.
    """
    if combo is None:
        combo = RenderCombo(integrator="path", execution_mode="megakernel")
    ref = metrics.read_exr(os.path.join(corpus_dir, spec.ref))
    img = render_combo(spec, combo, corpus_dir, gpu=gpu)
    return pbrt_truth_result(spec, combo, img, ref)
