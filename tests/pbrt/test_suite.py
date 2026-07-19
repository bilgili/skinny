"""Confirming-scene suite gates (change confirming-test-scenes).

The suite (``tests/assets/suite/``) is registered in the shared corpus manifest
as ``usd:``-source scenes. This module owns everything suite-specific:

* **hostless** — file presence, USD load, pbrt-counterpart import, and the
  coverage meta-tests that fail the build when a scene lacks a disposition for an
  applicable gate class (pbrt-truth / authoring-equivalence / furnace).
* **gpu** — the render gates: matrix (pbrt-truth + self-consistency), authoring
  equivalence (plain-USD vs MaterialX), and white-furnace closure.

The gpu gates skip cleanly when no backend/reference is available, mirroring
``test_parity.py``.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from skinny.pbrt import furnace as furnace_mod
from skinny.pbrt import metrics
from skinny.pbrt.parity import (
    ANCHOR,
    RenderCombo,
    authoring_equivalence_result,
    enumerate_combos,
    load_manifest,
    pbrt_truth_result,
    reference_exists,
    render_combo,
    self_consistency_anchor,
    self_consistency_result,
    spectral_selfconsistency_assertable,
)

usd_loader = pytest.importorskip("skinny.usd_loader")

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
REPO_ROOT = os.path.abspath(os.path.join(CORPUS_DIR, "..", "..", ".."))
ALL_SPECS = load_manifest(CORPUS_DIR)
SUITE = [s for s in ALL_SPECS if s.suite]
SUITE_BY_NAME = {s.name: s for s in SUITE}
FURNACE = [s for s in SUITE if s.furnace]
# _mtlx variants that declare an equivalence pair (skip-reasoned ones excluded).
EQUIV_PAIRS = [s for s in SUITE if s.equivalence and s.equivalence.get("pair")]
# Suite scenes carrying a spectral-discriminating disposition (change
# spectral-rendering, Group 6.5): a --spectral render is meant to differ from RGB.
SPECTRAL = [s for s in SUITE if s.spectral]

# Focused proposal gate measured on native Metal at 64×64, 128 spp:
# megakernel relMSE=0.002764 / FLIP=0.008695, wavefront
# relMSE=0.003192 / FLIP=0.009214.  The recorded ceilings retain modest
# cross-driver headroom while remaining much tighter than the standing matrix's
# general unbiased-proposal floor.
SPECTRAL_ENV_GATE = {
    "scene": "samp_env_glossy",
    "width": 64,
    "height": 64,
    "spp": 128,
    "relmse": 0.01,
    "flip": 0.015,
}


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _scene_dir(spec) -> str:
    return os.path.dirname(_abs(spec.usd))


# ─── hostless: file presence + coverage meta-tests ─────────────────────────


def test_suite_is_non_empty():
    """Guard against an empty registration (a manifest typo dropping the suite)."""
    assert SUITE, "no suite scenes registered in the manifest (suite=true)"


@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_suite_scene_files_present(spec):
    """Each suite scene's ``usd`` asset exists; a MaterialX variant has a
    ``.mtlx`` in its folder; a pbrt-gated scene has a ``.pbrt`` counterpart."""
    usd = _abs(spec.usd)
    assert os.path.isfile(usd), f"{spec.name}: missing usd {usd}"
    folder = os.path.dirname(usd)
    if spec.name.endswith("_mtlx") or spec.materialx:
        mtlx = [f for f in os.listdir(folder) if f.endswith(".mtlx")]
        assert mtlx, f"{spec.name}: MaterialX variant has no .mtlx in {folder}"
    # A scene that is pbrt-gated (has a ref and no pbrt_skip) must ship the
    # runnable .pbrt counterpart somewhere in its folder.
    if reference_exists(spec, CORPUS_DIR) and not spec.pbrt_skip:
        pbrts = [f for f in os.listdir(folder) if f.endswith(".pbrt")]
        assert pbrts, f"{spec.name}: pbrt-gated scene has no .pbrt counterpart"


@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_suite_coverage_pbrt_disposition(spec):
    """Every suite scene either ships a pbrt counterpart (so it *can* be pbrt-
    gated once the reference EXR is generated) or records a ``pbrt_skip`` reason —
    never a silent gap. The reference EXR itself is a separate, deliberate
    regen step; the gate skips when it is absent."""
    folder = _scene_dir(spec)
    base = spec.name[:-5] if spec.name.endswith("_mtlx") else spec.name
    has_pbrt = os.path.isfile(os.path.join(folder, f"{base}.pbrt"))
    assert has_pbrt or spec.pbrt_skip, (
        f"{spec.name}: no .pbrt counterpart and no pbrt_skip reason — author "
        f"{base}.pbrt or record a pbrt_skip string"
    )


@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_suite_coverage_equivalence_disposition(spec):
    """Every MaterialX variant declares an authoring-equivalence disposition
    (a plain-USD ``pair`` to compare against, or a ``skip`` reason), and a
    declared pair names an existing plain-USD scene."""
    is_mtlx = spec.name.endswith("_mtlx") or spec.materialx
    if not is_mtlx:
        return
    assert spec.equivalence, (
        f"{spec.name}: MaterialX variant missing `equivalence` "
        f"({{'pair': ...}} or {{'skip': reason}})"
    )
    pair = spec.equivalence.get("pair")
    if pair is not None:
        assert pair in SUITE_BY_NAME, f"{spec.name}: equivalence pair {pair!r} not a suite scene"
        assert "relmse" in spec.equivalence and "flip" in spec.equivalence, (
            f"{spec.name}: equivalence pair needs relmse+flip tolerances"
        )
    else:
        assert spec.equivalence.get("skip"), f"{spec.name}: equivalence needs pair or skip"


@pytest.mark.parametrize("spec", FURNACE, ids=[s.name for s in FURNACE])
def test_suite_coverage_furnace_disposition(spec):
    """Every furnace scene records a gate disposition: a uniformity tolerance (the
    object must vanish into the constant furnace) or, for the per-material probe, a
    minimum flagged/unflagged ratio."""
    cfg = spec.furnace
    if cfg.get("per_material"):
        assert "furnace_material" in cfg, (
            f"{spec.name}: per-material furnace needs a furnace_material index"
        )
    else:
        assert ("uniformity_tol" in cfg) or ("tol" in cfg) or ("baseline" in cfg), (
            f"{spec.name}: furnace scene needs a uniformity tolerance or baseline"
        )
    # A furnace scene's reference is the analytic invariant, not a pbrt EXR.
    assert spec.pbrt_skip, f"{spec.name}: furnace scene needs a pbrt_skip reason"


def test_suite_spectral_discriminator_present():
    """At least one suite scene carries a spectral-discriminating disposition
    (change spectral-rendering, Group 6.5) — a scene whose whole point is that a
    ``--spectral`` render differs from the RGB render (a named-glass dispersion
    prism and/or a blackbody-lit scene). This backstops Group 7.2's deferred
    assertion: the discriminator must exist before the spectral GPU sweep (7.3)
    has anything to measure spectral-vs-RGB deltas on."""
    assert SPECTRAL, (
        "no suite scene declares a `spectral` disposition — the spectral axis has "
        "no discriminating confirming-suite scene (Group 6.5)"
    )


@pytest.mark.parametrize("spec", SPECTRAL, ids=[s.name for s in SPECTRAL])
def test_suite_spectral_disposition_wellformed(spec):
    """A spectral disposition names a recognized kind and carries the payload that
    kind needs (a named glass for dispersion, a temperature for blackbody)."""
    cfg = spec.spectral
    kind = cfg.get("kind")
    assert kind in ("dispersion", "blackbody"), f"{spec.name}: unknown spectral kind {kind!r}"
    if kind == "dispersion":
        assert cfg.get("glass"), f"{spec.name}: dispersion disposition needs a `glass` key"
    else:  # blackbody
        assert "temperature" in cfg, f"{spec.name}: blackbody disposition needs a `temperature`"


@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_suite_usd_loads(spec):
    """Each suite USD opens and yields at least one renderable instance (no GPU)."""
    from pxr import Usd

    usd = _abs(spec.usd)
    stage = Usd.Stage.Open(usd)
    assert stage is not None, f"{spec.name}: could not open {usd}"
    scene = usd_loader.load_scene_from_stage(stage)
    assert len(scene.instances) >= 1, f"{spec.name}: no instances loaded"


@pytest.mark.parametrize(
    "spec", [s for s in SUITE if not s.pbrt_skip], ids=[s.name for s in SUITE if not s.pbrt_skip]
)
def test_suite_pbrt_counterpart_imports(spec):
    """A pbrt-gated suite scene's ``.pbrt`` counterpart imports cleanly through the
    pbrt importer (no unsupported feature)."""
    from skinny.pbrt.api import import_pbrt

    folder = _scene_dir(spec)
    # The pbrt counterpart is named after the plain scene (strip a _mtlx suffix).
    base = spec.name[:-5] if spec.name.endswith("_mtlx") else spec.name
    pbrt = os.path.join(folder, f"{base}.pbrt")
    if not os.path.isfile(pbrt):
        pytest.skip(f"{spec.name}: no .pbrt counterpart at {pbrt}")
    stage, report = import_pbrt(pbrt)
    scene = usd_loader.load_scene_from_stage(stage)
    assert len(scene.instances) >= 1
    assert report.count("skipped") == 0, str(report)


# ─── gpu: PBR shaderball smoke ─────────────────────────────────────────────

# Original OpenPBR "physically based material" shaderball cards live only in the
# main checkout (gitignored, absent in worktrees). The smoke test renders one or
# two AS-IS and skips cleanly when they are missing so worktrees still pass.
MATX_ROOT = "/Users/ahmetbilgili/projects/skinny/assets/materialxusd/tests/physically_based"


@pytest.mark.gpu
@pytest.mark.parametrize("card", ["Gold_OPBR_MAT_PBM", "Gray_Card_OPBR_MAT_PBM"])
def test_pbr_shaderball_smoke(card):
    """Render an original OpenPBR shaderball card headless and assert it produces
    a finite, non-black image. Opt-in (gpu-marked); skip-if-missing so worktrees
    without the source cards still pass."""
    src = os.path.join(MATX_ROOT, f"{card}.usda")
    if not os.path.isfile(src):
        pytest.skip(f"original shaderball card absent: {src}")

    try:
        from skinny.backend_select import select_backend
        from skinny.headless import HeadlessRenderer, RenderOptions

        backend = select_backend()
        with HeadlessRenderer(128, 128, backend=backend) as r:
            r._prepare(src, RenderOptions(samples=8, integrator="path"))
            if not r.renderer._backend_render_ready:
                pytest.skip(f"{card}: render pipeline not ready (no usable materials)")
            for _ in range(8):
                r.renderer.update(1.0 / 60.0)
                r.renderer.render_headless()
            hdr, samples = r.renderer.read_accumulation_hdr()
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    img = hdr[..., :3] / max(1, samples)
    assert np.isfinite(img).all(), f"{card}: non-finite pixels in render"
    assert img.mean() > 0, f"{card}: image is entirely black"


# ─── gpu: render gates ─────────────────────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.parametrize("execution_mode", ["megakernel", "wavefront"])
def test_spectral_environment_proposal_converges(execution_mode):
    """Spectral ``{bsdf,env}`` stays consistent with ``{bsdf}`` on the suite's
    IBL/glossy discriminator in both execution modes.

    This is the focused, affordable gate for the environment proposal itself;
    the full suite matrix independently enumerates the same spectral proposal
    axis at each scene's authored resolution and sample count.
    """
    cfg = SPECTRAL_ENV_GATE
    spec = replace(
        SUITE_BY_NAME[cfg["scene"]],
        width=cfg["width"],
        height=cfg["height"],
        spp=cfg["spp"],
    )
    bsdf = RenderCombo("path", execution_mode, spectral=True)
    bsdf_env = RenderCombo("path", execution_mode, ("env",), spectral=True)
    bsdf_img = render_combo(spec, bsdf, CORPUS_DIR)
    env_img = render_combo(spec, bsdf_env, CORPUS_DIR)

    assert np.isfinite(bsdf_img).all()
    assert np.isfinite(env_img).all()
    assert float(bsdf_img.mean()) > 1e-4, "spectral IBL control render is black"

    measured = metrics.compute_metrics(env_img, bsdf_img)
    print(f"[{spec.name}] {bsdf_env.label} vs {bsdf.label}  {measured.summary()}")
    assert measured.relmse <= cfg["relmse"], (
        f"{execution_mode}: spectral environment proposal relMSE "
        f"{measured.relmse:.4f} > recorded {cfg['relmse']:.4f}"
    )
    assert measured.flip <= cfg["flip"], (
        f"{execution_mode}: spectral environment proposal FLIP "
        f"{measured.flip:.4f} > recorded {cfg['flip']:.4f}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("spec", SUITE, ids=[s.name for s in SUITE])
def test_suite_matrix_gate(spec):
    """Suite matrix gate: render every valid combo, run self-consistency vs the
    anchor always, and pbrt-truth when a reference exists (and the scene is not
    ``pbrt_skip``). Furnace scenes are handled by ``test_suite_furnace_gate``."""
    if spec.furnace:
        pytest.skip(f"{spec.name}: furnace scene gated separately")
    combos = enumerate_combos(spec)
    want_pbrt = reference_exists(spec, CORPUS_DIR) and not spec.pbrt_skip
    try:
        ref = metrics.read_exr(os.path.join(CORPUS_DIR, spec.ref)) if want_pbrt else None
        imgs = {c.label: render_combo(spec, c, CORPUS_DIR) for c in combos}
    except Exception as exc:  # noqa: BLE001 - GPU/backend unavailable in this env
        pytest.skip(f"render backend unavailable: {exc}")

    anchor_img = imgs[ANCHOR.label]
    failures: list[str] = []
    for c in combos:
        img = imgs[c.label]
        if want_pbrt:
            pt = pbrt_truth_result(spec, c, img, ref)
            print(f"[{spec.name}] {c.label:28s} pbrt-truth {pt.metrics.summary()}")
            if not pt.passed:
                failures.append(f"{c.label}: pbrt-truth relMSE={pt.relmse:.4f} FLIP={pt.flip:.4f}")
        if c != ANCHOR:
            # Spectral-aware anchor (change spectral-wavefront, D7): a spectral
            # combo gates against the megakernel spectral path image, not the RGB
            # anchor. Report-only fallback if that anchor image is unavailable.
            anchor_combo = self_consistency_anchor(c)
            sc_anchor_img = imgs.get(anchor_combo.label, anchor_img)
            sc = self_consistency_result(spec, c, img, sc_anchor_img)
            print(f"[{spec.name}] {c.label:28s} vs-anchor  {sc.metrics.summary()}")
            if c.spectral and spectral_selfconsistency_assertable(c, spec) \
                    and anchor_combo.label in imgs:
                # mega≡wave ASSERTED for spectral path/bdpt against the spectral
                # anchor; the retained skip is spectral bdpt on a dispersion scene.
                if not sc.passed:
                    failures.append(
                        f"{c.label}: spectral self-consistency vs spectral anchor "
                        f"relMSE={sc.relmse:.4f} FLIP={sc.flip:.4f}"
                    )
            elif c.spectral:
                # Dispersion-splat bdpt (or missing spectral anchor): REPORTED, not
                # asserted (per-splat gamut clamp differs mega-vs-wave — D4/D7).
                print(f"[{spec.name}] {c.label:28s} spectral-vs-anchor "
                      f"Δ relMSE={sc.relmse:.4f} FLIP={sc.flip:.4f} (reported)")
            elif not sc.passed:
                failures.append(
                    f"{c.label}: self-consistency relMSE={sc.relmse:.4f} FLIP={sc.flip:.4f}"
                )

    # 7.3: REPORT the spectral-vs-RGB pbrt-truth direction (not a hard gate). On a
    # smooth chromaticity shift spectral improves; on a hero-λ dispersion caustic
    # (spec_prism) it can regress in pointwise relMSE while being the physically-
    # correct dispersive result — so spectral pbrt-truth is gated against its own
    # recorded baseline (above), and only the direction is logged here.
    if want_pbrt:
        anchor_pt = pbrt_truth_result(spec, ANCHOR, anchor_img, ref)
        for c in combos:
            if not c.spectral:
                continue
            spt = pbrt_truth_result(spec, c, imgs[c.label], ref)
            verdict = "improves" if spt.relmse <= anchor_pt.relmse else "regresses"
            print(f"[{spec.name}] {c.label:28s} spectral-vs-RGB pbrt-truth "
                  f"{verdict}: spectral relMSE={spt.relmse:.4f} vs RGB {anchor_pt.relmse:.4f}")

    if failures and spec.known_divergent:
        # A recorded, not-yet-fixed divergence (e.g. the MaterialX imagemap path);
        # xfail (visible, non-blocking). The follow-up fix flips known_divergent.
        pytest.xfail(f"{spec.name} known-divergent (follow-up):\n  " + "\n  ".join(failures))
    assert not failures, f"{spec.name} suite matrix failures:\n  " + "\n  ".join(failures)


@pytest.mark.gpu
@pytest.mark.parametrize("spec", EQUIV_PAIRS, ids=[s.name for s in EQUIV_PAIRS])
def test_suite_authoring_equivalence_gate(spec):
    """A MaterialX variant's anchor render matches its plain-USD sibling within
    the recorded tolerance (change render-parity-matrix delta)."""
    plain = SUITE_BY_NAME[spec.equivalence["pair"]]
    try:
        mtlx_img = render_combo(spec, ANCHOR, CORPUS_DIR)
        plain_img = render_combo(plain, ANCHOR, CORPUS_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"render backend unavailable: {exc}")
    res = authoring_equivalence_result(spec, plain_img, mtlx_img)
    print(f"[{spec.name}] equivalence vs {plain.name}  {res.metrics.summary()}")
    if not res.passed and spec.known_divergent:
        pytest.xfail(f"{spec.name} known-divergent authoring equivalence (follow-up): "
                     f"relMSE={res.relmse:.4f} FLIP={res.flip:.4f}")
    assert res.passed, (
        f"{spec.name}: authoring equivalence vs {plain.name} "
        f"relMSE={res.relmse:.4f} FLIP={res.flip:.4f}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("spec", FURNACE, ids=[s.name for s in FURNACE])
def test_suite_furnace_gate(spec):
    """White-furnace gate across the scene's valid combos (change furnace-closure).

    For a single-material scene the lossless object must vanish into the constant
    furnace (relative non-uniformity below tolerance). For the per-material scene
    the flagged sphere must light up while its unflagged neighbour stays dark.
    """
    per_material = bool((spec.furnace or {}).get("per_material"))
    combos = enumerate_combos(spec)
    try:
        imgs = {c.label: furnace_mod.render_furnace(spec, c, CORPUS_DIR) for c in combos}
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"render backend unavailable: {exc}")
    failures: list[str] = []
    for c in combos:
        if per_material:
            res = furnace_mod.furnace_per_material_result(spec, imgs[c.label], c)
            print(f"[{spec.name}] {c.label:28s} per-material divergence={res.statistic:.3f} "
                  f"(>= {res.target:.2f})")
            if not res.passed:
                failures.append(f"{c.label}: per-material divergence={res.statistic:.3f} "
                                f"< {res.target:.2f}")
        else:
            res = furnace_mod.furnace_closure_result(spec, imgs[c.label], c)
            tag = "(baseline)" if res.baseline_used else ""
            print(f"[{spec.name}] {c.label:28s} furnace {tag} nonunif={res.statistic:.4f} "
                  f"(<= {res.target:.4f}) envmean={res.mean:.4f}")
            if not res.passed:
                failures.append(f"{c.label}: non-uniformity={res.statistic:.4f} "
                                f"> {res.target:.4f}")
    assert not failures, f"{spec.name} furnace failures:\n  " + "\n  ".join(failures)
