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
import time
from dataclasses import dataclass

import numpy as np

from skinny import mlt_capability, spectral_capability

from . import metrics
from .api import import_pbrt


def render_log_path() -> str:
    """Where per-render progress lines are appended.

    Always on so a long headless sweep is trackable with ``tail -f`` without the
    caller having to remember to redirect output (the matrix/suite renders happen
    in one dict-comprehension, so stdout shows nothing until the run ends).
    Override with ``SKINNY_RENDER_LOG``; default is a stable per-user temp file.
    """
    return os.environ.get(
        "SKINNY_RENDER_LOG",
        os.path.join(tempfile.gettempdir(), "skinny_render_progress.log"),
    )


def _render_log(msg: str) -> None:
    """Append one timestamped line to the render progress log (best-effort)."""
    try:
        with open(render_log_path(), "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except OSError:
        pass  # logging must never break a render


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
    # When True the import goes through the ``-mtlx`` path (rich
    # ``standard_surface`` sidecar) instead of authoring UsdPreviewSurface
    # shaders. The same reference EXRs gate both export paths.
    materialx: bool = False
    # Heavy-scene source: a ``.usda`` asset loaded directly instead of importing
    # ``file`` (a ``.pbrt``) at gate time. When set, ``file`` is informational.
    usd: str | None = None
    # Material class drives the validity table: a ``subsurface`` scene skips the
    # (flat-only) neural axis; ``flat`` scenes exercise it; a ``volume`` scene
    # (heterogeneous participating media, nanovdb-volume-rendering) is
    # path-integrator-only — BDPT/SPPM have no volume transport, and the ReSTIR
    # reuse axis is untested with media (both recorded exclusions, follow-ups).
    material_class: str = "flat"
    # False for geometry too heavy for the megakernel (e.g. the 28.8M-tri
    # dragon, which OOMs) → that scene is wavefront-only.
    megakernel_ok: bool = True
    # Optional per-axis self-consistency tolerances, keyed by axis class
    # ("mode"/"integrator"/"sppm"/"unbiased"): {"mode": {"relmse":.., "flip":..}}.
    self_consistency: dict | None = None
    # Optional per-axis self-consistency tolerances for the SPECTRAL axis, same
    # shape/keys as ``self_consistency`` but consulted only for spectral combos
    # (change spectral-wavefront GPU-validation). Separate because RGB mega≡wave
    # is bit-identical (mode ≡ 0), whereas spectral wavefront threads the hero
    # wavelengths through the staged records and so draws a DIFFERENT sample
    # sequence than the fused megakernel — decorrelated-but-unbiased MC (the means
    # agree; the delta shrinks with spp), which needs a variance-sized floor on
    # high-variance scenes (caustics, dispersion) rather than the RGB 0.02.
    # Overriding this NEVER relaxes the RGB gate, so the strict RGB invariant is
    # preserved. Absent ⇒ the spectral defaults below apply.
    spectral_self_consistency: dict | None = None
    # Optional recorded pbrt-truth baselines, keyed by combo label:
    # {"path|wavefront": {"relmse":.., "flip":..}} — a known mismatch the gate
    # guards against regressing past (it does NOT relax self-consistency).
    baselines: dict | None = None
    # Harness-first allowance: a heavy scene with a known, not-yet-fixed
    # divergence (bathroom mismatches pbrt; BDPT diverges from path on it). When
    # True the matrix gate records the measured battery and ``xfail``s instead of
    # hard-failing, so the suite stays green while the deltas are pinned. The
    # follow-up fix flips this False.
    known_divergent: bool = False
    # Absolute-radiance gate (change pbrt-radiometric-parity). Opt-in per scene:
    # {"mean_ratio_tol": 0.1, "relmse_tol": 0.05, "baselines": {combo.label:
    # {"mean_ratio":.., "relmse":..}}}. Unlike the exposure-blind gate this is NOT
    # alignment-invariant — it catches a global brightness drift (the 1.6×
    # area-light offset). Absent ⇒ the absolute gate is skipped for the scene.
    absolute: dict | None = None
    # ─── confirming-scene-suite fields (change confirming-test-scenes) ──────
    # True marks a scene as a member of the tests/assets/suite/ discriminating
    # corpus (vs the legacy pbrt corpus). Drives the suite coverage meta-tests.
    suite: bool = False
    # A recorded reason this scene has NO pbrt reference EXR (e.g. a MaterialX-
    # only OpenPBR material with no pbrt counterpart, or a furnace scene whose
    # reference is the analytic value 1.0). When set, the pbrt-truth gate is
    # skipped for the scene instead of silently missing.
    pbrt_skip: str | None = None
    # Authoring-equivalence disposition for a MaterialX (_mtlx) variant:
    #   {"pair": "<plain_scene_name>", "relmse":.., "flip":..}  — compare this
    #     scene's anchor render against the named plain-USD sibling's, OR
    #   {"skip": "<reason>"}  — no plain-USD counterpart (OpenPBR-only material).
    # Absent on a plain-USD or single-authoring scene.
    equivalence: dict | None = None
    # Furnace-closure disposition (change confirming-test-scenes / furnace-closure).
    # When set, the scene is a white-furnace energy-closure probe rendered with
    # furnace mode on:
    #   {"material":"lambert"|"conductor"|"dielectric"|"rough_conductor",
    #    "closure":1.0, "tol":0.02}                — asserts mean ≈ closure±tol, or
    #   {..., "baseline":0.85}                     — a recorded legitimate energy
    #     loss (e.g. rough conductor w/o multiple-scattering compensation); the
    #     gate asserts against the baseline (tighten-only).
    #   {..., "per_material":true, "furnace_material":1}  — per-material furnace:
    #     only material index `furnace_material` carries the furnace bit.
    furnace: dict | None = None
    # Spectral-discriminating disposition (change spectral-rendering, Group 6.5).
    # Marks a suite scene whose whole point is that a `--spectral` render differs
    # from the RGB render — e.g. a named-glass dispersion prism (Cauchy IOR splits
    # the hero wavelengths) or a blackbody-lit scene. Shape:
    #   {"kind":"dispersion", "glass":"bk7", "note":...}   — dispersive dielectric,
    #   {"kind":"blackbody", "temperature":T, "note":...}  — blackbody emitter.
    # Consumed by the suite coverage meta-test (its presence is asserted once a
    # discriminator lands) and, on GPU, the spectral-vs-RGB delta report (7.3).
    spectral: dict | None = None


@dataclass
class ParityResult:
    name: str
    relmse: float
    flip: float
    passed: bool
    metrics: "metrics.ImageMetrics | None" = None
    combo: "RenderCombo | None" = None
    baseline_used: bool = False


# ─── render combination matrix ────────────────────────────────────────────
#
# One data-driven validity table mirrors the CLAUDE.md / README compatibility
# matrix. A combo is a point in (integrator × execution_mode × proposals ×
# reuse); ``combo_is_valid`` is the single source of truth for which combos run.

INTEGRATORS = ("path", "bdpt", "sppm", "mlt")
EXECUTION_MODES = ("megakernel", "wavefront")
# Proposal/reuse axes exercised by the matrix (beyond the bare baseline).
PROPOSAL_AXES = ("neural",)
REUSE_AXES = ("restir-di",)

# Capability gate for the spectral axis (change spectral-rendering) — the single
# source of truth lives in :mod:`skinny.spectral_capability` (shared with the
# `--spectral` CLI gate). Until the megakernel transport is wired, spectral
# combos are a recorded "not yet wired" SKIP so the matrix never renders one as
# RGB and gates it as if it were spectral; the validity ENVELOPE
# (:func:`spectral_envelope`) is enforced regardless. Referenced live below so a
# single flip (or a test monkeypatch of ``spectral_capability.SPECTRAL_IMPLEMENTED``)
# takes effect here.


@dataclass(frozen=True)
class RenderCombo:
    """A single renderer configuration the parity matrix can render.

    *proposals* is a tuple of scene-sampling proposal tokens beyond ``bsdf``
    (e.g. ``("neural",)``); *reuse* is ``"none"`` or a reuse-pass token
    (``"restir-di"``).
    """

    integrator: str = "path"
    execution_mode: str = "wavefront"
    proposals: tuple[str, ...] = ()
    reuse: str = "none"
    #: Spectral render variant (hero-wavelength). v1: path + megakernel + flat only.
    spectral: bool = False

    @property
    def has_neural(self) -> bool:
        return "neural" in self.proposals

    @property
    def has_reuse(self) -> bool:
        return bool(self.reuse) and self.reuse != "none"

    def proposals_token(self) -> str | None:
        """The ``proposals=`` string for HeadlessRenderer, or None for baseline."""
        if not self.proposals:
            return None
        return ",".join(("bsdf", *self.proposals))

    def reuse_token(self) -> str | None:
        return self.reuse if self.has_reuse else None

    @property
    def label(self) -> str:
        parts = [self.integrator, self.execution_mode]
        if self.proposals:
            parts.append("+".join(self.proposals))
        if self.has_reuse:
            parts.append(self.reuse)
        if self.spectral:
            parts.append("spectral")
        return "|".join(parts)


#: The self-consistency anchor: the unbiased baseline that supports every axis.
ANCHOR = RenderCombo(integrator="path", execution_mode="wavefront",
                     proposals=(), reuse="none")

#: The spectral self-consistency anchor (change spectral-wavefront, D7). Spectral
#: combos differ from the RGB anchor *by construction* on a spectrum-authored
#: scene (RGB↔spectrum round-trip is not identity), so they are gated against the
#: **megakernel spectral path** image, never the RGB golden. The wavefront spectral
#: path/bdpt combos anchor here (lifting the old blanket "spectral is megakernel-
#: only" self-consistency skip); spectral sppm anchors here too, at the `sppm`
#: tolerance class.
SPECTRAL_ANCHOR = RenderCombo(integrator="path", execution_mode="megakernel",
                              proposals=(), reuse="none", spectral=True)


def self_consistency_anchor(combo: RenderCombo) -> RenderCombo:
    """The anchor combo *combo*'s image is measured against for self-consistency.

    The RGB anchor for RGB combos; the megakernel spectral path anchor for the
    spectral axis (change spectral-wavefront, D7).
    """
    return SPECTRAL_ANCHOR if combo.spectral else ANCHOR


def spectral_selfconsistency_assertable(combo: RenderCombo, scene: SceneSpec) -> bool:
    """Whether a spectral *combo*'s self-consistency vs the spectral anchor is a
    hard assertion (True) or reported-only (False).

    With spectral transport in both execution modes the mega≡wave equivalence is
    asserted for spectral path/bdpt exactly like their RGB counterparts. The one
    retained skip (D4/D7): spectral ``bdpt`` on an out-of-gamut **dispersion**
    (light-tracer splat) scene, whose per-splat gamut clamp is nonlinear and
    differs by splat granularity between the fused (megakernel) and staged
    (wavefront) pipelines. The spectral anchor itself is not self-compared.
    """
    if not combo.spectral:
        return True
    if combo == SPECTRAL_ANCHOR:
        return False  # the anchor is not compared against itself
    if combo.integrator == "bdpt" and (scene.spectral or {}).get("kind") == "dispersion":
        return False  # recorded dispersion-splat mega≡wave skip (D4/D7)
    return True


def spectral_envelope(combo: RenderCombo, scene: SceneSpec) -> tuple[bool, str]:
    """The intended spectral validity envelope, independent of whether the
    transport is wired yet (:data:`SPECTRAL_IMPLEMENTED`).

    v1 (megakernel) admitted path/bdpt under the megakernel; the
    ``spectral-wavefront`` change extends the envelope to the **wavefront**
    execution mode too, for path, bdpt and sppm (flat materials, BSDF proposal,
    no reuse). SPPM has no megakernel path (photon pass is wavefront-only), so
    ``sppm`` is refused under the megakernel; the neural proposal and ReSTIR
    reuse remain unsupported under spectral (wavefront-only reuse/proposal axes
    that spectral does not yet cover). Returns ``(ok, reason)`` with a specific
    reason per out-of-scope axis. Mirrors ``cli_common.reject_spectral_unsupported``.
    """
    if combo.integrator not in ("path", "bdpt", "sppm"):
        return False, f"spectral supports path/bdpt/sppm; {combo.integrator.upper()} unsupported"
    if combo.has_neural:
        return False, "spectral is incompatible with the neural proposal (v1)"
    if combo.has_reuse:
        return False, "spectral is incompatible with ReSTIR reuse (v1)"
    if scene.material_class != "flat":
        return False, "spectral is flat-material only (v1); no skin/subsurface/volume"
    # SPPM has no megakernel path (photon pass is wavefront-only).
    if combo.integrator == "sppm" and combo.execution_mode != "wavefront":
        return False, "SPPM is wavefront-only"
    return True, ""


def combo_is_valid(combo: RenderCombo, scene: SceneSpec) -> tuple[bool, str]:
    """Return ``(valid, reason)``. Mirrors the documented compatibility matrix.

    A skipped combo always carries an explicit reason; nothing is dropped
    silently.
    """
    if combo.integrator not in INTEGRATORS:
        return False, f"unknown integrator {combo.integrator!r}"
    if combo.execution_mode not in EXECUTION_MODES:
        return False, f"unknown execution_mode {combo.execution_mode!r}"
    # SPPM has no megakernel path.
    if combo.integrator == "sppm" and combo.execution_mode != "wavefront":
        return False, "SPPM is wavefront-only"
    # MLT (PSSMLT over BDPT, change mlt-integrator): wavefront-only, RGB-only,
    # layer-free, flat-material scenes only — the fixed primary-sample dimension
    # budget is only boundable without volumetric/subsurface transport, and the
    # wavefront non-flat path-fallback is not extended into Markov chains
    # (mixing estimators inside a chain). Gated on MLT_IMPLEMENTED (referenced
    # live, monkeypatchable) so combos are recorded "not yet wired" skips until
    # the transport lands — never silently rendered as another integrator.
    if combo.integrator == "mlt":
        if combo.execution_mode != "wavefront":
            return False, "MLT is wavefront-only"
        if combo.spectral:
            return False, "spectral MLT is outside the v1 envelope (RGB only)"
        if combo.has_neural or combo.has_reuse:
            return False, "MLT is layer-free (no neural proposal, no ReSTIR reuse)"
        if scene.material_class != "flat":
            return False, "MLT is flat-material only (no skin/subsurface/volume chains)"
        if not mlt_capability.MLT_IMPLEMENTED:
            return False, "MLT transport not yet wired — change mlt-integrator group 5"
    # Neural directional proposal: wavefront + path + flat material only.
    if combo.has_neural:
        if combo.execution_mode != "wavefront":
            return False, "neural proposal is wavefront-only"
        if combo.integrator != "path":
            return False, "neural proposal requires the path integrator (BDPT ignores it)"
        if scene.material_class != "flat":
            return False, "neural proposal is flat-material only"
    # ReSTIR DI direct-light reuse: wavefront + path only (it reuses the path
    # tracer's NEE reservoirs; BDPT/SPPM have their own light handling).
    if combo.has_reuse:
        if combo.execution_mode != "wavefront":
            return False, "ReSTIR DI reuse is wavefront-only"
        if combo.integrator != "path":
            return False, "ReSTIR DI reuse is exercised on the path integrator"
    # Heterogeneous participating media (nanovdb-volume-rendering): the volume
    # walk is wired into the Path integrator only. BDPT's connection strategies
    # and SPPM's photon pass have no medium transport (recorded exclusions,
    # follow-up changes), and the ReSTIR reuse axis is untested with media.
    if scene.material_class == "volume":
        if combo.integrator != "path":
            return False, f"{combo.integrator.upper()} has no volume transport (follow-up)"
        if combo.has_reuse:
            return False, "ReSTIR DI reuse untested with volume media (follow-up)"
    # Spectral render variant (changes spectral-rendering / spectral-wavefront).
    # The envelope is path/bdpt under either execution mode plus sppm under the
    # wavefront mode, over flat materials (no neural, no reuse); out-of-scope
    # combos take their specific envelope reason. An in-envelope combo is only
    # rendered once the transport is wired (SPECTRAL_IMPLEMENTED); until then it
    # is a recorded "not yet wired" skip so the sweep never renders it as RGB.
    # Mirrors reject_spectral_unsupported.
    if combo.spectral:
        ok, reason = spectral_envelope(combo, scene)
        if not ok:
            return False, reason
        if not spectral_capability.SPECTRAL_IMPLEMENTED:
            return False, "spectral transport not yet wired — megakernel Group 5 follow-up"
    # Heavy geometry that OOMs the megakernel.
    if combo.execution_mode == "megakernel" and not scene.megakernel_ok:
        return False, "geometry exceeds megakernel budget"
    return True, ""


def all_combos() -> list[RenderCombo]:
    """The full (unfiltered) combo space the matrix considers per scene."""
    combos: list[RenderCombo] = []
    for integ in INTEGRATORS:
        for mode in EXECUTION_MODES:
            combos.append(RenderCombo(integ, mode, (), "none"))
            # proposal axis (only meaningful additions are enumerated)
            for prop in PROPOSAL_AXES:
                combos.append(RenderCombo(integ, mode, (prop,), "none"))
            # reuse axis
            for reuse in REUSE_AXES:
                combos.append(RenderCombo(integ, mode, (), reuse))
            # spectral axis — the bare variant per integrator×mode; combo_is_valid
            # keeps (path, megakernel) and (bdpt, megakernel) on flat scenes.
            combos.append(RenderCombo(integ, mode, (), "none", spectral=True))
    return combos


def enumerate_combos(scene: SceneSpec) -> list[RenderCombo]:
    """Valid combos for *scene*, in deterministic order (anchor first)."""
    valid = [c for c in all_combos() if combo_is_valid(c, scene)[0]]
    valid.sort(key=lambda c: (c != ANCHOR, c.label))
    return valid


def combo_axis_class(combo: RenderCombo) -> str:
    """Which self-consistency tolerance class applies for *combo* vs its anchor.

    The comparison is against :func:`self_consistency_anchor` — the RGB anchor
    for RGB combos, the megakernel spectral path anchor for the spectral axis —
    so a spectral wavefront path is a ``"mode"`` delta against the spectral
    anchor (not conflated with the RGB→spectral shift).
    """
    if combo.has_neural or combo.has_reuse:
        return "unbiased"
    if combo.integrator == "sppm":
        return "sppm"
    if combo.integrator == "mlt":
        return "mlt"
    if combo.integrator != self_consistency_anchor(combo).integrator:
        return "integrator"
    return "mode"  # same integrator, differs only in execution mode


#: Default self-consistency tolerances (relMSE, FLIP) per axis class, sized to a
#: noise-limited equal-spp A/B. A scene may override via ``self_consistency``.
_DEFAULT_SELF_CONSISTENCY = {
    "mode": {"relmse": 0.02, "flip": 0.03},
    "integrator": {"relmse": 0.06, "flip": 0.06},
    "sppm": {"relmse": 0.15, "flip": 0.12},
    # MLT: unbiased in expectation but Markov-correlated — different per-pixel
    # noise structure at equal spp. Placeholder sized to the SPPM row; measured
    # harness-first at GPU validation (mlt-integrator task 6.2), tighten-only.
    "mlt": {"relmse": 0.15, "flip": 0.12},
    "unbiased": {"relmse": 0.05, "flip": 0.05},
}

#: Default self-consistency tolerances for the SPECTRAL axis (change
#: spectral-wavefront GPU-validation). Wider than the RGB defaults on the
#: sample-sharing classes because spectral wavefront is NOT bit-identical to the
#: megakernel: it threads the hero wavelengths through the staged records and so
#: draws a different sample sequence, giving a decorrelated-but-unbiased MC delta
#: (measured on Metal: ≈0 on smooth scenes, growing with variance to ~0.08 on a
#: caustic). ``mode`` 0.02→0.03, ``integrator`` 0.06→0.09; ``sppm`` unchanged
#: (already noise-limited). A scene overrides via ``spectral_self_consistency``.
#: This is a spectral-only floor — the RGB ``mode`` mega≡wave bit-identity gate
#: (0.02) is untouched.
_DEFAULT_SPECTRAL_SELF_CONSISTENCY = {
    "mode": {"relmse": 0.03, "flip": 0.03},
    "integrator": {"relmse": 0.09, "flip": 0.06},
    "sppm": {"relmse": 0.15, "flip": 0.12},
    # MLT: unbiased in expectation but Markov-correlated — different per-pixel
    # noise structure at equal spp. Placeholder sized to the SPPM row; measured
    # harness-first at GPU validation (mlt-integrator task 6.2), tighten-only.
    "mlt": {"relmse": 0.15, "flip": 0.12},
    "unbiased": {"relmse": 0.05, "flip": 0.05},
}


def self_consistency_tol(combo: RenderCombo, scene: SceneSpec) -> tuple[float, float]:
    """(relmse_tol, flip_tol) for *combo* measured against the anchor.

    Spectral combos consult :data:`_DEFAULT_SPECTRAL_SELF_CONSISTENCY` and the
    scene's ``spectral_self_consistency`` override; RGB combos keep the strict
    RGB table. The axis *class* is the same for both (see ``combo_axis_class``);
    only the tolerance floor differs.
    """
    cls = combo_axis_class(combo)
    if combo.spectral:
        table = dict(_DEFAULT_SPECTRAL_SELF_CONSISTENCY)
        override = scene.spectral_self_consistency
    else:
        table = dict(_DEFAULT_SELF_CONSISTENCY)
        override = scene.self_consistency
    if override:
        for k, v in override.items():
            table.setdefault(k, {})
            table[k] = {**table.get(k, {}), **v}
    t = table[cls]
    return float(t["relmse"]), float(t["flip"])


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


def pbrt_truth_result(spec: SceneSpec, combo: RenderCombo, img: np.ndarray,
                      ref: np.ndarray) -> ParityResult:
    """pbrt-truth gate for a rendered *img*, honouring a recorded baseline.

    The pbrt-truth assertion uses ``max(tol, baseline*(1+margin))`` when a
    baseline is recorded for this combo, and the caller logs the delta. Returns
    the full :class:`metrics.ImageMetrics` battery on the result.
    """
    m = metrics.compute_metrics(img, ref)
    rel_tol, flip_tol = spec.relmse_tol, spec.flip_tol
    baseline_used = False
    base = (spec.baselines or {}).get(combo.label)
    if base is not None:
        margin = 1.05
        rel_tol = max(rel_tol, float(base["relmse"]) * margin)
        flip_tol = max(flip_tol, float(base["flip"]) * margin)
        baseline_used = True
    passed = m.relmse <= rel_tol and m.flip <= flip_tol
    return ParityResult(spec.name, m.relmse, m.flip, passed,
                        metrics=m, combo=combo, baseline_used=baseline_used)


def absolute_radiance_result(spec: SceneSpec, combo: RenderCombo, img: np.ndarray,
                             ref: np.ndarray) -> ParityResult | None:
    """Absolute (un-exposure-aligned) radiance gate for *img* vs the pbrt *ref*.

    Runs only when ``spec.absolute`` is set. Unlike :func:`pbrt_truth_result`
    (which aligns exposure and so is blind to a global brightness offset) this
    measures the un-aligned mean-luminance ratio and the un-aligned relMSE, so a
    scene that drifts globally brighter/dimmer than pbrt fails even though its
    exposure-blind structure matches. A recorded per-combo ``baselines`` entry
    relaxes the gate to the known offset (harness-first), never tighter than the
    scene tolerance. Returns ``None`` when the scene opts out.

    The returned :class:`ParityResult` carries the un-aligned relMSE in ``relmse``
    and the mean-luminance ratio in ``flip`` (reused slot) so the matrix can log
    both without a new result type.
    """
    cfg = spec.absolute
    if not cfg:
        return None
    m = metrics.compute_metrics(img, ref, align=False)
    ratio = metrics.mean_ratio(img, ref)
    mean_tol = float(cfg.get("mean_ratio_tol", 0.1))
    rel_tol = float(cfg.get("relmse_tol", spec.relmse_tol))
    baseline_used = False
    base = (cfg.get("baselines") or {}).get(combo.label)
    if base is not None:
        margin = 1.05
        rel_tol = max(rel_tol, float(base["relmse"]) * margin)
        # Center the mean-ratio window on the recorded offset rather than 1.0.
        base_ratio = float(base["mean_ratio"])
        passed_ratio = abs(ratio - base_ratio) <= mean_tol * max(base_ratio, 1.0)
        baseline_used = True
    else:
        passed_ratio = abs(ratio - 1.0) <= mean_tol
    passed = passed_ratio and m.relmse <= rel_tol
    return ParityResult(spec.name, m.relmse, ratio, passed,
                        metrics=m, combo=combo, baseline_used=baseline_used)


def self_consistency_result(spec: SceneSpec, combo: RenderCombo, img: np.ndarray,
                            anchor_img: np.ndarray) -> ParityResult:
    """Self-consistency gate: *img* vs the anchor image at the per-axis tolerance.

    No baseline escape — these are correctness invariants.
    """
    m = metrics.compute_metrics(img, anchor_img)
    rel_tol, flip_tol = self_consistency_tol(combo, spec)
    passed = m.relmse <= rel_tol and m.flip <= flip_tol
    return ParityResult(spec.name, m.relmse, m.flip, passed, metrics=m, combo=combo)


def authoring_equivalence_result(spec: SceneSpec, plain_img: np.ndarray,
                                 mtlx_img: np.ndarray) -> ParityResult:
    """Authoring-equivalence gate: a MaterialX (_mtlx) variant's render must match
    its plain-USD sibling within the recorded tolerance (change
    confirming-test-scenes / render-parity-matrix delta).

    The two authorings drive different codegen paths (UsdPreviewSurface vs the
    MaterialX standard_surface/OpenPBR intake), so bit-equality is not expected;
    the tolerance is measured and pinned per scene in ``spec.equivalence``. No
    baseline escape — divergence here means the two authorings disagree, which is
    a real defect. Called with *spec* being the ``_mtlx`` variant (it carries the
    ``equivalence`` disposition).
    """
    cfg = spec.equivalence or {}
    m = metrics.compute_metrics(mtlx_img, plain_img)
    rel_tol = float(cfg.get("relmse", 0.02))
    flip_tol = float(cfg.get("flip", 0.03))
    passed = m.relmse <= rel_tol and m.flip <= flip_tol
    return ParityResult(spec.name, m.relmse, m.flip, passed, metrics=m)


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


def materialx_specs(specs: list[SceneSpec]) -> list[SceneSpec]:
    """Return a parallel scene-set that imports each *spec* through ``-mtlx``.

    Each returned spec shares the source ``.pbrt`` file, reference EXR, and
    tolerances of its UsdPreviewSurface sibling but flips ``materialx=True`` and
    suffixes its ``name`` with ``"_mtlx"`` (so the two sets coexist as distinct
    parametrize ids). The intent: a ``-mtlx`` render must match the same pbrt v4
    reference within the same tolerance — i.e. switching the export path is a
    no-op on the rendered image for the supported material subset.
    """
    out: list[SceneSpec] = []
    for s in specs:
        if s.usd:  # usd-source heavy scenes have no .pbrt to re-export via -mtlx
            continue
        fields = {k: getattr(s, k) for k in SceneSpec.__dataclass_fields__}
        fields["name"] = f"{s.name}_mtlx"
        fields["materialx"] = True
        out.append(SceneSpec(**fields))
    return out


def reference_exists(spec: SceneSpec, corpus_dir: str) -> bool:
    return os.path.isfile(os.path.join(corpus_dir, spec.ref))
