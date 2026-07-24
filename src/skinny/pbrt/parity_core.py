"""Pure, hostless core of the parity harness: the scene manifest schema, the
combo matrix + validity oracle, the self-consistency tolerance tables, and the
result builders.

Nothing here touches a GPU, a renderer, or USD (``pxr``) — imports are limited to
stdlib, numpy, the capability flags, and :mod:`skinny.metrics`/
:mod:`skinny.render_envelope`, so the matrix logic can be exercised on any host.
The GPU render adapter (``render_linear``/``render_combo``/``evaluate``) lives in
:mod:`skinny.pbrt.parity`, which also re-exports everything below so the
historical ``skinny.pbrt.parity`` surface is unchanged.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from skinny import render_envelope

from . import metrics


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
# A combo is a point in (integrator × execution_mode × proposals × reuse ×
# spectral). Which of them actually run is NOT restated here: ``combo_is_valid``
# delegates to :mod:`skinny.render_envelope`, the single statement of the render
# envelope shared with the CLI refusal guards and the renderer's spectral scene
# gate. The CLAUDE.md / README compatibility tables document that predicate.

INTEGRATORS = render_envelope.INTEGRATORS
EXECUTION_MODES = render_envelope.EXECUTION_MODES
# Proposal/reuse axes exercised by the matrix (beyond the bare baseline).
PROPOSAL_AXES = ("env", "neural")
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
    def has_env_proposal(self) -> bool:
        return "env" in self.proposals

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


def _query(combo: RenderCombo, scene: SceneSpec, *, spectral: bool | None = None):
    """The :mod:`skinny.render_envelope` query for *combo* rendered on *scene*.

    The matrix never enables online training, so that axis stays at its default.
    """
    return render_envelope.EnvelopeQuery(
        integrator=combo.integrator,
        execution_mode=combo.execution_mode,
        proposals=tuple(combo.proposals),
        reuse=combo.reuse,
        spectral=combo.spectral if spectral is None else spectral,
        material_class=scene.material_class,
        megakernel_ok=scene.megakernel_ok,
    )


def spectral_envelope(combo: RenderCombo, scene: SceneSpec) -> tuple[bool, str]:
    """The intended spectral validity envelope, independent of whether the
    transport is wired yet (:data:`SPECTRAL_IMPLEMENTED`).

    v1 (megakernel) admitted path/bdpt under the megakernel; the
    ``spectral-wavefront`` change extends the envelope to the **wavefront**
    execution mode too, for path, bdpt and sppm; ``spectral-mlt`` adds MLT
    under wavefront (flat materials, no reuse).
    The analytic environment proposal is admitted on spectral path only; BDPT
    SPPM, and MLT retain native BSDF sampling. SPPM and MLT have no megakernel
    path (photon / Markov-chain passes are wavefront-only), so both are refused
    under the megakernel; the neural proposal and ReSTIR reuse remain
    unsupported under spectral. Returns ``(ok, reason)`` with a specific reason
    per out-of-scope axis.

    A thin view over :func:`skinny.render_envelope.evaluate` — this function owns
    only which verdict codes make up the *spectral scope* (and their precedence,
    which is spectral-axis-first, unlike the canonical order), never a rule.
    Spectral is forced on: the question is what the envelope *would* admit.
    """
    verdict = render_envelope.evaluate(_query(combo, scene, spectral=True))
    reason = verdict.reason_for(*render_envelope.SPECTRAL_ENVELOPE_CODES)
    return (reason is None), (reason or "")


def combo_is_valid(combo: RenderCombo, scene: SceneSpec) -> tuple[bool, str]:
    """Return ``(valid, reason)`` from the shared render-envelope predicate.

    Every rule lives in :mod:`skinny.render_envelope` — the same statement the
    CLI refusal guards and the renderer's spectral scene gate consume, so a combo
    this table renders can never be one the CLI refuses. The matrix takes the
    **first** violation in the predicate's canonical order, which is this
    function's historical precedence.

    A skipped combo always carries an explicit reason; nothing is dropped
    silently.
    """
    return render_envelope.evaluate(_query(combo, scene)).first()


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
            # keeps the valid transport/proposal envelope on flat scenes.
            combos.append(RenderCombo(integ, mode, (), "none", spectral=True))
            for prop in PROPOSAL_AXES:
                combos.append(RenderCombo(
                    integ, mode, (prop,), "none", spectral=True,
                ))
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
    if combo.proposals or combo.has_reuse:
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

#: The SPECTRAL axis widens exactly two rows over the RGB table (change
#: spectral-wavefront GPU-validation), because spectral wavefront is NOT
#: bit-identical to the megakernel: it threads the hero wavelengths through the
#: staged records and so draws a different sample sequence, giving a
#: decorrelated-but-unbiased MC delta (measured on Metal: ≈0 on smooth scenes,
#: growing with variance to ~0.08 on a caustic). Every other class — and every
#: ``flip`` value — is inherited, so a new tolerance class is written once.
_SPECTRAL_TOL_OVERLAY = {
    "mode": {"relmse": 0.03},
    "integrator": {"relmse": 0.09},
}

#: Default self-consistency tolerances for the SPECTRAL axis, derived from the RGB
#: table by :data:`_SPECTRAL_TOL_OVERLAY`. A scene overrides via
#: ``spectral_self_consistency``. Spectral-only floors — the RGB ``mode``
#: mega≡wave bit-identity gate (0.02) is untouched.
_DEFAULT_SPECTRAL_SELF_CONSISTENCY = {
    cls: {**tol, **_SPECTRAL_TOL_OVERLAY.get(cls, {})}
    for cls, tol in _DEFAULT_SELF_CONSISTENCY.items()
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
