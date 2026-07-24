"""The render envelope: one statement of "does combo X run, and if not, why".

Which combinations of integrator × execution mode × directional proposals ×
reuse × spectral × material class × online-training actually run used to be
restated in three independent layers — the parity validity table
(:mod:`skinny.pbrt.parity`), the four CLI refusal guards
(:mod:`skinny.cli_common`), and the renderer's spectral scene-level gate — which
had to be edited in lockstep. This module is the single statement they all
consume (change ``unified-render-envelope-predicate``).

:func:`evaluate` returns **every** violated rule, in canonical rule order, as
``(code, reason)`` pairs — never just the first. A single-code verdict is
unimplementable: ``bdpt`` + neural under the megakernel violates both
"neural is wavefront-only" and "BDPT does not consume the neural proposal", and
the CLI must print the *second* while **accepting** ``path`` + neural under the
megakernel, which violates only the first (the renderer strips the neural bit at
runtime). So consumers select the codes they own:

* **parity** takes the first violation in canonical order — that reproduces its
  historical skip reasons and precedence exactly;
* **each CLI guard** scans for its own codes from :data:`CLI_GUARD_CODES`, in its
  own precedence (``reject_mlt_unsupported`` reports not-yet-wired *before* the
  megakernel refusal; parity reports them the other way round), and maps the code
  it finds to its own user-facing prose;
* **the renderer scene gate** scans for :data:`SPECTRAL_FLAT_ONLY` only, so it
  never newly refuses at runtime what today is a recorded parity skip.

Reason strings are verbatim the parity matrix's historical skip reasons, so
recorded skips and the matrix tests' substring assertions keep their wording. The
CLI's longer, flag-naming prose stays in ``cli_common`` keyed by code: prose is
presentation, not a rule.

Imports nothing above the two capability-flag leaf modules, so parity, the CLI
layer, and the renderer can all import it without a cycle. Both flags are read at
**evaluation** time (never captured at import), so a test monkeypatch takes
effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from skinny import mlt_capability, spectral_capability

#: Integrators the renderer knows. An integrator outside this set is refused as
#: unknown — the parity coverage meta-test fails the build when the app exposes
#: an integrator with no entry here.
INTEGRATORS = ("path", "bdpt", "sppm", "mlt")
EXECUTION_MODES = ("megakernel", "wavefront")

#: Integrators with no megakernel path: SPPM shares a global visible-point /
#: photon grid across pixels, MLT shares bootstrap/normalization state — both are
#: staged-wavefront constructs.
WAVEFRONT_ONLY_INTEGRATORS = ("sppm", "mlt")

#: Integrators that admit no sampling layer at all (no non-BSDF directional
#: proposal, no reuse pass): a layer inside a Markov-chain mutation would change
#: the target function MLT normalizes against.
LAYER_FREE_INTEGRATORS = ("mlt",)

#: Material classes the parity matrix distinguishes. ``flat`` is everything the
#: flat/standard_surface/OpenPBR BSDF shades.
FLAT_MATERIAL_CLASS = "flat"


# ─── reason codes ──────────────────────────────────────────────────────────
#
# One code per distinct reason *clause*, not per axis: "neural is wavefront-only"
# and "spectral is incompatible with the neural proposal" are two clauses of the
# neural axis and both are emitted when both apply, because consumers refuse on
# different ones.

UNKNOWN_INTEGRATOR = "unknown_integrator"
UNKNOWN_EXECUTION_MODE = "unknown_execution_mode"
SPPM_WAVEFRONT_ONLY = "sppm_wavefront_only"
MLT_WAVEFRONT_ONLY = "mlt_wavefront_only"
MLT_NO_PROPOSALS = "mlt_no_proposals"
MLT_NO_REUSE = "mlt_no_reuse"
MLT_FLAT_ONLY = "mlt_flat_only"
MLT_NOT_WIRED = "mlt_not_wired"
NEURAL_WAVEFRONT_ONLY = "neural_wavefront_only"
NEURAL_PATH_ONLY = "neural_path_only"
NEURAL_FLAT_ONLY = "neural_flat_only"
ENV_PATH_ONLY = "env_path_only"
ENV_FLAT_ONLY = "env_flat_only"
REUSE_WAVEFRONT_ONLY = "reuse_wavefront_only"
REUSE_PATH_ONLY = "reuse_path_only"
VOLUME_INTEGRATOR_PATH_ONLY = "volume_integrator_path_only"
VOLUME_NO_REUSE = "volume_no_reuse"
ONLINE_TRAINING_PATH_ONLY = "online_training_path_only"
SPECTRAL_INTEGRATOR_UNSUPPORTED = "spectral_integrator_unsupported"
SPECTRAL_NO_NEURAL = "spectral_no_neural"
SPECTRAL_ENV_PATH_ONLY = "spectral_env_path_only"
SPECTRAL_NO_REUSE = "spectral_no_reuse"
SPECTRAL_FLAT_ONLY = "spectral_flat_only"
SPECTRAL_NOT_WIRED = "spectral_not_wired"
MEGAKERNEL_BUDGET = "megakernel_budget"

_WAVEFRONT_ONLY_CODE = {"sppm": SPPM_WAVEFRONT_ONLY, "mlt": MLT_WAVEFRONT_ONLY}

#: Every code the predicate can emit, in canonical rule order. Each one is owned
#: by a CLI guard (:data:`CLI_GUARD_CODES`), by the renderer scene gate
#: (:data:`RENDERER_SCENE_CODES`), or by nobody on purpose
#: (:data:`CLI_UNOWNED_CODES`) — a test asserts the partition.
ALL_CODES = (
    UNKNOWN_INTEGRATOR, UNKNOWN_EXECUTION_MODE, SPPM_WAVEFRONT_ONLY,
    MLT_WAVEFRONT_ONLY, MLT_NO_PROPOSALS, MLT_NO_REUSE, MLT_FLAT_ONLY,
    MLT_NOT_WIRED, NEURAL_WAVEFRONT_ONLY, NEURAL_PATH_ONLY, NEURAL_FLAT_ONLY,
    ENV_PATH_ONLY, ENV_FLAT_ONLY, REUSE_WAVEFRONT_ONLY, REUSE_PATH_ONLY,
    VOLUME_INTEGRATOR_PATH_ONLY, VOLUME_NO_REUSE, ONLINE_TRAINING_PATH_ONLY,
    SPECTRAL_INTEGRATOR_UNSUPPORTED, SPECTRAL_NO_NEURAL, SPECTRAL_ENV_PATH_ONLY,
    SPECTRAL_NO_REUSE, SPECTRAL_FLAT_ONLY, SPECTRAL_NOT_WIRED, MEGAKERNEL_BUDGET,
)


@dataclass(frozen=True)
class EnvelopeQuery:
    """One point in the envelope's query space.

    *proposals* holds the directional-proposal tokens **beyond** the material's
    own BSDF sampler (``("env",)``, ``("neural",)``, …) — ``bsdf`` is implicit
    and never listed. *reuse* is ``"none"`` or a reuse-pass token. *material_class*
    defaults to ``"flat"`` so the flag-level consumers (which cannot see the
    scene) get flag-level answers only. *megakernel_ok* is False for geometry too
    heavy for the fused megakernel dispatch.
    """

    integrator: str = "path"
    execution_mode: str = "wavefront"
    proposals: tuple[str, ...] = ()
    reuse: str = "none"
    spectral: bool = False
    material_class: str = FLAT_MATERIAL_CLASS
    online_training: bool = False
    megakernel_ok: bool = True

    @property
    def has_neural(self) -> bool:
        return "neural" in self.proposals

    @property
    def has_env_proposal(self) -> bool:
        return "env" in self.proposals

    @property
    def has_reuse(self) -> bool:
        return bool(self.reuse) and self.reuse != "none"

    @property
    def is_flat(self) -> bool:
        return self.material_class == FLAT_MATERIAL_CLASS


@dataclass(frozen=True)
class Verdict:
    """Every rule *query* violates, in canonical rule order."""

    violations: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(code for code, _ in self.violations)

    def first_code(self, *codes: str) -> str | None:
        """The first of *codes* this verdict violates, scanned in the order the
        **caller** lists them — its precedence, not the predicate's. ``None`` when
        the verdict violates none of them. This is how a consumer selects the
        codes it owns without acting on rules it does not."""
        violated = set(self.codes)
        for code in codes:
            if code in violated:
                return code
        return None

    def reason_for(self, *codes: str) -> str | None:
        """The reason string of :meth:`first_code`, or ``None``."""
        code = self.first_code(*codes)
        return None if code is None else dict(self.violations)[code]

    def first(self) -> tuple[bool, str]:
        """``(valid, reason)`` in the predicate's canonical order — the parity
        matrix's historical contract."""
        if not self.violations:
            return True, ""
        return False, self.violations[0][1]


def evaluate(query: EnvelopeQuery) -> Verdict:
    """Every envelope rule *query* violates, in canonical rule order.

    The order below **is** the parity matrix's historical precedence, so
    ``Verdict.first()`` reproduces its skip reasons verbatim. Rules never
    short-circuit: a consumer that needs a later clause (the CLI's bdpt-neural
    prose, the renderer's flat-material clause) must still see it.
    """
    v: list[tuple[str, str]] = []
    integ = query.integrator

    if integ not in INTEGRATORS:
        v.append((UNKNOWN_INTEGRATOR, f"unknown integrator {integ!r}"))
    if query.execution_mode not in EXECUTION_MODES:
        v.append((UNKNOWN_EXECUTION_MODE,
                  f"unknown execution_mode {query.execution_mode!r}"))
    wavefront = query.execution_mode == "wavefront"

    # SPPM has no megakernel path (global photon grid); nor does MLT (below).
    if integ in WAVEFRONT_ONLY_INTEGRATORS and not wavefront:
        v.append((_WAVEFRONT_ONLY_CODE[integ], f"{integ.upper()} is wavefront-only"))

    # MLT (PSSMLT over BDPT, changes mlt-integrator / spectral-mlt): wavefront-only,
    # RGB or spectral, layer-free, flat-material scenes only — the fixed
    # primary-sample dimension budget is only boundable without
    # volumetric/subsurface transport, and the wavefront non-flat path-fallback is
    # not extended into Markov chains (mixing estimators inside a chain). Gated on
    # MLT_IMPLEMENTED so combos are recorded "not yet wired" skips until the
    # transport lands — never silently rendered as another integrator.
    if integ in LAYER_FREE_INTEGRATORS:
        layer_free = (f"{integ.upper()} is layer-free "
                      "(no directional proposal, no ReSTIR reuse)")
        if query.proposals:
            v.append((MLT_NO_PROPOSALS, layer_free))
        if query.has_reuse:
            v.append((MLT_NO_REUSE, layer_free))
        if not query.is_flat:
            v.append((MLT_FLAT_ONLY,
                      f"{integ.upper()} is flat-material only "
                      "(no skin/subsurface/volume chains)"))
        if not mlt_capability.MLT_IMPLEMENTED:
            v.append((MLT_NOT_WIRED,
                      "MLT transport not yet wired — change mlt-integrator group 5"))

    # Neural directional proposal: wavefront + path + flat material only.
    if query.has_neural:
        if not wavefront:
            v.append((NEURAL_WAVEFRONT_ONLY, "neural proposal is wavefront-only"))
        if integ != "path":
            v.append((NEURAL_PATH_ONLY,
                      "neural proposal requires the path integrator (BDPT ignores it)"))
        if not query.is_flat:
            v.append((NEURAL_FLAT_ONLY, "neural proposal is flat-material only"))
    # Analytic environment directional proposal: consumed by the path integrator's
    # bounce seam in either execution mode, on flat materials. BDPT/SPPM/MLT keep
    # their native sampling and must not advertise the axis.
    if query.has_env_proposal:
        if integ != "path":
            v.append((ENV_PATH_ONLY, "environment proposal requires the path integrator"))
        if not query.is_flat:
            v.append((ENV_FLAT_ONLY, "environment proposal is flat-material only"))
    # ReSTIR DI direct-light reuse: wavefront + path only (it reuses the path
    # tracer's NEE reservoirs; BDPT/SPPM have their own light handling).
    if query.has_reuse:
        if not wavefront:
            v.append((REUSE_WAVEFRONT_ONLY, "ReSTIR DI reuse is wavefront-only"))
        if integ != "path":
            v.append((REUSE_PATH_ONLY, "ReSTIR DI reuse is exercised on the path integrator"))
    # Heterogeneous participating media (nanovdb-volume-rendering): the volume walk
    # is wired into the Path integrator only. BDPT's connection strategies and
    # SPPM's photon pass have no medium transport (recorded exclusions, follow-up
    # changes), and the ReSTIR reuse axis is untested with media.
    if query.material_class == "volume":
        if integ != "path":
            v.append((VOLUME_INTEGRATOR_PATH_ONLY,
                      f"{integ.upper()} has no volume transport (follow-up)"))
        if query.has_reuse:
            v.append((VOLUME_NO_REUSE, "ReSTIR DI reuse untested with volume media (follow-up)"))
    # Online training exists to improve the neural directional proposal, which only
    # the path integrator consumes. A flag-level axis: the parity matrix never
    # enables it, so this clause is CLI-only in practice.
    if query.online_training and integ != "path":
        v.append((ONLINE_TRAINING_PATH_ONLY,
                  f"online training requires the path integrator ({integ.upper()} "
                  "does not consume the neural proposal)"))

    # Spectral render variant (changes spectral-rendering / spectral-wavefront /
    # spectral-mlt). The envelope is path/bdpt under either execution mode plus
    # sppm/mlt under wavefront, over flat materials (no neural, no reuse;
    # environment proposal on path only). An in-envelope combo is only rendered
    # once the transport is wired (SPECTRAL_IMPLEMENTED); until then it is a
    # recorded "not yet wired" skip so the sweep never renders it as RGB.
    if query.spectral:
        if integ not in INTEGRATORS:
            v.append((SPECTRAL_INTEGRATOR_UNSUPPORTED,
                      f"spectral supports path/bdpt/sppm/mlt; {integ.upper()} unsupported"))
        # Any proposal layer other than the analytic environment one: the env
        # proposal shares the wavelength-independent direction/pdf seam with RGB,
        # while stateful neural inference stays outside the spectral envelope.
        # Stated as "not env" rather than "is neural" so an unvalidated
        # SKINNY_PROPOSALS token is refused rather than silently admitted.
        if any(p != "env" for p in query.proposals):
            v.append((SPECTRAL_NO_NEURAL,
                      "spectral is incompatible with the neural proposal (v1)"))
        if query.has_env_proposal and integ != "path":
            v.append((SPECTRAL_ENV_PATH_ONLY,
                      "spectral environment proposal requires the path integrator"))
        if query.has_reuse:
            v.append((SPECTRAL_NO_REUSE, "spectral is incompatible with ReSTIR reuse (v1)"))
        if not query.is_flat:
            v.append((SPECTRAL_FLAT_ONLY,
                      "spectral is flat-material only (v1); no skin/subsurface/volume"))
        if not spectral_capability.SPECTRAL_IMPLEMENTED:
            v.append((SPECTRAL_NOT_WIRED,
                      "spectral transport not yet wired — megakernel Group 5 follow-up"))

    # Heavy geometry that OOMs the megakernel.
    if query.execution_mode == "megakernel" and not query.megakernel_ok:
        v.append((MEGAKERNEL_BUDGET, "geometry exceeds megakernel budget"))
    return Verdict(tuple(v))


#: The spectral-envelope view (``parity.spectral_envelope``): the *intended*
#: spectral scope, independent of whether the transport is wired. Listed in that
#: function's own historical precedence, which differs from the canonical order —
#: it reports a spectral-axis clause before the integrator's execution-mode clause.
SPECTRAL_ENVELOPE_CODES = (
    SPECTRAL_INTEGRATOR_UNSUPPORTED,
    SPECTRAL_NO_NEURAL,
    SPECTRAL_ENV_PATH_ONLY,
    SPECTRAL_NO_REUSE,
    SPECTRAL_FLAT_ONLY,
    SPPM_WAVEFRONT_ONLY,
    MLT_WAVEFRONT_ONLY,
)

#: Which verdict codes each CLI guard enforces, in that guard's own precedence.
#: A guard owns code *selection* and code→prose, never a rule. Codes absent from
#: every tuple are deliberately **not** refused at the CLI — see
#: :data:`CLI_UNOWNED_CODES`.
CLI_GUARD_CODES = {
    "reject_sppm_without_wavefront": (SPPM_WAVEFRONT_ONLY,),
    # Not-yet-wired before the megakernel refusal — the opposite of the canonical
    # order, because an unimplemented integrator is the more useful thing to say.
    "reject_mlt_unsupported": (
        MLT_NOT_WIRED, MLT_WAVEFRONT_ONLY, MLT_NO_PROPOSALS, MLT_NO_REUSE,
        ONLINE_TRAINING_PATH_ONLY,
    ),
    "reject_spectral_unsupported": (
        SPECTRAL_NO_NEURAL, SPECTRAL_ENV_PATH_ONLY, SPECTRAL_NO_REUSE,
        SPECTRAL_NOT_WIRED,
    ),
    # The bdpt branch only: the sppm/mlt integrators are delegated to their own
    # guards above (and return), so this never fires for them.
    "validate_render_flags": (ONLINE_TRAINING_PATH_ONLY, NEURAL_PATH_ONLY),
}

#: The renderer's scene-level gate owns exactly one code — the material set is
#: invisible to the CLI, so this is the one envelope fact checked at pack time.
RENDERER_SCENE_CODES = (SPECTRAL_FLAT_ONLY,)

#: Codes no CLI guard refuses, each for a recorded reason — this one-sided
#: acceptance is data, not an oversight:
#:
#: * ``neural_wavefront_only`` / ``neural_flat_only`` — the renderer strips the
#:   neural bit at runtime under the megakernel (and on non-flat materials), so
#:   ``--integrator path --execution-mode megakernel --proposals bsdf,neural`` is
#:   accepted and silently falls back to the analytic subset.
#: * ``env_*`` / ``reuse_*`` / ``volume_*`` / ``mlt_flat_only`` — matrix-only
#:   exclusions: the CLI cannot see the scene, and the renderer degrades rather
#:   than refusing.
#: * ``spectral_flat_only`` — owned by the renderer scene gate instead.
#: * ``spectral_integrator_unsupported`` / ``unknown_*`` — unreachable through
#:   argparse ``choices``.
#: * ``megakernel_budget`` — a per-scene harness fact, not a flag.
CLI_UNOWNED_CODES = (
    UNKNOWN_INTEGRATOR, UNKNOWN_EXECUTION_MODE, MLT_FLAT_ONLY,
    NEURAL_WAVEFRONT_ONLY, NEURAL_FLAT_ONLY, ENV_PATH_ONLY, ENV_FLAT_ONLY,
    REUSE_WAVEFRONT_ONLY, REUSE_PATH_ONLY, VOLUME_INTEGRATOR_PATH_ONLY,
    VOLUME_NO_REUSE, SPECTRAL_INTEGRATOR_UNSUPPORTED, SPECTRAL_FLAT_ONLY,
    MEGAKERNEL_BUDGET,
)


#: Packed material-type code (the renderer's `materialTypes` vocabulary) → the
#: query's material class. **Pinned, not derived**: `skin`=0, `subsurface`=4 and
#: `volume`=5 are the parity matrix's own non-flat classes, while `debug`=2 and
#: `python`=3 are *not* parity classes — they map to a non-flat class solely so
#: the renderer's spectral scene gate keeps refusing them, exactly as it does
#: today. `flat`=1 is the only flat code; anything unmapped is treated as
#: non-flat (every non-flat class trips the same rule).
MATERIAL_CLASS_FOR_TYPE_CODE = {0: "skin", 1: FLAT_MATERIAL_CLASS, 2: "subsurface",
                                3: "subsurface", 4: "subsurface", 5: "volume"}


def spectral_refuses_material_types(type_codes) -> bool:
    """Whether ``--spectral`` is refused on a scene carrying these packed
    material-type codes — the renderer's scene-level gate (design D4).

    The rule ("spectral is flat-material only in v1") is an ordinary envelope rule
    evaluated by :func:`evaluate`; only the flat-material code is scanned, so this
    gate can never newly refuse another rule at runtime, where today's behaviour is
    a recorded parity skip or a runtime fallback rather than a refusal. Lives here
    rather than in the renderer so it stays importable — and testable — without a
    GPU.
    """
    return any(
        evaluate(EnvelopeQuery(
            spectral=True,
            material_class=MATERIAL_CLASS_FOR_TYPE_CODE.get(int(t), "subsurface"),
        )).first_code(*RENDERER_SCENE_CODES) is not None
        for t in type_codes
    )


def parse_proposals(proposals: str | None) -> tuple[str, ...]:
    """The non-BSDF proposal tokens of a ``--proposals`` flag value.

    ``None``/``""``/``"bsdf"`` → ``()``; ``"bsdf,neural"`` → ``("neural",)``. The
    implicit ``bsdf`` term is dropped so a query's ``proposals`` always means
    "layers beyond the material's own sampler".
    """
    return tuple(
        p.strip() for p in (proposals or "").split(",")
        if p.strip() and p.strip() != "bsdf"
    )
