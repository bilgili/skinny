"""Shared command-line surface for the render-selection flags.

The three render-selection axes — integrator (`path`/`bdpt`), execution mode
(`megakernel`/`wavefront`), and the wavefront-bdpt subpath-build walk
(`fused`/`eye`/`eye_light`) — are defined here once and consumed identically by
every front-end (``skinny``, ``skinny-gui``, ``skinny-web``, ``skinny-render``)
so they cannot drift apart.

``megakernel`` is intentionally *not* a walk value: it names the execution mode
(the monolithic ``main_pass`` dispatch). The wavefront single-kernel subpath
build is named ``fused``. The old ``megakernel`` walk name is still accepted as
a deprecated alias resolving to ``fused`` (see :func:`resolve_walk`).
"""

from __future__ import annotations

import argparse
import os

from skinny import spectral_capability

# path → integrator_index, mirroring the renderer's integrator ordering.
INTEGRATOR_INDEX = {"path": 0, "bdpt": 1, "sppm": 2}

# Execution mode `auto` derives from the integrator: `path`/`bdpt` run under the
# megakernel; `sppm` has no megakernel path and runs under the wavefront backend.
# An explicit `--execution-mode` (flag or env) overrides this (see
# :func:`resolve_execution_mode`).
DEFAULT_EXECUTION_FOR_INTEGRATOR = {
    "path": "megakernel",
    "bdpt": "megakernel",
    "sppm": "wavefront",
}

# Advertised walk choices. `megakernel` is accepted as a deprecated alias but is
# not listed, so only the execution axis owns that word.
WALK_CHOICES = ("fused", "eye", "eye_light")
_WALK_ALIASES = {"megakernel": "fused"}


def _env_flag(name: str) -> bool:
    """Truthy parse of a boolean env var for a ``store_true`` flag default —
    '1'/'true'/'yes'/'on' (case-insensitive) enable it; unset/empty is False."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    """Parse a float env var for a numeric flag default — unset/empty → ``None``
    (the sentinel meaning "use the renderer's built-in default"). A malformed
    value raises a clear, user-facing error naming the env var rather than a raw
    ValueError at parser-construction time."""
    v = os.environ.get(name)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        raise SystemExit(f"invalid {name}={v!r}: expected a float")


def _env_int(name: str) -> int | None:
    """Parse an integer env var for a numeric flag default — unset/empty →
    ``None`` (the sentinel meaning "use the built-in default"). A malformed value
    raises a clear, user-facing error naming the env var rather than a raw
    ValueError at parser-construction time."""
    v = os.environ.get(name)
    if v in (None, ""):
        return None
    try:
        return int(v)
    except ValueError:
        raise SystemExit(f"invalid {name}={v!r}: expected an integer")


def validate_render_flags(args) -> None:
    """Reject render-flag combinations that cannot work, with a clear error +
    exit (change bdpt-neural-incompatibility).

    The neural directional proposal — and the online training that exists to
    improve it — is consumed only by the **path** integrator (``path.slang``
    imports ``sampling.proposal``). BDPT samples directions with native BSDF
    sampling on every backend and execution mode (``bdpt.slang`` /
    ``wavefront_bdpt.slang`` never import the proposal seam), so a neural
    proposal or ``--online-training`` under ``--integrator bdpt`` would train /
    select weights the bdpt render never reads — and the online-training record
    drain has no wavefront source for bdpt (it would crash mid-frame on Metal).
    Refuse up front rather than silently no-op or crash.

    Raises ``SystemExit`` (a usage error) on the incompatible combo. Only an
    explicit ``--integrator bdpt`` trips the bdpt×neural guard; ``integrator=None``
    (the persisted/default path) does not. SPPM has no megakernel path; the
    execution mode defaults to ``auto`` which derives ``wavefront`` for ``sppm``
    (see :func:`resolve_execution_mode`, run before this validator), so this guard
    trips **only** when the user explicitly forced ``--execution-mode megakernel``
    — the fix is to drop that flag or use ``--execution-mode wavefront``.
    Tolerant of a ``Namespace`` without a ``proposals`` attribute (the GUI/web
    front-ends suppress ``--proposals``).

    Also rejects a non-positive render-area ``--width``/``--height`` (flag or env
    fallback) with a clear usage error, before any GPU init. Front-ends that do
    not expose the shared resolution flags simply lack the attributes and are
    skipped."""
    for dim in ("width", "height"):
        val = getattr(args, dim, None)
        if val is not None and val <= 0:
            raise SystemExit(
                f"skinny: --{dim} must be a positive integer (got {val})."
            )
    integrator = getattr(args, "integrator", None)
    if integrator == "sppm":
        # SPPM is wavefront-only (no global photon map under the megakernel). The
        # execution mode is resolved before this validator, so plain `--integrator
        # sppm` already derived `wavefront` (auto) — this only trips when the user
        # explicitly forced `--execution-mode megakernel`. The interactive
        # front-ends additionally re-check the *persisted* sppm case (which this
        # CLI-integrator-keyed guard cannot see) via reject_sppm_without_wavefront.
        reject_sppm_without_wavefront(integrator, getattr(args, "execution_mode", "megakernel"))
        return
    if integrator != "bdpt":
        return
    proposals = getattr(args, "proposals", None) or ""
    online = bool(getattr(args, "online_training", False))
    neural = "neural" in proposals
    if not (neural or online):
        return
    what = ("--online-training" if online
            else "the neural proposal (--proposals …,neural)")
    raise SystemExit(
        f"skinny: {what} is incompatible with --integrator bdpt — BDPT does not "
        "consume the neural directional proposal (it samples directions with "
        "native BSDF sampling, on every backend and execution mode). Use "
        "--integrator path for neural guiding / online training."
    )


def resolve_walk(value: str) -> str:
    """Normalize a bdpt-walk value, mapping the deprecated ``megakernel`` alias
    to ``fused``. Raises ``ValueError`` on a genuinely unknown value."""
    v = str(value).strip().lower()
    v = _WALK_ALIASES.get(v, v)
    if v not in WALK_CHOICES:
        raise ValueError(
            f"unknown bdpt walk {value!r} "
            f"(expected one of {WALK_CHOICES} or the deprecated alias 'megakernel')"
        )
    return v


def resolve_execution_mode(execution_mode: str | None, integrator: str | None) -> str:
    """Resolve the concrete execution mode (`megakernel` / `wavefront`).

    ``auto`` (the default) — meaning neither ``--execution-mode`` nor
    ``SKINNY_EXECUTION_MODE`` pinned a concrete mode — derives the mode from the
    startup ``integrator`` via :data:`DEFAULT_EXECUTION_FOR_INTEGRATOR`
    (``path``/``bdpt`` → ``megakernel``, ``sppm`` → ``wavefront``). An explicit
    ``megakernel``/``wavefront`` wins. Precedence: explicit mode > integrator
    default. Called once at startup, before ``validate_render_flags`` and before
    the renderer is constructed; the result is fixed for the session. A ``None``
    integrator (the persisted/default path) derives ``megakernel``."""
    if execution_mode and execution_mode != "auto":
        return execution_mode
    return DEFAULT_EXECUTION_FOR_INTEGRATOR.get(integrator or "path", "megakernel")


def startup_integrator_name(cli_integrator: str | None, persisted_index=None) -> str:
    """Resolve the integrator name active at launch on the interactive front-ends,
    for deriving the ``auto`` execution mode: an explicit ``--integrator`` wins,
    else the persisted ``integrator_index`` (mapped back through
    :data:`INTEGRATOR_INDEX`), else ``'path'``. A missing/malformed persisted
    index falls back to ``'path'``."""
    if cli_integrator is not None:
        return cli_integrator
    try:
        return {v: k for k, v in INTEGRATOR_INDEX.items()}.get(int(persisted_index), "path")
    except (TypeError, ValueError):
        return "path"


def reject_sppm_without_wavefront(integrator: str | None, execution_mode: str | None) -> None:
    """Refuse an ``sppm`` integrator under the megakernel — SPPM has no megakernel
    path. Checks the **effective** startup integrator against the **resolved**
    execution mode, so it catches both the explicit ``--integrator sppm`` case
    (via :func:`validate_render_flags`) and the interactive case where ``sppm``
    comes from the persisted setting while ``--execution-mode megakernel`` was
    explicitly forced — which ``validate_render_flags`` (keyed on the CLI
    ``--integrator``) cannot see. Raises ``SystemExit``; a no-op otherwise."""
    if integrator == "sppm" and (execution_mode or "megakernel") == "megakernel":
        raise SystemExit(
            "skinny: --integrator sppm has no megakernel path — SPPM "
            "(Stochastic Progressive Photon Mapping) shares a global "
            "visible-point / photon-grid structure across pixels. Drop the "
            "explicit --execution-mode megakernel (sppm auto-selects "
            "wavefront) or pass --execution-mode wavefront."
        )


def reject_spectral_unsupported(
    spectral: bool,
    integrator: str | None,
    execution_mode: str | None,
    proposals: str | None = None,
    reuse: str | None = None,
) -> None:
    """Refuse ``--spectral`` outside its v1 envelope. No-op when not spectral.

    Spectral mode is a compile-time variant chosen at startup (path or bdpt
    integrator, megakernel execution, flat materials). Checked against the
    **resolved** execution mode after :func:`resolve_execution_mode`, so it
    catches an explicit ``--execution-mode wavefront`` while ``auto`` (which
    derives ``megakernel`` for ``path``/``bdpt``) is allowed. SPPM has no
    megakernel path, and the neural proposal and ReSTIR reuse layers are
    wavefront-only, so they are refused. Raises ``SystemExit``.

    Scene-level unsupported transport (a skin/subsurface or heterogeneous-volume
    scene) is refused later, at renderer setup, where the material set is known —
    this CLI guard covers only the flag-level combinations.
    """
    if not spectral:
        return
    integ = integrator or "path"
    if integ not in ("path", "bdpt"):
        raise SystemExit(
            f"skinny: --spectral supports --integrator path or bdpt in the "
            f"megakernel (got {integ}). SPPM has no megakernel path (its photon "
            "pass is wavefront-only), so spectral SPPM awaits the spectral "
            "wavefront follow-up."
        )
    if (execution_mode or "megakernel") == "wavefront":
        raise SystemExit(
            "skinny: --spectral runs only under the megakernel execution mode in "
            "v1 — drop --execution-mode wavefront (path/bdpt default to "
            "megakernel). Spectral wavefront is the designated follow-up."
        )
    # Only the native BSDF proposal is supported in v1. A non-BSDF proposal
    # (environment-importance or neural) draws the bounce direction from a
    # mixture the megakernel spectral path does not sample — it uses the material's
    # native sample() while NEE's MIS companion assumes the mixture pdf, biasing
    # direct+indirect coupling. Refuse the whole non-BSDF set (neural is also
    # wavefront-only); mixture sampling under spectral is a follow-up.
    extra_proposals = [
        p.strip() for p in (proposals or "").split(",") if p.strip() and p.strip() != "bsdf"
    ]
    if extra_proposals:
        raise SystemExit(
            f"skinny: --spectral supports only the BSDF proposal in v1 (got "
            f"--proposals {proposals}). The environment/neural directional proposals "
            "draw the bounce from a mixture the megakernel spectral path does not "
            "sample, which biases MIS — they are a spectral follow-up."
        )
    if reuse and reuse not in ("none",):
        raise SystemExit(
            f"skinny: --spectral is incompatible with --reuse {reuse} — reservoir "
            "reuse is wavefront-only and spectral is megakernel-only in v1."
        )
    # The envelope is satisfied, but the megakernel spectral transport (Group 5)
    # is not wired yet — the renderer would silently produce an RGB frame. Refuse
    # rather than mislead. Gated by the shared capability flag (flip it with the
    # transport to enable). Referenced live so a test monkeypatch takes effect.
    if not spectral_capability.SPECTRAL_IMPLEMENTED:
        raise SystemExit(
            "skinny: --spectral is not yet implemented — the hero-wavelength "
            "megakernel transport is a work in progress. The data, importer, CLI, "
            "and shader-source foundation have landed, but no spectral render path "
            "runs yet, so the flag is refused instead of silently rendering RGB."
        )


def apply_sppm_glossy_roughness(renderer, args) -> None:
    """Apply the parsed ``--sppm-glossy-roughness`` override onto ``renderer``.

    A ``None`` value (flag/env unset) leaves the override at its sentinel so the
    renderer falls back to the tuned built-in default; an explicit value (incl.
    ``0.0`` for PM-1 delta-only) sets ``renderer._sppm_glossy_roughness_override``.
    Call once, after the renderer is constructed and other CLI overrides applied.
    """
    v = getattr(args, "sppm_glossy_roughness", None)
    if v is not None:
        renderer._sppm_glossy_roughness_override = float(v)


def add_render_flags(
    parser: argparse.ArgumentParser,
    *,
    backend: bool = True,
    integrator: bool = True,
    execution: bool = True,
    walk: bool = True,
    proposals: bool = True,
    reuse: bool = True,
    lobe_samplers: bool = True,
    neural_handoff: bool = True,
    neural_trainer: bool = True,
    train_precision: bool = True,
    online_training: bool = True,
    encoding: bool = True,
    resolution: bool = True,
    spectral: bool = True,
) -> None:
    """Add the shared `--backend` / `--integrator` / `--execution-mode` /
    `--bdpt-walk` / `--proposals` / `--reuse` / `--lobe-samplers` /
    `--neural-handoff` / `--neural-trainer` / `--train-precision` /
    `--online-training` / `--encoding` flags to ``parser``. Each flag can be
    suppressed via its keyword in the rare case a front-end must omit it.
    ``resolution`` is suppressed by ``skinny-render`` (which defines its own
    ``--width``/``--height`` for offline output size) and ``skinny-web`` (which
    is out of scope for the render-area flags)."""
    if resolution:
        # Render-area pixel size. Defaults 640x480, with SKINNY_WIDTH /
        # SKINNY_HEIGHT env fallbacks (precedence flag > env > default). On
        # `skinny` these size the window and the GPU render target; on
        # `skinny-gui` they size the offscreen render area (the Qt window and
        # docks keep their own size).
        _w_default = _env_int("SKINNY_WIDTH")
        _h_default = _env_int("SKINNY_HEIGHT")
        parser.add_argument(
            "--width", type=int,
            default=640 if _w_default is None else _w_default,
            help="Render-area width in pixels (+ SKINNY_WIDTH env). Default 640. "
                 "On 'skinny' sizes the window and the GPU render target; on "
                 "'skinny-gui' sizes the offscreen render area (the Qt window "
                 "and docks keep their own size). Precedence: flag > env > "
                 "default.",
        )
        parser.add_argument(
            "--height", type=int,
            default=480 if _h_default is None else _h_default,
            help="Render-area height in pixels (+ SKINNY_HEIGHT env). Default "
                 "480. See --width.",
        )
    if backend:
        # `default=None` is a sentinel meaning "use env / persisted / auto",
        # resolved by skinny.backend_select.select_backend (precedence: explicit
        # flag > SKINNY_BACKEND env > persisted > auto). Same pattern as
        # --integrator's None default.
        parser.add_argument(
            "--backend", choices=("auto", "metal", "vulkan"), default=None,
            help="GPU backend, fixed for the session (+ SKINNY_BACKEND env). "
                 "'auto' (default) resolves to native 'metal' on a Metal-capable "
                 "Apple-Silicon host — the native backend is at full parity with "
                 "Vulkan (geometry, shaded color, windowed present) — and falls "
                 "back to 'vulkan' everywhere else. 'metal' forces the native "
                 "backend (errors if no Metal device); 'vulkan' forces MoltenVK "
                 "under Vulkan (the production path on every platform).",
        )
    if integrator:
        parser.add_argument(
            "--integrator", choices=("path", "bdpt", "sppm"), default=None,
            help="Light-transport integrator (default: 'path', or the persisted "
                 "value on the interactive front-ends). 'path' is the "
                 "unidirectional path tracer; 'bdpt' is the bidirectional path "
                 "tracer; 'sppm' is the Stochastic Progressive Photon Mapping "
                 "integrator (caustic-efficient, flat materials, wavefront-only — "
                 "requires --execution-mode wavefront). On the interactive "
                 "front-ends this sets the initial integrator and it remains "
                 "runtime-cycleable.",
        )
        # SPPM glossy-continue threshold (always present; only read under SPPM).
        parser.add_argument(
            "--sppm-glossy-roughness", type=float,
            default=_env_float("SKINNY_SPPM_GLOSSY_ROUGHNESS"),
            help="SPPM eye-walk: continue (reflect, don't gather) through a metallic "
                 "lobe whose roughness is below this threshold, so polished metals "
                 "reconstruct sharp reflections. Default (unset) uses the tuned built-in "
                 "(~0.5); 0 = PM-1 delta-only. SPPM + wavefront only.")
    if proposals:
        parser.add_argument(
            "--proposals",
            choices=("bsdf", "bsdf,env", "env", "bsdf,neural", "neural"),
            default=os.environ.get("SKINNY_PROPOSALS"),
            metavar="{bsdf,bsdf+env,env,bsdf+neural,neural}",
            help="Directional-proposal mixture at the BSDF bounce (+ "
                 "SKINNY_PROPOSALS env). 'bsdf' (default) is the material's own "
                 "importance sampler — bit-identical to the classic renderer; "
                 "'bsdf,env' MIS-mixes an environment-importance proposal "
                 "(lower variance on IBL); 'env' is env-only; 'bsdf,neural' "
                 "MIS-mixes the learned neural spline-flow proposal "
                 "(wavefront-only, flat materials). Runtime-selectable + "
                 "persisted on the interactive front-ends.",
        )
    if encoding:
        parser.add_argument(
            "--encoding", choices=("E0", "E1", "E3"),
            default=os.environ.get("SKINNY_ENCODING", "E0"),
            help="Conditioner positional encoding for the neural directional "
                 "proposal (axis 2; + SKINNY_ENCODING env). 'E0' (default) feeds "
                 "the raw condition — byte-identical to the shipped net; 'E1' "
                 "applies a NeRF-γ feature map to every condition scalar; 'E3' is "
                 "E1 plus the raw condition appended. Jacobian-free (only the "
                 "conditioner input changes — |J| and the pdf path are unchanged). "
                 "Must match the loaded network's encoding — a first-layer-width "
                 "mismatch is refused, not rendered mis-conditioned. Build dim "
                 "(recompiles the neural .spv). Persisted on the interactive "
                 "front-ends.",
        )
    if reuse:
        parser.add_argument(
            "--reuse", choices=("none",), default=os.environ.get("SKINNY_REUSE"),
            help="Reuse/resampling mode around direct + indirect lighting (+ "
                 "SKINNY_REUSE env). Only 'none' (stock NEE) ships today; "
                 "ReSTIR-style reservoir reuse is a future mode.",
        )
    if spectral:
        parser.add_argument(
            "--spectral", action="store_true", default=_env_flag("SKINNY_SPECTRAL"),
            help="Render spectrally — hero-wavelength transport (4 samples over "
                 "360–830 nm) with a CIE film resolve — instead of RGB. Chosen "
                 "once at startup and fixed for the session; kernels compile "
                 "spectral or RGB accordingly and it is not persisted (+ "
                 "SKINNY_SPECTRAL env). v1 scope is --integrator path under the "
                 "megakernel execution mode over flat materials; it is refused at "
                 "startup with bdpt/sppm, an explicit --execution-mode wavefront, "
                 "ReSTIR reuse, the neural proposal, or a skin/subsurface/volume "
                 "scene.",
        )
    if lobe_samplers:
        parser.add_argument(
            "--lobe-samplers", default=os.environ.get("SKINNY_LOBE_SAMPLERS"),
            metavar="coat=…,spec=…,diff=…",
            help="Per-lobe importance sampler for the flat/std_surface BSDF (+ "
                 "SKINNY_LOBE_SAMPLERS env). Comma-separated lobe=strategy terms; "
                 "unspecified lobes stay native. coat/spec accept 'native' "
                 "(spherical-cap VNDF) or 'basis' (Heitz-2018 VNDF); diff accepts "
                 "'native' (cosine) or 'uniform'. Runtime-selectable + persisted "
                 "on the interactive front-ends.",
        )
    if neural_handoff:
        parser.add_argument(
            "--neural-handoff", choices=("file", "interop", "shared"),
            default=os.environ.get("SKINNY_NEURAL_HANDOFF", "file"),
            help="Weight-handoff backend for online neural-proposal training (+ "
                 "SKINNY_NEURAL_HANDOFF env). 'file' (default) double-buffers "
                 "through an NFW1 file the renderer hot-reloads — a CPU "
                 "round-trip through disk that runs on any platform; 'shared' is "
                 "an in-process CPU double-buffer held in RAM — no disk write and "
                 "no CUDA/unified-memory device, any platform, while 'interop' "
                 "publishes weights GPU-side with no file round-trip — on Vulkan, "
                 "CUDA writes the exported weight buffer (VK_KHR_external_memory); "
                 "on the native Metal backend the unified-memory shared-storage "
                 "buffers are written in place at the frame boundary. 'interop' "
                 "raises on hosts with neither GPU path; 'file' and 'shared' "
                 "always work.",
        )
    if neural_trainer:
        parser.add_argument(
            "--neural-trainer", choices=("cpu", "cuda", "mlx", "auto"),
            default=os.environ.get("SKINNY_NEURAL_TRAINER", "auto"),
            help="Training-compute backend for online neural-proposal training "
                 "(+ SKINNY_NEURAL_TRAINER env). 'auto' (default) precedence is "
                 "cuda > mlx > cpu: the torch CUDA backend when torch + a CUDA "
                 "device are present, else Apple MLX on an Apple-Silicon Metal "
                 "host, else the torch-free numpy reference oracle. 'cpu' forces "
                 "the numpy reference (always available); 'cuda' forces torch on "
                 "CUDA (raises if torch/CUDA are absent); 'mlx' forces Apple MLX "
                 "on the Metal GPU (needs the '[mlx]' extra on Apple Silicon; "
                 "raises otherwise). Persisted on the interactive front-ends.",
        )
    if train_precision:
        parser.add_argument(
            "--train-precision", choices=("fp32", "fp16"),
            default=os.environ.get("SKINNY_TRAIN_PRECISION", "fp32"),
            help="Optimizer compute precision for online training (+ "
                 "SKINNY_TRAIN_PRECISION env), independent of the inference "
                 "precision. 'fp32' (default) trains in full precision; 'fp16' "
                 "uses the framework's mixed precision where the device supports "
                 "it (torch autocast on CUDA; float16 compute over fp32 masters "
                 "on Apple MLX) and falls back to fp32 otherwise. "
                 "Training always bakes fp32 weights, so the on-disk/handoff "
                 "format is unchanged. Persisted on the interactive front-ends.",
        )
    if online_training:
        parser.add_argument(
            "--online-training", action="store_true",
            default=_env_flag("SKINNY_ONLINE_TRAINING"),
            help="Enable the online neural-proposal training loop on the "
                 "interactive front-ends (+ SKINNY_ONLINE_TRAINING env). Requires "
                 "--execution-mode wavefront AND a neural proposal in the mixture "
                 "(--proposals …,neural); a missing prerequisite is refused with a "
                 "clear message, not a silent no-op. Off by default (behavior is "
                 "byte-identical to today). Persisted on the interactive "
                 "front-ends like --neural-handoff.",
        )
    if execution:
        parser.add_argument(
            "--execution-mode", choices=("auto", "megakernel", "wavefront"),
            default=os.environ.get("SKINNY_EXECUTION_MODE", "auto"),
            help="GPU execution backend, fixed for the session (+ "
                 "SKINNY_EXECUTION_MODE env). 'auto' (default) derives the mode "
                 "from the startup integrator — path/bdpt → megakernel, sppm → "
                 "wavefront (mirroring --backend auto). 'megakernel' is the "
                 "single main_pass dispatch; 'wavefront' is the staged "
                 "per-material backend. An explicit megakernel/wavefront (flag "
                 "or env) overrides the derived default. Only the selected "
                 "backend is compiled.",
        )
    if walk:
        # Free string (not `choices=`) so the deprecated `megakernel` alias is
        # accepted; `resolve_walk` normalizes + validates it.
        parser.add_argument(
            "--bdpt-walk", default=os.environ.get("SKINNY_BDPT_WALK", "fused"),
            metavar="{fused,eye,eye_light}",
            help="Subpath-build strategy for wavefront + bdpt only — no effect "
                 "otherwise (+ SKINNY_BDPT_WALK env). 'fused' (default) builds "
                 "both subpaths in one kernel + the connect counting-sort "
                 "(fastest); 'eye' stages the eye walk into per-bounce "
                 "dispatches; 'eye_light' also stages the light walk + splat. "
                 "All produce the identical image — this trades dispatch "
                 "overhead vs occupancy.",
        )


def resolve_encoding(value: str | None):
    """Map an ``--encoding`` string (``E0``/``E1``/``E3``, case-insensitive) to a
    :class:`skinny.sampling.neural_weights.Encoding`. ``None``/empty → ``E0``."""
    from skinny.sampling.neural_weights import Encoding

    return Encoding((value or "E0").upper())


def neural_config_from_args(args, *, base=None):
    """Build the :class:`NeuralBuildConfig` for the neural directional proposal
    from parsed render flags, or return ``base`` when every neural build axis is at
    its default — so a default invocation keeps emitting NO ``-D`` flags and stays
    byte-identical to the shipped proposal.

    Today only the conditioner ``--encoding`` (axis 2) is wired through here; the
    size/precision axes are constructed programmatically. ``base`` (default
    ``None``) lets a caller thread a persisted/restored config in as the starting
    point — its non-encoding axes are preserved.
    """
    from dataclasses import replace

    from skinny.sampling.neural_weights import Encoding, NeuralBuildConfig

    enc = resolve_encoding(getattr(args, "encoding", None))
    if base is None:
        # Default (E0) and no base → keep the renderer's own default config.
        return None if enc is Encoding.E0 else NeuralBuildConfig(encoding=enc)
    return replace(base, encoding=enc)
