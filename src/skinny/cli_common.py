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

# path → integrator_index, mirroring the renderer's integrator ordering.
INTEGRATOR_INDEX = {"path": 0, "bdpt": 1}

# Advertised walk choices. `megakernel` is accepted as a deprecated alias but is
# not listed, so only the execution axis owns that word.
WALK_CHOICES = ("fused", "eye", "eye_light")
_WALK_ALIASES = {"megakernel": "fused"}


def _env_flag(name: str) -> bool:
    """Truthy parse of a boolean env var for a ``store_true`` flag default —
    '1'/'true'/'yes'/'on' (case-insensitive) enable it; unset/empty is False."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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
) -> None:
    """Add the shared `--backend` / `--integrator` / `--execution-mode` /
    `--bdpt-walk` / `--proposals` / `--reuse` / `--lobe-samplers` /
    `--neural-handoff` / `--neural-trainer` / `--train-precision` /
    `--online-training` flags to ``parser``. Each flag can be suppressed via its
    keyword in the rare case a front-end must omit it (none currently does)."""
    if backend:
        # `default=None` is a sentinel meaning "use env / persisted / auto",
        # resolved by skinny.backend_select.select_backend (precedence: explicit
        # flag > SKINNY_BACKEND env > persisted > auto). Same pattern as
        # --integrator's None default.
        parser.add_argument(
            "--backend", choices=("auto", "metal", "vulkan"), default=None,
            help="GPU backend, fixed for the session (+ SKINNY_BACKEND env). "
                 "'auto' (default) resolves to 'vulkan' in this build: native "
                 "Metal is in a foundation phase — the device + a trivial compute "
                 "dispatch + present are proven, but the full renderer is not yet "
                 "ported to Metal, so '--backend metal' on a real front-end "
                 "reports that full rendering lands in a later phase and exits. "
                 "'vulkan' is byte-identical to today (MoltenVK under Vulkan on "
                 "macOS).",
        )
    if integrator:
        parser.add_argument(
            "--integrator", choices=("path", "bdpt"), default=None,
            help="Light-transport integrator (default: 'path', or the persisted "
                 "value on the interactive front-ends). 'path' is the "
                 "unidirectional path tracer; 'bdpt' is the bidirectional path "
                 "tracer. On the interactive front-ends this sets the initial "
                 "integrator and it remains runtime-cycleable.",
        )
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
    if reuse:
        parser.add_argument(
            "--reuse", choices=("none",), default=os.environ.get("SKINNY_REUSE"),
            help="Reuse/resampling mode around direct + indirect lighting (+ "
                 "SKINNY_REUSE env). Only 'none' (stock NEE) ships today; "
                 "ReSTIR-style reservoir reuse is a future mode.",
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
            "--neural-handoff", choices=("file", "interop"),
            default=os.environ.get("SKINNY_NEURAL_HANDOFF", "file"),
            help="Weight-handoff backend for online neural-proposal training (+ "
                 "SKINNY_NEURAL_HANDOFF env). 'file' (default) double-buffers "
                 "through an NFW1 file the renderer hot-reloads — a CPU "
                 "round-trip that runs on any platform; 'interop' writes weights "
                 "straight into the Vulkan-exported buffer from CUDA "
                 "(VK_KHR_external_memory) with no round-trip — the real-time "
                 "path, CUDA-only (raises on platforms without it).",
        )
    if neural_trainer:
        parser.add_argument(
            "--neural-trainer", choices=("cpu", "cuda", "mlx", "auto"),
            default=os.environ.get("SKINNY_NEURAL_TRAINER", "auto"),
            help="Training-compute backend for online neural-proposal training "
                 "(+ SKINNY_NEURAL_TRAINER env). 'auto' (default) picks the torch "
                 "CUDA backend when torch + a CUDA device are present, else the "
                 "torch-free numpy reference oracle. 'cpu' forces the numpy "
                 "reference (always available); 'cuda' forces torch on CUDA "
                 "(raises if torch/CUDA are absent); 'mlx' is reserved for a "
                 "later change. Persisted on the interactive front-ends.",
        )
    if train_precision:
        parser.add_argument(
            "--train-precision", choices=("fp32", "fp16"),
            default=os.environ.get("SKINNY_TRAIN_PRECISION", "fp32"),
            help="Optimizer compute precision for online training (+ "
                 "SKINNY_TRAIN_PRECISION env), independent of the inference "
                 "precision. 'fp32' (default) trains in full precision; 'fp16' "
                 "uses the framework's mixed precision where the device supports "
                 "it (torch autocast on CUDA) and falls back to fp32 otherwise. "
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
            "--execution-mode", choices=("megakernel", "wavefront"),
            default=os.environ.get("SKINNY_EXECUTION_MODE", "megakernel"),
            help="GPU execution backend, fixed for the session (+ "
                 "SKINNY_EXECUTION_MODE env). 'megakernel' (default) is the "
                 "single main_pass dispatch; 'wavefront' is the staged "
                 "per-material backend (Vulkan only — pinned to megakernel on "
                 "Metal). Only the selected backend is compiled.",
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
