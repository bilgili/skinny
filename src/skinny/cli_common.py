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
    integrator: bool = True,
    execution: bool = True,
    walk: bool = True,
    proposals: bool = True,
    reuse: bool = True,
) -> None:
    """Add the shared `--integrator` / `--execution-mode` / `--bdpt-walk` /
    `--proposals` / `--reuse` flags to ``parser``. Each flag can be suppressed
    via its keyword in the rare case a front-end must omit it (none currently
    does)."""
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
            "--proposals", choices=("bsdf", "bsdf,env", "env"),
            default=os.environ.get("SKINNY_PROPOSALS"),
            metavar="{bsdf,bsdf+env,env}",
            help="Directional-proposal mixture at the BSDF bounce (+ "
                 "SKINNY_PROPOSALS env). 'bsdf' (default) is the material's own "
                 "importance sampler — bit-identical to the classic renderer; "
                 "'bsdf,env' MIS-mixes an environment-importance proposal "
                 "(lower variance on IBL); 'env' is env-only. Runtime-selectable "
                 "+ persisted on the interactive front-ends.",
        )
    if reuse:
        parser.add_argument(
            "--reuse", choices=("none",), default=os.environ.get("SKINNY_REUSE"),
            help="Reuse/resampling mode around direct + indirect lighting (+ "
                 "SKINNY_REUSE env). Only 'none' (stock NEE) ships today; "
                 "ReSTIR-style reservoir reuse is a future mode.",
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
