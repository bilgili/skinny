"""The shared, staged renderer bring-up sequence — one owner for all front-ends.

Every front-end (``skinny`` → :mod:`skinny.app`, ``skinny-gui`` →
:mod:`skinny.ui.qt.app`, ``skinny-render`` → :mod:`skinny.headless`,
``skinny-web`` → :mod:`skinny.web_app`) has to turn parsed render flags into a
validated ``(backend, execution mode, startup integrator)`` triple and then a
``(context, Renderer)`` pair. That orchestration used to be copied four times
and had already drifted (two different guard orders, two different guard
subsets, four hand-rolled ``RuntimeError`` → ``SystemExit`` wrappers). This
module owns it once; a new refusal guard added here is effective on all four
without per-front-end edits.

**Staged**, because two front-ends cannot construct the context where they
resolve the flags: the Qt GUI builds it on its render thread and the web server
builds one per session on a background thread.

1. :func:`plan_bringup` — resolution + every refusal guard, in one canonical
   order, returning a frozen :class:`BringupPlan`.
2. :meth:`BringupPlan.create` — the GPU context plus the ``Renderer``, with
   destroy-on-failure. Callable later, on another thread, with no guard re-run.

**Canonical order** (``startup_integrator_name`` → ``resolve_execution_mode`` →
``validate_render_flags`` → ``reject_sppm_without_wavefront`` →
``reject_mlt_unsupported`` → ``reject_spectral_unsupported`` →
``reject_mcp_unsupported`` → ``select_backend``) matches the
``resolve_execution_mode`` docstring and the ``render-cli`` spec's "validation
SHALL run after the execution mode is resolved". It is a superset of both
pre-existing orders: the non-interactive pair already resolved first, and the
interactive pair's pre-resolution ``validate_render_flags`` was a no-op for the
megakernel refusals anyway (it saw the unresolved string ``"auto"``, which
:func:`skinny.cli_common._envelope_mode` maps to ``"wavefront"``), with the
explicit ``reject_*`` re-checks doing the real work. ``tests/test_bringup.py``
proves the equivalence against both transcribed legacy sequences.

**Persistence is the caller's one knob.** ``persisted=None`` (``skinny-render``,
``skinny-web``, which do not persist) means CLI + environment only.  Passing the
settings dict (``skinny``, ``skinny-gui``) lets the persisted integrator feed
startup-integrator resolution, the persisted backend feed
:func:`~skinny.backend_select.select_backend`, and the persisted ``encoding``
feed the neural build config. The flag > env > persisted > auto precedence
itself is unchanged — it lives inside those resolvers already; this module only
decides *whether* the persisted value is offered.

The builder ends at a constructed ``(ctx, renderer)``: post-construction
renderer state (``skinny``'s persisted overrides, integrator / reuse indices,
lobe samplers) stays at the call sites.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from skinny.backend_select import make_context, select_backend
from skinny.cli_common import (
    neural_config_from_args,
    reject_mcp_unsupported,
    reject_mlt_unsupported,
    reject_spectral_unsupported,
    reject_sppm_without_wavefront,
    resolve_execution_mode,
    resolve_walk,
    startup_integrator_name,
    validate_render_flags,
)

#: Persisted ``encoding`` values the interactive front-ends accept on restore.
#: A value outside this set (a hand-edited or stale ``settings.json``) is
#: ignored in favour of the flag/env value, as before this module existed.
ENCODING_CHOICES = ("E0", "E1", "E3")


@dataclass(frozen=True)
class BringupPlan:
    """A validated bring-up: everything the guards resolved, plus
    :meth:`create`.

    The fields are exactly the guard-vetted inputs that are the same on every
    front-end. Front-end-specific ``Renderer`` constructor inputs (scene path,
    asset directories, neural handoff/trainer/precision) are *not* here — they
    are forwarded verbatim through :meth:`create`'s ``**renderer_kwargs``.
    """

    prog: str
    backend: str
    execution_mode: str
    startup_integrator: str
    spectral: bool
    bdpt_walk: str
    encoding: str | None
    neural_config: object | None

    def create(
        self,
        *,
        window=None,
        width: int = 1280,
        height: int = 720,
        gpu_preference: str | None = None,
        context_factory=make_context,
        renderer_factory=None,
        **renderer_kwargs,
    ):
        """Construct ``(ctx, renderer)`` from this plan. No guard is re-run.

        ``context_factory`` and ``renderer_factory`` are the hostless-test
        seams: the defaults are :func:`~skinny.backend_select.make_context` and
        :class:`skinny.renderer.Renderer`, and ``Renderer`` is imported lazily
        so a test with a stub factory never pulls in the ``vulkan`` extension
        that :mod:`skinny.renderer` imports at module load.

        The context is destroyed if renderer construction raises, so a failed
        bring-up never leaks a live GPU context (the rule
        ``metal-dispatch-hygiene`` makes structural on Metal). The handler
        catches ``BaseException``, not ``Exception``: renderer construction
        compiles shaders and can run for seconds, so a Ctrl-C landing inside it
        is an ordinary event — and on Metal a leaked context is not merely
        untidy, it wedges GPU capacity until reboot.
        """
        ctx = context_factory(
            self.backend, window=window, width=width, height=height,
            gpu_preference=gpu_preference,
        )
        if renderer_factory is None:
            from skinny.renderer import Renderer

            renderer_factory = Renderer
        try:
            renderer = renderer_factory(
                vk_ctx=ctx,
                execution_mode=self.execution_mode,
                bdpt_walk=self.bdpt_walk,
                neural_config=self.neural_config,
                spectral=self.spectral,
                **renderer_kwargs,
            )
        except BaseException:
            ctx.destroy()
            raise
        return ctx, renderer


def _resolve_encoding_value(args, persisted: dict | None) -> str | None:
    """``--encoding`` with the interactive persisted restore applied.

    A build dimension, so it is resolved once at plan time and baked into
    :attr:`BringupPlan.neural_config`. CLI/env wins; otherwise a front-end that
    persists restores its saved value. ``persisted=None`` reproduces the
    non-interactive behaviour (flag/env only).
    """
    value = getattr(args, "encoding", None)
    if persisted is None:
        return value
    if "--encoding" in sys.argv or os.environ.get("SKINNY_ENCODING"):
        return value
    saved = persisted.get("encoding")
    return saved if saved in ENCODING_CHOICES else value


def plan_bringup(args, prog: str, persisted: dict | None = None) -> BringupPlan:
    """Run the canonical resolve-and-refuse sequence; return a
    :class:`BringupPlan`.

    ``args`` is the front-end's parsed ``Namespace`` (flags from
    :func:`skinny.cli_common.add_render_flags`; attributes a front-end
    suppresses are simply absent and default). ``prog`` is the front-end's
    program name, used as the ``SystemExit`` prefix for a backend-selection
    failure — the guard refusals carry their own ``skinny:`` prefix, unchanged.
    ``persisted`` is the loaded settings dict on the front-ends that persist,
    else ``None``.

    ``args.execution_mode`` is overwritten with the resolved concrete mode, as
    all four front-ends already did — front-ends that read it afterwards keep
    working. Raises ``SystemExit`` on any refused combination.
    """
    settings = persisted or {}
    startup_integrator = startup_integrator_name(
        getattr(args, "integrator", None),
        settings.get("params", {}).get("integrator_index"),
    )
    args.execution_mode = resolve_execution_mode(
        getattr(args, "execution_mode", None), startup_integrator)

    validate_render_flags(args)
    reject_sppm_without_wavefront(startup_integrator, args.execution_mode)
    reject_mlt_unsupported(
        startup_integrator, args.execution_mode,
        spectral=bool(getattr(args, "spectral", False)),
        proposals=getattr(args, "proposals", None),
        reuse=getattr(args, "reuse", None),
        online_training=bool(getattr(args, "online_training", False)),
    )
    reject_spectral_unsupported(
        getattr(args, "spectral", False), startup_integrator, args.execution_mode,
        getattr(args, "proposals", None), getattr(args, "reuse", None),
    )
    reject_mcp_unsupported(bool(getattr(args, "mcp", False)))

    # The remaining fallible plan input. `--bdpt-walk` has no argparse
    # `choices=` (it accepts a deprecated alias), and its default comes from
    # SKINNY_BDPT_WALK — which argparse does not validate — so a bad value only
    # surfaces here. Resolve it before the GPU probe below, not after: every way
    # a launch can be refused should be exhausted before a Metal device is
    # constructed.
    bdpt_walk = resolve_walk(getattr(args, "bdpt_walk", "fused"))

    # Last, so every flag-level refusal happens before the GPU probe.
    try:
        backend = select_backend(
            getattr(args, "backend", None), persisted=settings.get("backend"))
    except RuntimeError as exc:
        raise SystemExit(f"{prog}: {exc}")

    encoding = _resolve_encoding_value(args, persisted)
    args.encoding = encoding
    return BringupPlan(
        prog=prog,
        backend=backend,
        execution_mode=args.execution_mode,
        startup_integrator=startup_integrator,
        spectral=bool(getattr(args, "spectral", False)),
        bdpt_walk=bdpt_walk,
        encoding=encoding,
        neural_config=neural_config_from_args(args),
    )
