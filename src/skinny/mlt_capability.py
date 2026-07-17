"""Single source of truth for whether the MLT integrator is wired.

``--integrator mlt``, its startup refusals (``cli_common``), and the parity
matrix's integrator axis (``pbrt.parity``) all key off this one constant —
the same pattern as :mod:`skinny.spectral_capability`.

The Metropolis Light Transport integrator (change ``mlt-integrator``) is
PSSMLT — Kelemen primary-sample-space Metropolis driving the existing
wavefront BDPT strategies, matching pbrt-v4's ``MLTIntegrator``. While the
GPU transport (sampler, bootstrap, mutation/splat sequence) is not wired,
``MLT_IMPLEMENTED`` stays ``False``: ``--integrator mlt`` is refused with a
clear error instead of silently rendering another integrator, and the parity
matrix records ``(mlt, wavefront)`` combos as "not yet wired" skips.

Flip to ``True`` together with the wavefront MLT transport (tasks group 5).
"""

from __future__ import annotations

MLT_IMPLEMENTED = False
